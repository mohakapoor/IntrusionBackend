"""
pcap_to_cicids.py
=================
Converts a PCAP file into a CICIDS2017-style CSV file.

CICIDS2017 features extracted (78 features + Label):
    - Flow identifiers: src/dst IP, src/dst port, protocol
    - Duration, packet counts, byte counts
    - Inter-arrival time (IAT) statistics for forward/backward
    - Packet-length statistics (mean, std, min, max) for fwd/bwd
    - TCP flag counts
    - Flow rates (bytes/s, packets/s)
    - Active/idle time stats
    - Header lengths, bulk rates, subflow stats, and more

Usage:
    python pcap_to_cicids.py --input sample.pcap --output output.csv

Dependencies: Python stdlib + pandas + numpy  (no scapy required)
"""

import struct
import socket
import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


PCAP_GLOBAL_HEADER_FMT = "<IHHiIII"   # magic, ver_major, ver_minor, thiszone,
                                        # sigfigs, snaplen, network
PCAP_PKT_HEADER_FMT    = "<IIII"       # ts_sec, ts_usec, incl_len, orig_len

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW_IP   = 101
LINKTYPE_NULL     = 0


def _read_pcap(path: str):
    """
    Yield (timestamp_us, raw_ip_bytes) for every packet in the PCAP.
    Returns the link-type as first yield (sentinel = None).
    """
    with open(path, "rb") as fh:
        gh = fh.read(struct.calcsize(PCAP_GLOBAL_HEADER_FMT))
        if len(gh) < struct.calcsize(PCAP_GLOBAL_HEADER_FMT):
            raise ValueError("File too short to be a PCAP")

        magic, vmaj, vmin, tz, sigfigs, snaplen, network = struct.unpack(
            PCAP_GLOBAL_HEADER_FMT, gh
        )
        if magic not in (0xA1B2C3D4, 0xD4C3B2A1):
            raise ValueError(f"Not a PCAP file (magic=0x{magic:08X})")

        pkt_hdr_size = struct.calcsize(PCAP_PKT_HEADER_FMT)
        while True:
            raw = fh.read(pkt_hdr_size)
            if not raw:
                break
            if len(raw) < pkt_hdr_size:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                PCAP_PKT_HEADER_FMT, raw
            )
            data = fh.read(incl_len)
            ts_us = ts_sec * 1_000_000 + ts_usec

            # Strip link-layer header to reach raw IP
            if network == LINKTYPE_ETHERNET:
                ip_bytes = data[14:] if len(data) > 14 else b""
            elif network == LINKTYPE_NULL:
                ip_bytes = data[4:]  if len(data) > 4  else b""
            else:                              # assume raw IP
                ip_bytes = data

            yield ts_us, ip_bytes


@dataclass
class Packet:
    ts_us:    int
    src_ip:   str
    dst_ip:   str
    src_port: int
    dst_port: int
    proto:    int            # 6=TCP, 17=UDP, etc.
    ip_len:   int            # total IP datagram length
    payload_len: int         # transport payload length
    tcp_flags:   int = 0     # raw flags byte (only for TCP)
    fwd:         bool = True  # filled in by FlowTable


def _parse_ip(ts_us: int, raw: bytes) -> Optional[Packet]:
    if len(raw) < 20:
        return None
    ver = (raw[0] >> 4)
    if ver != 4:            # IPv6 – skip for now
        return None

    ihl    = (raw[0] & 0x0F) * 4
    ip_len = struct.unpack("!H", raw[2:4])[0]
    proto  = raw[9]
    src_ip = socket.inet_ntoa(raw[12:16])
    dst_ip = socket.inet_ntoa(raw[16:20])

    transport = raw[ihl:]
    src_port = dst_port = 0
    tcp_flags = 0
    payload_len = 0

    if proto == 6 and len(transport) >= 20:    # TCP
        src_port  = struct.unpack("!H", transport[0:2])[0]
        dst_port  = struct.unpack("!H", transport[2:4])[0]
        data_off  = (transport[12] >> 4) * 4
        tcp_flags = transport[13]
        payload_len = max(0, len(transport) - data_off)
    elif proto == 17 and len(transport) >= 8:  # UDP
        src_port  = struct.unpack("!H", transport[0:2])[0]
        dst_port  = struct.unpack("!H", transport[2:4])[0]
        payload_len = max(0, len(transport) - 8)

    return Packet(
        ts_us=ts_us, src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port, proto=proto,
        ip_len=ip_len, payload_len=payload_len, tcp_flags=tcp_flags,
    )



FlowKey = Tuple[str, str, int, int, int]   # src_ip, dst_ip, sport, dport, proto

FLOW_TIMEOUT_US  = 120 * 1_000_000   # 2 min activity timeout
ACTIVE_TIMEOUT_US =  60 * 1_000_000  # 1 min active timeout


@dataclass
class FlowStats:
    # identifiers
    src_ip:   str = ""
    dst_ip:   str = ""
    src_port: int = 0
    dst_port: int = 0
    proto:    int = 0

    # timestamps
    start_ts: int = 0
    last_ts:  int = 0

    # packet lists (fwd = initiator direction)
    fwd_pkts:  List[int] = field(default_factory=list)  # ip_len each packet
    bwd_pkts:  List[int] = field(default_factory=list)
    fwd_iats:  List[int] = field(default_factory=list)  # microseconds
    bwd_iats:  List[int] = field(default_factory=list)
    flow_iats: List[int] = field(default_factory=list)

    fwd_hdr_bytes: int = 0
    bwd_hdr_bytes: int = 0

    # TCP flags aggregated
    fwd_fin: int = 0; fwd_syn: int = 0; fwd_rst: int = 0
    fwd_psh: int = 0; fwd_ack: int = 0; fwd_urg: int = 0
    bwd_fin: int = 0; bwd_syn: int = 0; bwd_rst: int = 0
    bwd_psh: int = 0; bwd_ack: int = 0; bwd_urg: int = 0

    # active/idle
    last_active_start: int = 0
    active_times: List[int] = field(default_factory=list)
    idle_times:   List[int] = field(default_factory=list)
    prev_last_ts:  int = 0


def _safe_stats(vals: List[float]):
    """Return (mean, std, max, min) or zeros."""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def _flow_to_row(fs: FlowStats, label: str = "BENIGN") -> Dict:
    dur_us  = max(1, fs.last_ts - fs.start_ts)
    dur_s   = dur_us / 1_000_000

    all_pkts = fs.fwd_pkts + fs.bwd_pkts
    tot_pkts = len(all_pkts)
    tot_bytes = sum(all_pkts)

    fw = len(fs.fwd_pkts); bw = len(fs.bwd_pkts)
    fb = sum(fs.fwd_pkts); bb = sum(fs.bwd_pkts)

    # packet-length stats
    fl_mean, fl_std, fl_max, fl_min = _safe_stats(all_pkts)
    fw_mean, fw_std, fw_max, fw_min = _safe_stats(fs.fwd_pkts)
    bw_mean, bw_std, bw_max, bw_min = _safe_stats(fs.bwd_pkts)

    # IAT stats
    fi_mean, fi_std, fi_max, fi_min = _safe_stats(fs.flow_iats)
    fwi_mean, fwi_std, fwi_max, fwi_min = _safe_stats(fs.fwd_iats)
    bwi_mean, bwi_std, bwi_max, bwi_min = _safe_stats(fs.bwd_iats)

    # rates
    flow_bps  = tot_bytes / dur_s  if dur_s > 0 else 0
    flow_pps  = tot_pkts  / dur_s  if dur_s > 0 else 0
    fwd_bps   = fb / dur_s         if dur_s > 0 else 0
    bwd_bps   = bb / dur_s         if dur_s > 0 else 0

    # active / idle
    act_mean, act_std, act_max, act_min = _safe_stats(fs.active_times)
    idl_mean, idl_std, idl_max, idl_min = _safe_stats(fs.idle_times)

    # subflow (simple: treat whole flow as one subflow)
    sf_fwd_pkts  = fw
    sf_fwd_bytes = fb
    sf_bwd_pkts  = bw
    sf_bwd_bytes = bb

    # bulk stats (simplified: avg bytes per active burst)
    fwd_bulk_rate = fb / max(1, len(fs.active_times)) if fs.active_times else 0
    bwd_bulk_rate = bb / max(1, len(fs.active_times)) if fs.active_times else 0

    # PSH/URG counts
    fwd_psh_flags = fs.fwd_psh; bwd_psh_flags = fs.bwd_psh
    fwd_urg_flags = fs.fwd_urg; bwd_urg_flags = fs.bwd_urg

    # min/max segment size (approx from packet lengths)
    fwd_seg_size = fw_mean
    bwd_seg_size = bw_mean

    # header length estimates (TCP=20, UDP=8)
    proto_hdr = 20 if fs.proto == 6 else 8
    fwd_hdr_len = proto_hdr * fw
    bwd_hdr_len = proto_hdr * bw

    # down/up ratio
    down_up = bb / max(1, fb)

    row = {
        " Source IP":          fs.src_ip,
        " Source Port":        fs.src_port,
        " Destination IP":     fs.dst_ip,
        " Destination Port":   fs.dst_port,
        " Protocol":           fs.proto,
        " Flow Duration":      dur_us,
        " Total Fwd Packets":  fw,
        " Total Backward Packets": bw,
        "Total Length of Fwd Packets": fb,
        " Total Length of Bwd Packets": bb,
        " Fwd Packet Length Max": fw_max,
        " Fwd Packet Length Min": fw_min,
        " Fwd Packet Length Mean": fw_mean,
        " Fwd Packet Length Std": fw_std,
        "Bwd Packet Length Max": bw_max,
        " Bwd Packet Length Min": bw_min,
        " Bwd Packet Length Mean": bw_mean,
        " Bwd Packet Length Std": bw_std,
        "Flow Bytes/s":        flow_bps,
        " Flow Packets/s":     flow_pps,
        " Flow IAT Mean":      fi_mean,
        " Flow IAT Std":       fi_std,
        " Flow IAT Max":       fi_max,
        " Flow IAT Min":       fi_min,
        "Fwd IAT Total":       sum(fs.fwd_iats),
        " Fwd IAT Mean":       fwi_mean,
        " Fwd IAT Std":        fwi_std,
        " Fwd IAT Max":        fwi_max,
        " Fwd IAT Min":        fwi_min,
        "Bwd IAT Total":       sum(fs.bwd_iats),
        " Bwd IAT Mean":       bwi_mean,
        " Bwd IAT Std":        bwi_std,
        " Bwd IAT Max":        bwi_max,
        " Bwd IAT Min":        bwi_min,
        "Fwd PSH Flags":       fwd_psh_flags,
        " Bwd PSH Flags":      bwd_psh_flags,
        " Fwd URG Flags":      fwd_urg_flags,
        " Bwd URG Flags":      bwd_urg_flags,
        " Fwd Header Length":  fwd_hdr_len,
        " Bwd Header Length":  bwd_hdr_len,
        "Fwd Packets/s":       fw / dur_s if dur_s > 0 else 0,
        " Bwd Packets/s":      bw / dur_s if dur_s > 0 else 0,
        " Min Packet Length":  fl_min,
        " Max Packet Length":  fl_max,
        " Packet Length Mean": fl_mean,
        " Packet Length Std":  fl_std,
        " Packet Length Variance": fl_std ** 2,
        "FIN Flag Count":      fs.fwd_fin + fs.bwd_fin,
        " SYN Flag Count":     fs.fwd_syn + fs.bwd_syn,
        " RST Flag Count":     fs.fwd_rst + fs.bwd_rst,
        " PSH Flag Count":     fs.fwd_psh + fs.bwd_psh,
        " ACK Flag Count":     fs.fwd_ack + fs.bwd_ack,
        " URG Flag Count":     fs.fwd_urg + fs.bwd_urg,
        " CWE Flag Count":     0,
        " ECE Flag Count":     0,
        " Down/Up Ratio":      down_up,
        " Average Packet Size": fl_mean,
        " Avg Fwd Segment Size": fwd_seg_size,
        " Avg Bwd Segment Size": bwd_seg_size,
        " Fwd Header Length.1": fwd_hdr_len,
        "Fwd Avg Bytes/Bulk":  fwd_bulk_rate,
        " Fwd Avg Packets/Bulk": fw / max(1, len(fs.active_times)) if fs.active_times else fw,
        " Fwd Avg Bulk Rate":  fwd_bulk_rate,
        " Bwd Avg Bytes/Bulk": bwd_bulk_rate,
        " Bwd Avg Packets/Bulk": bw / max(1, len(fs.active_times)) if fs.active_times else bw,
        "Bwd Avg Bulk Rate":   bwd_bulk_rate,
        "Subflow Fwd Packets": sf_fwd_pkts,
        " Subflow Fwd Bytes":  sf_fwd_bytes,
        " Subflow Bwd Packets": sf_bwd_pkts,
        " Subflow Bwd Bytes":  sf_bwd_bytes,
        "Init_Win_bytes_forward":  0,
        " Init_Win_bytes_backward": 0,
        " act_data_pkt_fwd":   fw,
        " min_seg_size_forward": fw_min,
        "Active Mean":         act_mean,
        " Active Std":         act_std,
        " Active Max":         act_max,
        " Active Min":         act_min,
        "Idle Mean":           idl_mean,
        " Idle Std":           idl_std,
        " Idle Max":           idl_max,
        " Idle Min":           idl_min,
        " Label":              label,
    }
    return row


class FlowTable:
    def __init__(self):
        self._flows: Dict[FlowKey, FlowStats] = {}
        self._completed: List[Dict] = []

    def _key(self, pkt: Packet) -> Tuple[FlowKey, bool]:
        """Return (canonical_key, is_forward)."""
        k_fwd = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.proto)
        k_bwd = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.proto)
        if k_fwd in self._flows:
            return k_fwd, True
        if k_bwd in self._flows:
            return k_bwd, False
        return k_fwd, True   # new flow

    def _update_active_idle(self, fs: FlowStats, ts: int):
        ACTIVE_THRESHOLD = 1 * 1_000_000   # 1-second gap = idle
        if fs.prev_last_ts > 0:
            gap = ts - fs.prev_last_ts
            if gap > ACTIVE_THRESHOLD:
                active_dur = fs.prev_last_ts - fs.last_active_start
                if active_dur > 0:
                    fs.active_times.append(active_dur)
                fs.idle_times.append(gap)
                fs.last_active_start = ts
        fs.prev_last_ts = ts

    def process(self, pkt: Packet, label: str = "BENIGN"):
        key, is_fwd = self._key(pkt)

        if key not in self._flows:
            fs = FlowStats(
                src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                proto=pkt.proto,
                start_ts=pkt.ts_us, last_ts=pkt.ts_us,
                last_active_start=pkt.ts_us, prev_last_ts=pkt.ts_us,
            )
            self._flows[key] = fs
        else:
            fs = self._flows[key]

        # timeout check
        if pkt.ts_us - fs.last_ts > FLOW_TIMEOUT_US:
            self._complete(key, label)
            fs = FlowStats(
                src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                proto=pkt.proto,
                start_ts=pkt.ts_us, last_ts=pkt.ts_us,
                last_active_start=pkt.ts_us, prev_last_ts=pkt.ts_us,
            )
            self._flows[key] = fs
            is_fwd = True

        # IAT
        if fs.last_ts > 0:
            iat = pkt.ts_us - fs.last_ts
            fs.flow_iats.append(iat)
            if is_fwd:
                if fs.fwd_pkts:
                    fs.fwd_iats.append(iat)
            else:
                if fs.bwd_pkts:
                    fs.bwd_iats.append(iat)

        self._update_active_idle(fs, pkt.ts_us)
        fs.last_ts = pkt.ts_us

        # accumulate
        if is_fwd:
            fs.fwd_pkts.append(pkt.ip_len)
        else:
            fs.bwd_pkts.append(pkt.ip_len)

        # TCP flags
        if pkt.proto == 6:
            f = pkt.tcp_flags
            if is_fwd:
                fs.fwd_fin += (f >> 0) & 1
                fs.fwd_syn += (f >> 1) & 1
                fs.fwd_rst += (f >> 2) & 1
                fs.fwd_psh += (f >> 3) & 1
                fs.fwd_ack += (f >> 4) & 1
                fs.fwd_urg += (f >> 5) & 1
            else:
                fs.bwd_fin += (f >> 0) & 1
                fs.bwd_syn += (f >> 1) & 1
                fs.bwd_rst += (f >> 2) & 1
                fs.bwd_psh += (f >> 3) & 1
                fs.bwd_ack += (f >> 4) & 1
                fs.bwd_urg += (f >> 5) & 1

        # TCP FIN/RST → close flow
        if pkt.proto == 6 and (pkt.tcp_flags & 0x01 or pkt.tcp_flags & 0x04):
            self._complete(key, label)

    def _complete(self, key: FlowKey, label: str):
        fs = self._flows.pop(key, None)
        if fs and (fs.fwd_pkts or fs.bwd_pkts):
            # finalise active time
            if fs.prev_last_ts > fs.last_active_start:
                fs.active_times.append(fs.prev_last_ts - fs.last_active_start)
            self._completed.append(_flow_to_row(fs, label))

    def flush(self, label: str = "BENIGN"):
        for key in list(self._flows.keys()):
            self._complete(key, label)

    def get_rows(self) -> List[Dict]:
        return self._completed



def pcap_to_cicids(pcap_path: str, csv_path: str, label: str = "BENIGN"):
    table = FlowTable()
    n_pkts = 0
    n_skipped = 0

    for ts_us, raw_ip in _read_pcap(pcap_path):
        pkt = _parse_ip(ts_us, raw_ip)
        if pkt is None:
            n_skipped += 1
            continue
        table.process(pkt, label)
        n_pkts += 1

    table.flush(label)
    rows = table.get_rows()

    if not rows:
        print("⚠  No flows extracted — check that the PCAP contains IPv4 TCP/UDP traffic.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✓  Processed {n_pkts} packets ({n_skipped} skipped)")
    print(f"✓  Extracted {len(df)} flows → {csv_path}")
    return df



def main():
    parser = argparse.ArgumentParser(
        description="Convert a PCAP file to a CICIDS2017-style CSV"
    )
    parser.add_argument("--input",  required=True, help="Input .pcap file")
    parser.add_argument("--output", required=True, help="Output .csv file")
    parser.add_argument("--label",  default="BENIGN",
                        help="Traffic label (default: BENIGN). "
                             "Use attack names like DoS Hulk, PortScan, etc.")
    args = parser.parse_args()
    pcap_to_cicids(args.input, args.output, args.label)


if __name__ == "__main__":
    main()
