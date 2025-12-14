from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Allows requests from specified origins with credentials support.
    """
    # Define allowed origins
    allowed_origins = [
        "https://mohakapoor.in",
        "https://sisyphus.mohakapoor.in",
        "http://mohakapoor.in",
        "http://sisyphus.mohakapoor.in",
        "http://localhost:8000"
    ]
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600  # Cache preflight requests for 1 hour
    )