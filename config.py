import os
from enum import Enum

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"
DEBUG = not IS_PRODUCTION

def _get_fly_config():
    """Helper to get Fly.io configuration"""
    return {
        "api_key": os.getenv("FLY_API_KEY"),
        "api_host": os.getenv("FLY_API_HOST", "https://api.machines.dev/v1"),
        "app_name": os.getenv("FLY_APP_NAME", "connectiumai"),
    }

def _get_fly_headers(api_key: str):
    """Helper to create Fly.io headers"""
    return {
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }