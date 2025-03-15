from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ProctorAI Classifier API"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list = ["*"]  # In production, replace with actual origins
    
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings() 