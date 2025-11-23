from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Any
import json


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Embedding backend selection
    embeddings_provider: str = Field(
        default="fake",  # "fake" for tests, "bge" for real model in future
        validation_alias="EMBEDDINGS_PROVIDER",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_device: str = Field(
        default="cpu",  # later you can try "cuda"
        validation_alias="EMBEDDING_DEVICE",
    )

    # Basic app info
    app_name: str = Field(default="AI Microservice", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    env: str = Field(default="development", validation_alias="ENV")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Server bind options
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        validation_alias="CORS_ORIGINS",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: Any) -> Any:
        """
        Accept:
          - Python list (already parsed)
          - JSON array string: '["http://...","http://..."]'
          - Comma-separated string: 'http://...,http://...'
        """
        if isinstance(v, list):
            return v

        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []

            # Try JSON array first
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    # fall through to comma-split
                    pass

            # Fallback: comma-separated
            return [item.strip() for item in s.split(",") if item.strip()]

        return v


# Singleton settings instance
settings = Settings()
