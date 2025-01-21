# src/cue/config.py
import os
import sys
import logging
import platform
from enum import Enum
from typing import ClassVar, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    ENVIRONMENT: str = Environment.PRODUCTION.value
    API_URL: str = ""  # Default to localhost
    AGENTS_CONFIG_FILE: str = ""
    ACCESS_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION.value

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT.value

    @property
    def is_test(self) -> bool:
        return self.ENVIRONMENT == Environment.TEST.value or "pytest" in sys.modules

    model_config = SettingsConfigDict(
        env_file=".env.production",
        env_file_encoding="utf-8",
        from_attributes=True,
        extra="ignore",
        use_enum_values=True,
    )

    def get_base_url(self) -> str:
        base_url = self.API_URL
        if platform.system() != "Darwin" and "http://localhost" in base_url:
            base_url = base_url.replace("http://localhost", "http://host.docker.internal")
        return base_url


class SettingsManager:
    _instance: ClassVar[Optional[Settings]] = None

    @classmethod
    def get_settings(cls) -> Settings:
        if cls._instance is None:
            env = os.getenv("ENVIRONMENT", Environment.PRODUCTION.value)
            logger.info(f"Loading settings for environment: {env}")

            if env == Environment.DEVELOPMENT.value:
                cls._instance = Settings(_env_file=".env.development")
            elif env == Environment.TEST.value:
                cls._instance = Settings(_env_file=".env.test")
            else:
                cls._instance = Settings(_env_file=".env.production")

        return cls._instance


def get_settings() -> Settings:
    return SettingsManager.get_settings()
