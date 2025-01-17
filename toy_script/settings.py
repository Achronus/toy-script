from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Pulls the agent settings from a `.env` file in the root directory."""

    GEMINI_API_KEY: str = Field(..., description="API Key for the Gemini model")

    model_config = SettingsConfigDict(env_file=".env")
