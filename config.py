from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import Field


class Settings(BaseSettings):
    PORT: int = Field(7860, env="PORT")
    HUGGINGFACE_MODEL_EMBEDDING: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="HUGGINGFACE_MODEL_EMBEDDING"
    )
    HUGGINGFACE_MODEL_LLM: str = Field(..., env="HUGGINGFACE_MODEL_LLM")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    OPENROUTER_API_KEY: str = Field(..., env="OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field("deepseek/deepseek-chat-v3.1", env="OPENROUTER_MODEL")

    TOP_K: int = Field(5, env="TOP_K")
    TOP_P: float = Field(0.95, env="TOP_P")

    MAX_NEW_TOKENS: int = Field(512, env="MAX_NEW_TOKENS")
    MIN_NEW_TOKENS: int = Field(256, env="MIN_NEW_TOKENS")
    TEMPERATURE: float = Field(0.1, env="TEMPERATURE")

    CHUNK_SIZE: int = Field(400, env="CHUNK_SIZE")
    SIMILARITY_TOP_K: int = Field(7, env="SIMILARITY_TOP_K")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
