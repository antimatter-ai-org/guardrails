from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GR_", env_file=".env", extra="ignore")

    service_name: str = "guardrails-service"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    policy_path: str = "configs/policy.yaml"
    redis_url: str = "redis://redis:6379/0"

    gliner_backend: str = "local_torch"
    gliner_device: str = "auto"
    gliner_use_fp16_on_cuda: bool = False


settings = Settings()
