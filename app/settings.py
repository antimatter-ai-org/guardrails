from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GR_", env_file=".env", extra="ignore")

    service_name: str = "guardrails-service"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    policy_path: str = "configs/policy.yaml"
    redis_url: str = "redis://redis:6379/0"

    runtime_mode: Literal["cpu", "cuda"] = "cpu"
    enable_nemotron: bool = False
    model_dir: str | None = None
    offline_mode: bool = False

    cpu_device: str = "auto"
    pytriton_url: str = "127.0.0.1:8000"
    pytriton_init_timeout_s: float = 120.0
    pytriton_infer_timeout_s: float = 30.0
    allow_missing_reidentify_session: bool = False


settings = Settings()
