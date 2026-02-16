from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any

from app.model_assets import apply_model_env, resolve_gliner_model_source, resolve_token_classifier_model_source
from app.pytriton_server.registry import build_bindings


@dataclass(slots=True)
class EmbeddedPyTritonConfig:
    pytriton_url: str
    gliner_model_ref: str
    token_model_ref: str
    model_dir: str | None
    offline_mode: bool
    device: str
    max_batch_size: int
    enable_nemotron: bool
    grpc_port: int = 8001
    metrics_port: int = 8002


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _parse_pytriton_url(url: str) -> tuple[str, int]:
    raw = url.strip()
    if not raw:
        raise ValueError("GR_PYTRITON_URL must not be empty")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.split("/", 1)[0]
    if not raw:
        raise ValueError(f"invalid GR_PYTRITON_URL: {url!r}")

    if ":" not in raw:
        host = raw
        port = 8000
    else:
        host, port_raw = raw.rsplit(":", 1)
        if not port_raw:
            raise ValueError(f"invalid GR_PYTRITON_URL: {url!r}")
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"invalid GR_PYTRITON_URL port: {port_raw!r}") from exc

    if not _is_loopback_host(host):
        raise ValueError(
            "embedded PyTriton requires loopback GR_PYTRITON_URL host "
            f"(got {host!r}); use localhost or 127.0.0.1"
        )
    return "127.0.0.1", port


class EmbeddedPyTritonManager:
    def __init__(self, config: EmbeddedPyTritonConfig) -> None:
        self._config = config
        self._triton: Any | None = None
        self._ready = False
        self._last_error: str | None = None
        self._client_url = "127.0.0.1:8000"

    @property
    def client_url(self) -> str:
        return self._client_url

    def start(self) -> None:
        if self._triton is not None and self._ready:
            return

        try:
            host, http_port = _parse_pytriton_url(self._config.pytriton_url)
        except ValueError as exc:
            self._last_error = str(exc)
            raise
        self._client_url = f"{host}:{http_port}"
        self._ready = False
        self._last_error = None
        self._triton = None

        apply_model_env(model_dir=self._config.model_dir, offline_mode=self._config.offline_mode)
        gliner_source = resolve_gliner_model_source(
            model_name=self._config.gliner_model_ref,
            model_dir=self._config.model_dir,
            strict=self._config.offline_mode,
        )
        token_source = ""
        if self._config.enable_nemotron:
            token_source = resolve_token_classifier_model_source(
                model_name=self._config.token_model_ref,
                model_dir=self._config.model_dir,
                strict=self._config.offline_mode,
            )

        bindings = build_bindings(
            gliner_model_ref=gliner_source,
            token_classifier_model_ref=token_source,
            device=self._config.device,
            max_batch_size=self._config.max_batch_size,
            enable_nemotron=self._config.enable_nemotron,
        )

        try:
            from pytriton.triton import Triton, TritonConfig
        except Exception as exc:
            self._last_error = f"PyTriton import error: {exc}"
            raise RuntimeError(self._last_error) from exc

        triton: Any | None = None
        try:
            triton = Triton(
                config=TritonConfig(
                    http_address="127.0.0.1",
                    http_port=http_port,
                    grpc_address="127.0.0.1",
                    grpc_port=int(self._config.grpc_port),
                    metrics_address="127.0.0.1",
                    metrics_port=int(self._config.metrics_port),
                )
            )
            for binding in bindings:
                triton.bind(
                    model_name=binding.name,
                    infer_func=binding.infer_func,
                    inputs=binding.inputs,
                    outputs=binding.outputs,
                    config=binding.config,
                )
            triton.run()
            self._triton = triton
            self._ready = True
            self._last_error = None
        except Exception as exc:
            self._last_error = f"PyTriton startup failed: {exc}"
            try:
                if triton is not None:
                    triton.stop()
            except Exception:
                pass
            self._triton = None
            self._ready = False
            raise RuntimeError(self._last_error) from exc

    def stop(self) -> None:
        triton = self._triton
        if triton is None:
            self._ready = False
            return
        try:
            triton.stop()
        except Exception as exc:
            self._last_error = f"PyTriton stop failed: {exc}"
        finally:
            self._triton = None
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def last_error(self) -> str | None:
        return self._last_error
