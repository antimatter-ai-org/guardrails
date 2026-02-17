from __future__ import annotations

import ipaddress
import os
import sysconfig
from dataclasses import dataclass
from typing import Any

from app.model_assets import apply_model_env, resolve_gliner_model_source, resolve_token_classifier_model_source
from app.pytriton_server.registry import build_bindings
from app.runtime.triton_readiness import contract_from_binding, parse_pytriton_url, wait_for_triton_ready


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
    readiness_timeout_s: float = 120.0


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _ensure_libpython_on_ld_library_path() -> None:
    # Triton's python backend stub needs to dlopen libpythonX.Y.so at runtime.
    # When using uv-managed Python, the shared library lives under sysconfig LIBDIR
    # and may not be in the system dynamic loader search path.
    libdir = sysconfig.get_config_var("LIBDIR")
    if not libdir:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [item for item in current.split(":") if item]
    if str(libdir) in parts:
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join([str(libdir)] + parts)


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

        _ensure_libpython_on_ld_library_path()

        try:
            host, http_port = parse_pytriton_url(self._config.pytriton_url)
        except ValueError as exc:
            self._last_error = str(exc)
            raise
        if not _is_loopback_host(host):
            self._last_error = (
                "embedded PyTriton requires loopback GR_PYTRITON_URL host "
                f"(got {host!r}); use localhost or 127.0.0.1"
            )
            raise ValueError(self._last_error)
        normalized_host = "127.0.0.1"
        self._client_url = f"{normalized_host}:{http_port}"
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
        contracts = [contract_from_binding(binding) for binding in bindings]

        try:
            from pytriton.triton import Triton, TritonConfig
        except Exception as exc:
            self._last_error = f"PyTriton import error: {exc}"
            raise RuntimeError(self._last_error) from exc

        triton: Any | None = None
        try:
            triton = Triton(
                config=TritonConfig(
                    http_address=normalized_host,
                    http_port=http_port,
                    grpc_address=normalized_host,
                    grpc_port=int(self._config.grpc_port),
                    metrics_address=normalized_host,
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
            wait_for_triton_ready(
                pytriton_url=self._client_url,
                contracts=contracts,
                timeout_s=self._config.readiness_timeout_s,
            )
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
