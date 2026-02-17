from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.analysis.service import AnalysisFailedError, PresidioAnalysisService
from app.core.masking.reversible import ReversibleMaskingEngine
from app.models.entities import AnalysisDiagnostics, Detection
from app.policy import PolicyResolver
from app.storage.redis_store import RedisMappingStore

logger = logging.getLogger(__name__)


class GuardrailsError(RuntimeError):
    pass


class GuardrailsBlockedError(GuardrailsError):
    def __init__(self, message: str, findings_count: int) -> None:
        super().__init__(message)
        self.findings_count = findings_count


class GuardrailsNotFoundError(GuardrailsError):
    pass


class GuardrailsAnalysisFailedError(GuardrailsError):
    def __init__(self, *, detector_errors: dict[str, str], diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__("analysis failed: one or more detectors errored")
        self.detector_errors = dict(detector_errors)
        self.diagnostics = dict(diagnostics or {})


@dataclass(slots=True)
class TextInput:
    id: str
    text: str


@dataclass(slots=True)
class ItemDetectionResult:
    id: str
    detections: list[Detection]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class ItemMaskResult:
    id: str
    text: str
    detections: list[Detection]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class DetectOperationResult:
    policy_name: str
    mode: str
    findings_count: int
    items: list[ItemDetectionResult]


@dataclass(slots=True)
class MaskOperationResult:
    request_id: str
    policy_name: str
    mode: str
    findings_count: int
    placeholders_count: int
    context_stored: bool
    items: list[ItemMaskResult]


@dataclass(slots=True)
class ItemUnmaskResult:
    id: str
    text: str
    replacements: int


@dataclass(slots=True)
class UnmaskOperationResult:
    request_id: str
    context_found: bool
    replacements: int
    context_deleted: bool
    items: list[ItemUnmaskResult]


@dataclass(slots=True)
class StreamUnmaskOperationResult:
    request_id: str
    stream_id: str
    context_found: bool
    output: str
    replacements: int
    buffered_chars: int
    final: bool
    context_deleted: bool


class GuardrailsService:
    def __init__(
        self,
        policy_resolver: PolicyResolver,
        analysis_service: PresidioAnalysisService,
        mapping_store: RedisMappingStore,
    ) -> None:
        self._policy_resolver = policy_resolver
        self._analysis_service = analysis_service
        self._mapping_store = mapping_store

    @staticmethod
    def _request_hint(request_id: str) -> str:
        return "".join(char for char in request_id if char.isalnum())[:8].upper() or "REQ"

    @staticmethod
    def _max_placeholder_len(placeholders: dict[str, str]) -> int:
        return max((len(key) for key in placeholders), default=1)

    @staticmethod
    def _pending_placeholder_prefix_len(text: str, placeholders: dict[str, str]) -> int:
        if not text or not placeholders:
            return 0
        max_placeholder_len = max((len(placeholder) for placeholder in placeholders), default=0)
        max_suffix = min(max(0, max_placeholder_len - 1), len(text))
        for suffix_len in range(max_suffix, 0, -1):
            suffix = text[-suffix_len:]
            if any(placeholder.startswith(suffix) for placeholder in placeholders):
                return suffix_len
        return 0

    @staticmethod
    def _serialize_diagnostics(diagnostics: AnalysisDiagnostics | None) -> dict[str, Any]:
        if diagnostics is None:
            return {
                "elapsed_ms": 0.0,
                "detector_timing_ms": {},
                "detector_span_counts": {},
                "detector_errors": {},
                "postprocess_mutations": {},
                "limit_flags": {},
            }
        return {
            "elapsed_ms": float(diagnostics.elapsed_ms),
            "detector_timing_ms": dict(diagnostics.detector_timing_ms),
            "detector_span_counts": dict(diagnostics.detector_span_counts),
            "detector_errors": dict(diagnostics.detector_errors),
            "postprocess_mutations": dict(diagnostics.postprocess_mutations),
            "limit_flags": dict(diagnostics.limit_flags),
        }

    def _analyze_with_diagnostics(
        self,
        *,
        text: str,
        profile_name: str,
        policy_min_score: float,
    ) -> tuple[list[Detection], dict[str, Any]]:
        analyzer = getattr(self._analysis_service, "analyze_text_with_diagnostics", None)
        if callable(analyzer):
            try:
                detections, diagnostics = analyzer(
                    text=text,
                    profile_name=profile_name,
                    policy_min_score=policy_min_score,
                )
                return detections, self._serialize_diagnostics(diagnostics)
            except AnalysisFailedError as exc:
                diagnostics = AnalysisDiagnostics(
                    elapsed_ms=0.0,
                    detector_timing_ms=dict(exc.detector_timing_ms),
                    detector_span_counts=dict(exc.detector_span_counts),
                    detector_errors=dict(exc.detector_errors),
                    postprocess_mutations={},
                    limit_flags={},
                )
                raise GuardrailsAnalysisFailedError(
                    detector_errors=exc.detector_errors,
                    diagnostics=self._serialize_diagnostics(diagnostics),
                ) from exc

        detections = self._analysis_service.analyze_text(
            text=text,
            profile_name=profile_name,
            policy_min_score=policy_min_score,
        )
        return detections, self._serialize_diagnostics(None)

    async def detect_items(
        self,
        items: list[TextInput],
        policy_name: str | None = None,
    ) -> DetectOperationResult:
        resolved_policy_name, policy = self._policy_resolver.resolve_policy(policy_name=policy_name)

        item_results: list[ItemDetectionResult] = []
        findings_count = 0
        for item in items:
            detections, diagnostics = self._analyze_with_diagnostics(
                text=item.text,
                profile_name=policy.analyzer_profile,
                policy_min_score=policy.min_score,
            )
            findings_count += len(detections)
            item_results.append(
                ItemDetectionResult(
                    id=item.id,
                    detections=detections,
                    diagnostics=diagnostics,
                )
            )

        return DetectOperationResult(
            policy_name=resolved_policy_name,
            mode=policy.mode,
            findings_count=findings_count,
            items=item_results,
        )

    async def mask_items(
        self,
        request_id: str,
        items: list[TextInput],
        policy_name: str | None = None,
        store_context: bool = True,
        storage_ttl_seconds: int | None = None,
    ) -> MaskOperationResult:
        resolved_policy_name, policy = self._policy_resolver.resolve_policy(policy_name=policy_name)
        effective_ttl = int(storage_ttl_seconds or policy.storage_ttl_seconds)

        if policy.mode == "passthrough":
            if store_context:
                await self._mapping_store.save_mapping(
                    request_id=request_id,
                    payload={
                        "policy_name": resolved_policy_name,
                        "placeholders": {},
                        "max_placeholder_len": 1,
                        "storage_ttl_seconds": effective_ttl,
                    },
                    ttl_seconds=effective_ttl,
                )
            return MaskOperationResult(
                request_id=request_id,
                policy_name=resolved_policy_name,
                mode=policy.mode,
                findings_count=0,
                placeholders_count=0,
                context_stored=store_context,
                items=[
                    ItemMaskResult(
                        id=item.id,
                        text=item.text,
                        detections=[],
                        diagnostics={},
                    )
                    for item in items
                ],
            )

        all_placeholders: dict[str, str] = {}
        findings_count = 0
        output_items: list[ItemMaskResult] = []
        request_hint = self._request_hint(request_id)

        for idx, item in enumerate(items):
            detections, diagnostics = self._analyze_with_diagnostics(
                text=item.text,
                profile_name=policy.analyzer_profile,
                policy_min_score=policy.min_score,
            )
            prefix = f"{policy.placeholder_prefix}{idx:02d}{request_hint}"
            masker = ReversibleMaskingEngine(prefix)
            masked = masker.mask(item.text, detections)
            findings_count += len(masked.detections)
            output_items.append(
                ItemMaskResult(
                    id=item.id,
                    text=masked.text,
                    detections=masked.detections,
                    diagnostics=diagnostics,
                )
            )
            for placeholder, original in masked.placeholders.items():
                existing = all_placeholders.get(placeholder)
                if existing is not None and existing != original:
                    raise GuardrailsError("placeholder collision detected")
                all_placeholders[placeholder] = original

        if policy.mode == "block" and findings_count > 0:
            raise GuardrailsBlockedError("blocked by policy: sensitive data detected", findings_count=findings_count)

        if store_context:
            await self._mapping_store.save_mapping(
                request_id=request_id,
                payload={
                    "policy_name": resolved_policy_name,
                    "placeholders": all_placeholders,
                    "max_placeholder_len": self._max_placeholder_len(all_placeholders),
                    "storage_ttl_seconds": effective_ttl,
                },
                ttl_seconds=effective_ttl,
            )

        return MaskOperationResult(
            request_id=request_id,
            policy_name=resolved_policy_name,
            mode=policy.mode,
            findings_count=findings_count,
            placeholders_count=len(all_placeholders),
            context_stored=store_context,
            items=output_items,
        )

    async def unmask_items(
        self,
        request_id: str,
        items: list[TextInput],
        delete_context: bool = False,
        allow_missing_context: bool = True,
    ) -> UnmaskOperationResult:
        context = await self._mapping_store.load_mapping(request_id)
        if context is None:
            if not allow_missing_context:
                raise GuardrailsNotFoundError(f"request context not found: {request_id}")
            return UnmaskOperationResult(
                request_id=request_id,
                context_found=False,
                replacements=0,
                context_deleted=False,
                items=[ItemUnmaskResult(id=item.id, text=item.text, replacements=0) for item in items],
            )

        placeholders: dict[str, str] = context.get("placeholders", {})
        replacements_total = 0
        output_items: list[ItemUnmaskResult] = []
        for item in items:
            result = ReversibleMaskingEngine.unmask(item.text, placeholders)
            replacements_total += result.replaced
            output_items.append(ItemUnmaskResult(id=item.id, text=result.text, replacements=result.replaced))

        if delete_context:
            await self._mapping_store.delete_all_for_request(request_id)

        return UnmaskOperationResult(
            request_id=request_id,
            context_found=True,
            replacements=replacements_total,
            context_deleted=delete_context,
            items=output_items,
        )

    async def unmask_stream_chunk(
        self,
        request_id: str,
        stream_id: str,
        chunk: str,
        final: bool,
        delete_context: bool = False,
        allow_missing_context: bool = True,
    ) -> StreamUnmaskOperationResult:
        if delete_context and not final:
            raise GuardrailsError("delete_context=true requires final=true")

        context = await self._mapping_store.load_mapping(request_id)
        if context is None:
            if not allow_missing_context:
                raise GuardrailsNotFoundError(f"request context not found: {request_id}")
            return StreamUnmaskOperationResult(
                request_id=request_id,
                stream_id=stream_id,
                context_found=False,
                output=chunk,
                replacements=0,
                buffered_chars=0,
                final=final,
                context_deleted=False,
            )

        placeholders: dict[str, str] = context.get("placeholders", {})
        ttl_seconds = int(context.get("storage_ttl_seconds", 3600))

        if not placeholders:
            if final:
                await self._mapping_store.delete_stream_state(request_id, stream_id)
                if delete_context:
                    await self._mapping_store.delete_all_for_request(request_id)
            return StreamUnmaskOperationResult(
                request_id=request_id,
                stream_id=stream_id,
                context_found=True,
                output=chunk,
                replacements=0,
                buffered_chars=0,
                final=final,
                context_deleted=delete_context,
            )

        state = await self._mapping_store.load_stream_state(request_id, stream_id) or {"buffer": ""}
        combined = str(state.get("buffer", "")) + chunk

        if final:
            unmasked = ReversibleMaskingEngine.unmask(combined, placeholders)
            await self._mapping_store.delete_stream_state(request_id, stream_id)
            if delete_context:
                await self._mapping_store.delete_all_for_request(request_id)
            return StreamUnmaskOperationResult(
                request_id=request_id,
                stream_id=stream_id,
                context_found=True,
                output=unmasked.text,
                replacements=unmasked.replaced,
                buffered_chars=0,
                final=True,
                context_deleted=delete_context,
            )

        holdback = self._pending_placeholder_prefix_len(combined, placeholders)
        safe_text = combined[:-holdback] if holdback else combined
        pending_buffer = combined[-holdback:] if holdback else ""
        unmasked = ReversibleMaskingEngine.unmask(safe_text, placeholders)
        await self._mapping_store.save_stream_state(
            request_id=request_id,
            stream_id=stream_id,
            payload={"buffer": pending_buffer},
            ttl_seconds=ttl_seconds,
        )

        return StreamUnmaskOperationResult(
            request_id=request_id,
            stream_id=stream_id,
            context_found=True,
            output=unmasked.text,
            replacements=unmasked.replaced,
            buffered_chars=len(pending_buffer),
            final=False,
            context_deleted=False,
        )

    async def finalize_request(self, request_id: str) -> bool:
        context = await self._mapping_store.load_mapping(request_id)
        if context is None:
            return False
        await self._mapping_store.delete_all_for_request(request_id)
        return True
