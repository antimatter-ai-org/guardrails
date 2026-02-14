from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import PolicyDefinition
from app.detectors.factory import DetectorRegistry
from app.masking.engine import SensitiveDataMasker
from app.models.entities import Detection
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


@dataclass(slots=True)
class TextInput:
    id: str
    text: str


@dataclass(slots=True)
class ItemDetectionResult:
    id: str
    detections: list[Detection]


@dataclass(slots=True)
class ItemMaskResult:
    id: str
    text: str
    detections: list[Detection]


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
        detector_registry: DetectorRegistry,
        mapping_store: RedisMappingStore,
    ) -> None:
        self._policy_resolver = policy_resolver
        self._detector_registry = detector_registry
        self._mapping_store = mapping_store

    def _resolve_detectors(self, policy: PolicyDefinition):
        detectors = []
        for detector_name in policy.detectors:
            detector = self._detector_registry.get(detector_name)
            if detector is None:
                logger.warning("detector '%s' is referenced by policy but unavailable", detector_name)
                continue
            detectors.append(detector)
        return detectors

    @staticmethod
    def _request_hint(request_id: str) -> str:
        return "".join(ch for ch in request_id if ch.isalnum())[:8].upper() or "REQ"

    @staticmethod
    def _max_placeholder_len(placeholders: dict[str, str]) -> int:
        return max((len(key) for key in placeholders), default=1)

    def _mask_one(
        self,
        text: str,
        detectors,
        min_score: float,
        placeholder_prefix: str,
    ):
        masker = SensitiveDataMasker(
            detectors=detectors,
            min_score=min_score,
            placeholder_prefix=placeholder_prefix,
        )
        return masker.mask(text)

    async def detect_items(
        self,
        items: list[TextInput],
        policy_name: str | None = None,
    ) -> DetectOperationResult:
        resolved_policy_name, policy = self._policy_resolver.resolve_policy(policy_name=policy_name)
        detectors = self._resolve_detectors(policy)

        item_results: list[ItemDetectionResult] = []
        findings_count = 0
        for idx, item in enumerate(items):
            result = self._mask_one(
                text=item.text,
                detectors=detectors,
                min_score=policy.min_score,
                placeholder_prefix=f"DET{idx:03d}",
            )
            findings_count += len(result.detections)
            item_results.append(ItemDetectionResult(id=item.id, detections=result.detections))

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
    ) -> MaskOperationResult:
        resolved_policy_name, policy = self._policy_resolver.resolve_policy(policy_name=policy_name)

        if policy.mode == "passthrough":
            if store_context:
                await self._mapping_store.save_mapping(
                    request_id=request_id,
                    payload={
                        "policy_name": resolved_policy_name,
                        "placeholders": {},
                        "max_placeholder_len": 1,
                        "storage_ttl_seconds": policy.storage_ttl_seconds,
                    },
                    ttl_seconds=policy.storage_ttl_seconds,
                )
            return MaskOperationResult(
                request_id=request_id,
                policy_name=resolved_policy_name,
                mode=policy.mode,
                findings_count=0,
                placeholders_count=0,
                context_stored=store_context,
                items=[ItemMaskResult(id=item.id, text=item.text, detections=[]) for item in items],
            )

        detectors = self._resolve_detectors(policy)
        if not detectors:
            logger.warning("policy '%s' has no active detectors", resolved_policy_name)

        all_placeholders: dict[str, str] = {}
        findings_count = 0
        request_hint = self._request_hint(request_id)

        output_items: list[ItemMaskResult] = []
        for idx, item in enumerate(items):
            prefix = f"{policy.placeholder_prefix}{idx:02d}{request_hint}"
            masked = self._mask_one(
                text=item.text,
                detectors=detectors,
                min_score=policy.min_score,
                placeholder_prefix=prefix,
            )
            output_items.append(ItemMaskResult(id=item.id, text=masked.text, detections=masked.detections))
            findings_count += len(masked.detections)

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
                    "storage_ttl_seconds": policy.storage_ttl_seconds,
                },
                ttl_seconds=policy.storage_ttl_seconds,
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
            result = SensitiveDataMasker.unmask(item.text, placeholders)
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
        previous_buffer = str(state.get("buffer", ""))
        combined = previous_buffer + chunk

        if final:
            unmasked = SensitiveDataMasker.unmask(combined, placeholders)
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

        max_placeholder_len = int(context.get("max_placeholder_len") or self._max_placeholder_len(placeholders))
        holdback = max(0, max_placeholder_len - 1)

        if holdback and len(combined) <= holdback:
            await self._mapping_store.save_stream_state(
                request_id=request_id,
                stream_id=stream_id,
                payload={"buffer": combined},
                ttl_seconds=ttl_seconds,
            )
            return StreamUnmaskOperationResult(
                request_id=request_id,
                stream_id=stream_id,
                context_found=True,
                output="",
                replacements=0,
                buffered_chars=len(combined),
                final=False,
                context_deleted=False,
            )

        if holdback:
            safe_text = combined[:-holdback]
            trailing_buffer = combined[-holdback:]
        else:
            safe_text = combined
            trailing_buffer = ""

        unmasked = SensitiveDataMasker.unmask(safe_text, placeholders)

        if trailing_buffer:
            await self._mapping_store.save_stream_state(
                request_id=request_id,
                stream_id=stream_id,
                payload={"buffer": trailing_buffer},
                ttl_seconds=ttl_seconds,
            )
        else:
            await self._mapping_store.delete_stream_state(request_id, stream_id)

        return StreamUnmaskOperationResult(
            request_id=request_id,
            stream_id=stream_id,
            context_found=True,
            output=unmasked.text,
            replacements=unmasked.replaced,
            buffered_chars=len(trailing_buffer),
            final=False,
            context_deleted=False,
        )

    async def finalize_request(self, request_id: str) -> bool:
        await self._mapping_store.delete_all_for_request(request_id)
        return True
