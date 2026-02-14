from __future__ import annotations

import logging
from typing import Any

from app.config import PolicyDefinition
from app.detectors.factory import DetectorRegistry
from app.masking.engine import SensitiveDataMasker
from app.masking.payload import (
    collect_request_text_slots,
    collect_response_text_slots,
    copy_payload,
    set_value,
)
from app.policy import PolicyResolver
from app.storage.redis_store import RedisMappingStore

logger = logging.getLogger(__name__)


class GuardrailsError(RuntimeError):
    pass


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

    async def mask_request(
        self,
        request_id: str,
        payload: dict[str, Any],
        model_name: str,
        policy_name: str | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        resolved_policy_name, policy = self._policy_resolver.resolve_policy(model_name=model_name, policy_name=policy_name)
        if policy.mode == "passthrough":
            return payload, resolved_policy_name, {"placeholders": {}}

        detectors = self._resolve_detectors(policy)
        if not detectors:
            logger.warning("policy '%s' has no active detectors", resolved_policy_name)

        masked_payload = copy_payload(payload)
        slots = collect_request_text_slots(masked_payload)

        all_placeholders: dict[str, str] = {}
        findings_count = 0

        for slot_idx, slot in enumerate(slots):
            slot_prefix = f"{policy.placeholder_prefix}{slot_idx:02d}"
            masker = SensitiveDataMasker(
                detectors=detectors,
                min_score=policy.min_score,
                placeholder_prefix=slot_prefix,
            )
            result = masker.mask(slot.text)
            set_value(masked_payload, slot.path, result.text)
            findings_count += len(result.detections)
            for placeholder, original in result.placeholders.items():
                if placeholder in all_placeholders and all_placeholders[placeholder] != original:
                    raise GuardrailsError("placeholder collision detected")
                all_placeholders[placeholder] = original

        if policy.mode == "block" and findings_count:
            raise GuardrailsError("blocked by policy: sensitive data detected")

        context = {
            "policy_name": resolved_policy_name,
            "placeholders": all_placeholders,
        }
        await self._mapping_store.save(
            request_id=request_id,
            payload=context,
            ttl_seconds=policy.storage_ttl_seconds,
        )

        return masked_payload, resolved_policy_name, context

    async def unmask_response(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        context = await self._mapping_store.load(request_id)
        if not context:
            return payload

        placeholders: dict[str, str] = context.get("placeholders", {})
        if not placeholders:
            return payload

        unmasked_payload = copy_payload(payload)
        slots = collect_response_text_slots(unmasked_payload)
        for slot in slots:
            unmasked = SensitiveDataMasker.unmask(slot.text, placeholders)
            if unmasked.replaced:
                set_value(unmasked_payload, slot.path, unmasked.text)

        await self._mapping_store.delete(request_id)
        return unmasked_payload
