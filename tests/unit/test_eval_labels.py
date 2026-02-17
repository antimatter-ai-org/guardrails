from __future__ import annotations

from app.core.analysis.mapping import canonicalize_entity_type


def test_ip_address_canonicalizes_to_ip_not_location() -> None:
    assert canonicalize_entity_type("IP_ADDRESS") == "ip"


def test_url_canonicalizes_to_url() -> None:
    assert canonicalize_entity_type("URL") == "url"
