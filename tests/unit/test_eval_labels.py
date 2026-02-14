from __future__ import annotations

from app.eval.labels import canonicalize_prediction_label


def test_ip_address_canonicalizes_to_ip_not_location() -> None:
    assert canonicalize_prediction_label("IP_ADDRESS") == "ip"
