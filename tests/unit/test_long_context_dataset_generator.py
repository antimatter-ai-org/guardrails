from __future__ import annotations

import json

from app.tools.generate_long_context_dataset import BucketSpec, generate_sample, generate_split


def test_generate_sample_spans_are_in_bounds_and_non_overlapping() -> None:
    bucket = BucketSpec(name="100k", target_chars=5000, fast_rows=0, full_rows=0)
    sample = generate_sample(
        sample_id="test::sample",
        bucket=bucket,
        lang="en",
        fmt="logs_jsonl",
        placement_profile="spread",
        is_negative=False,
        seed=123,
    )
    text = sample.source_text
    spans = sorted(sample.spans, key=lambda s: (s.start, s.end))
    assert len(text) == 5000
    assert spans, "expected some spans in a positive sample"
    for sp in spans:
        assert 0 <= sp.start < sp.end <= len(text)
        assert sp.label in {
            "email",
            "phone",
            "ip",
            "url",
            "identifier",
            "date",
            "location",
            "secret",
            "payment_card",
        }
    for a, b in zip(spans, spans[1:], strict=False):
        assert b.start >= a.end, f"overlap: {a} then {b}"


def test_expected_in_chunk_windows_matches_chunker() -> None:
    bucket = BucketSpec(name="250k", target_chars=20000, fast_rows=0, full_rows=0)
    sample = generate_sample(
        sample_id="test::coverage",
        bucket=bucket,
        lang="ru",
        fmt="dump_json",
        placement_profile="spread",
        is_negative=False,
        seed=7,
    )
    for sp in sample.spans:
        assert sp.expected_in_chunk_windows is True


def test_generate_split_is_deterministic() -> None:
    buckets = (BucketSpec(name="10k", target_chars=2000, fast_rows=10, full_rows=10),)
    a = generate_split(split="fast", total_rows=10, seed=42, buckets=buckets)
    b = generate_split(split="fast", total_rows=10, seed=42, buckets=buckets)
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
