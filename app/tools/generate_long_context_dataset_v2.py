from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random
from typing import Any, Literal

from app.eval.script_profile import classify_script_profile


SplitName = Literal["fast", "full"]
Language = Literal["ru", "en"]
FormatName = Literal["logs_jsonl", "logs_plain", "code_python", "code_js", "dump_csv", "dump_json"]
LengthBucket = Literal["10k", "50k", "100k", "250k", "1m"]
PlacementProfile = Literal["head", "middle", "tail", "spread", "middle_only"]
Placement = Literal["head", "middle", "tail"]


CANONICAL_LABELS: tuple[str, ...] = (
    "person",
    "email",
    "phone",
    "ip",
    "url",
    "identifier",
    "date",
    "location",
    "secret",
    "payment_card",
)


@dataclass(frozen=True, slots=True)
class BucketSpec:
    name: LengthBucket
    target_chars: int
    fast_rows: int
    full_rows: int


def _default_buckets(*, max_chars: int) -> tuple[BucketSpec, ...]:
    return (
        BucketSpec(name="10k", target_chars=10_000, fast_rows=20, full_rows=150),
        BucketSpec(name="50k", target_chars=50_000, fast_rows=20, full_rows=150),
        BucketSpec(name="100k", target_chars=100_000, fast_rows=16, full_rows=100),
        BucketSpec(name="250k", target_chars=250_000, fast_rows=16, full_rows=75),
        BucketSpec(name="1m", target_chars=min(1_000_000, int(max_chars)), fast_rows=8, full_rows=25),
    )


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: str
    value_type: str
    placement: Placement
    expected_in_chunk_windows: bool


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    source_text: str
    spans: list[Span]
    language: Language
    fmt: FormatName
    length_bucket: LengthBucket
    placement_profile: PlacementProfile
    seed: int

    @property
    def entity_count(self) -> int:
        return len(self.spans)


@dataclass(frozen=True, slots=True)
class SectionCtx:
    start_len: int
    target_len: int
    total_len: int
    is_first: bool
    is_last: bool


class _TextBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._len = 0

    @property
    def length(self) -> int:
        return self._len

    def append(self, text: str) -> None:
        if not text:
            return
        self._parts.append(text)
        self._len += len(text)

    def add_value(
        self,
        *,
        label: str,
        value: str,
        value_type: str,
        placement: Placement,
        expected_in_chunk_windows: bool = True,  # computed later
    ) -> Span:
        start = self._len
        self.append(value)
        end = self._len
        return Span(
            start=start,
            end=end,
            label=label,
            value_type=value_type,
            placement=placement,
            expected_in_chunk_windows=bool(expected_in_chunk_windows),
        )

    def build(self) -> str:
        return "".join(self._parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and optionally publish a synthetic long-context PII dataset (v2).")
    p.add_argument("--repo-id", default="antimatter-ai/guardrails-long-context-pii-synth-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast-rows", type=int, default=80)
    p.add_argument("--full-rows", type=int, default=500)
    p.add_argument("--max-chars", type=int, default=1_000_000)
    p.add_argument("--push", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--audit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache-dir", default=".eval_cache/hf", help="HF cache dir (hub/datasets caches).")
    p.add_argument(
        "--out-dir",
        default=".local/generated_datasets/long_context_pii_synth_v2",
        help="Local artifact output dir.",
    )
    return p.parse_args()


def _configure_hf_cache(cache_dir: str) -> str:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    # Do not override HF_HOME (can hide auth token); only point caches.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    return str(cache_path)


def _alloc_bucket_counts(*, total_rows: int, buckets: tuple[BucketSpec, ...]) -> dict[LengthBucket, int]:
    # Use fixed distribution unless overridden by --fast-rows/--full-rows.
    # We scale bucket counts proportionally to keep the shape stable.
    base: dict[LengthBucket, int] = {}
    for b in buckets:
        base[b.name] = int(b.fast_rows if total_rows == 80 else b.full_rows)
    base_total = sum(base.values())
    if base_total <= 0:
        raise ValueError("invalid base bucket totals")

    # Largest remainder method for stable totals.
    scaled: dict[LengthBucket, float] = {k: (v / base_total) * total_rows for k, v in base.items()}
    floors: dict[LengthBucket, int] = {k: int(v) for k, v in scaled.items()}
    remainder = total_rows - sum(floors.values())
    order = sorted(scaled.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
    for k, _ in order[: max(0, remainder)]:
        floors[k] += 1
    return floors


def _alloc_negatives(bucket_counts: dict[LengthBucket, int], total_neg: int) -> dict[LengthBucket, int]:
    total = sum(bucket_counts.values())
    if total <= 0:
        return {k: 0 for k in bucket_counts}

    scaled = {k: (v / total) * total_neg for k, v in bucket_counts.items()}
    floors = {k: int(v) for k, v in scaled.items()}
    remainder = total_neg - sum(floors.values())
    order = sorted(scaled.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=True)
    for k, _ in order[: max(0, remainder)]:
        floors[k] += 1
    return floors


def _pick_language(idx: int) -> Language:
    # Keep counts as balanced as possible (odd totals differ by 1).
    return "ru" if (idx % 2 == 0) else "en"


def _pick_format(idx: int) -> FormatName:
    # Enforce ~1/3 logs, 1/3 code, 1/3 dumps; round-robin within each family.
    families = ["logs", "code", "dump"]
    fam = families[idx % len(families)]
    if fam == "logs":
        return "logs_jsonl" if (idx // 3) % 2 == 0 else "logs_plain"
    if fam == "code":
        return "code_python" if (idx // 3) % 2 == 0 else "code_js"
    return "dump_json" if (idx // 3) % 2 == 0 else "dump_csv"


def _pick_placement_profile(bucket: LengthBucket, idx: int) -> PlacementProfile:
    if bucket in {"100k", "250k", "1m"}:
        choices: tuple[PlacementProfile, ...] = ("head", "middle", "tail", "spread", "middle_only")
        return choices[idx % len(choices)]
    choices2: tuple[PlacementProfile, ...] = ("head", "middle", "tail", "spread")
    return choices2[idx % len(choices2)]


def _random_hex(rng: Random, n_bytes: int) -> str:
    return rng.randbytes(n_bytes).hex()


def _b64url(rng: Random, n_bytes: int) -> str:
    raw = rng.randbytes(n_bytes)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _luhn_check_digit(prefix: str) -> str:
    digits = [int(c) for c in prefix]
    s = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d2 = d * 2
            if d2 > 9:
                d2 -= 9
            s += d2
        else:
            s += d
    check = (10 - (s % 10)) % 10
    return str(check)


def _synthetic_payment_card(rng: Random) -> str:
    # 16-digit number, Luhn-valid.
    prefix = "79927398713"
    while len(prefix) < 15:
        prefix += str(rng.randint(0, 9))
    prefix = prefix[:15]
    return prefix + _luhn_check_digit(prefix)


def _synthetic_ip(rng: Random) -> tuple[str, str]:
    pool = rng.choice(["192.0.2", "198.51.100", "203.0.113", "10"])
    if pool == "10":
        return f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}", "ipv4_private"
    return f"{pool}.{rng.randint(1,254)}", "ipv4_testnet"


def _synthetic_email(rng: Random) -> str:
    tld = rng.choice(["com", "net", "org", "io", "dev"])
    return f"user{rng.randint(1, 10_000_000)}@example.{tld}"


def _synthetic_url(rng: Random) -> str:
    return f"https://example.com/{_random_hex(rng, 6)}"


def _synthetic_phone(rng: Random, lang: Language) -> tuple[str, str]:
    if lang == "ru":
        return (
            f"+7 (9{rng.randint(0,9)}{rng.randint(0,9)}) {rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(10,99)}",
            "ru_phone",
        )
    return (f"+1 ({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}", "us_phone")


def _synthetic_date(rng: Random) -> tuple[str, str]:
    y = rng.randint(2018, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    if rng.random() < 0.5:
        return f"{y:04d}-{m:02d}-{d:02d}", "iso_date"
    return f"{d:02d}.{m:02d}.{y:04d}", "dmy_date"


def _synthetic_identifier(rng: Random, lang: Language) -> tuple[str, str]:
    if lang == "ru":
        inn = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([10, 12])))
        return inn, "ru_inn"
    return f"ID-{_random_hex(rng, 6)}", "generic_id"


def _synthetic_secret(rng: Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        return f"GR_SECRET_{_b64url(rng, rng.randint(24, 48))}", "gr_secret"
    return f"ghp_{_b64url(rng, 24)}", "github_like"


def _synthetic_location(rng: Random, lang: Language) -> tuple[str, str]:
    if lang == "ru":
        street = rng.choice(["ул. Ленина", "ул. Пушкина", "пр-т Мира", "ул. Гагарина", "ул. Советская"])
        city = rng.choice(["Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Екатеринбург"])
        house = rng.randint(1, 250)
        apt = rng.randint(1, 500)
        return f"{city}, {street}, д. {house}, кв. {apt}", "ru_address"
    street = rng.choice(["Main St", "Oak Ave", "Pine Rd", "Maple Dr", "Cedar Ln"])
    city = rng.choice(["Springfield", "Riverton", "Fairview", "Madison", "Georgetown"])
    house = rng.randint(1, 9999)
    return f"{house} {street}, {city}", "en_address"


_EN_FIRST = (
    "John",
    "Emma",
    "Michael",
    "Olivia",
    "James",
    "Ava",
    "Daniel",
    "Sophia",
    "David",
    "Isabella",
)
_EN_LAST = (
    "Smith",
    "Johnson",
    "Brown",
    "Taylor",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
)
_RU_FIRST = (
    "Иван",
    "Анна",
    "Алексей",
    "Мария",
    "Павел",
    "Елена",
    "Дмитрий",
    "Ольга",
    "Сергей",
    "Наталья",
)
_RU_LAST = (
    "Иванов",
    "Петров",
    "Смирнов",
    "Кузнецов",
    "Попов",
    "Соколов",
    "Лебедев",
    "Козлов",
    "Новиков",
    "Морозов",
)


def _synthetic_person(rng: Random, lang: Language) -> tuple[str, str]:
    if lang == "ru":
        return f"{rng.choice(_RU_FIRST)} {rng.choice(_RU_LAST)}", "ru_name"
    return f"{rng.choice(_EN_FIRST)} {rng.choice(_EN_LAST)}", "en_name"


@dataclass(frozen=True, slots=True)
class _ValueSpec:
    label: str
    value: str
    value_type: str
    placement: Placement


def _target_span_count(bucket: LengthBucket, rng: Random) -> int:
    if bucket in {"10k", "50k"}:
        return rng.randint(15, 40)
    if bucket in {"100k", "250k"}:
        return rng.randint(25, 60)
    return rng.randint(40, 80)


def _build_value_plan(*, bucket: LengthBucket, lang: Language, placement_profile: PlacementProfile, rng: Random) -> list[_ValueSpec]:
    count = _target_span_count(bucket, rng)

    required = ["email", "ip", "identifier", "secret"]
    labels: list[str] = list(required)

    # Encourage coverage.
    if rng.random() < 0.85:
        labels.append("person")
    if rng.random() < 0.85:
        labels.append("location")
    if rng.random() < 0.85:
        labels.append("phone")

    # payment_card ~15% of positives.
    if rng.random() < 0.15:
        labels.append("payment_card")

    weights = [
        ("person", 1.2),
        ("email", 1.0),
        ("phone", 1.0),
        ("ip", 1.0),
        ("url", 0.8),
        ("identifier", 1.2),
        ("date", 0.8),
        ("location", 1.1),
        ("secret", 1.2),
        ("payment_card", 0.25),
    ]
    total = sum(w for _, w in weights)
    cum: list[tuple[str, float]] = []
    acc = 0.0
    for lab, w in weights:
        acc += w
        cum.append((lab, acc))

    def pick_label() -> str:
        x = rng.random() * total
        for lab, edge in cum:
            if x <= edge:
                return lab
        return cum[-1][0]

    while len(labels) < count:
        labels.append(pick_label())

    rng.shuffle(labels)

    if placement_profile == "head":
        placements = ["head"] * len(labels)
    elif placement_profile == "tail":
        placements = ["tail"] * len(labels)
    elif placement_profile in {"middle", "middle_only"}:
        placements = ["middle"] * len(labels)
    else:
        cycle: tuple[Placement, ...] = ("head", "middle", "tail")
        placements = [cycle[i % len(cycle)] for i in range(len(labels))]
        rng.shuffle(placements)

    out: list[_ValueSpec] = []
    for lab, place in zip(labels, placements, strict=False):
        if lab == "person":
            v, vt = _synthetic_person(rng, lang)
            out.append(_ValueSpec(label="person", value=v, value_type=vt, placement=place))
        elif lab == "email":
            out.append(_ValueSpec(label="email", value=_synthetic_email(rng), value_type="email", placement=place))
        elif lab == "phone":
            v, vt = _synthetic_phone(rng, lang)
            out.append(_ValueSpec(label="phone", value=v, value_type=vt, placement=place))
        elif lab == "ip":
            v, vt = _synthetic_ip(rng)
            out.append(_ValueSpec(label="ip", value=v, value_type=vt, placement=place))
        elif lab == "url":
            out.append(_ValueSpec(label="url", value=_synthetic_url(rng), value_type="url", placement=place))
        elif lab == "identifier":
            v, vt = _synthetic_identifier(rng, lang)
            out.append(_ValueSpec(label="identifier", value=v, value_type=vt, placement=place))
        elif lab == "date":
            v, vt = _synthetic_date(rng)
            out.append(_ValueSpec(label="date", value=v, value_type=vt, placement=place))
        elif lab == "location":
            v, vt = _synthetic_location(rng, lang)
            out.append(_ValueSpec(label="location", value=v, value_type=vt, placement=place))
        elif lab == "secret":
            v, vt = _synthetic_secret(rng)
            out.append(_ValueSpec(label="secret", value=v, value_type=vt, placement=place))
        elif lab == "payment_card":
            out.append(
                _ValueSpec(label="payment_card", value=_synthetic_payment_card(rng), value_type="luhn_cc", placement=place)
            )
        else:
            v, vt = _synthetic_identifier(rng, lang)
            out.append(_ValueSpec(label="identifier", value=v, value_type=vt, placement=place))
    return out


_EN_WORDS = (
    "error",
    "warning",
    "request",
    "response",
    "trace",
    "token",
    "config",
    "service",
    "worker",
    "timeout",
    "payload",
    "message",
    "handler",
    "parser",
    "stream",
    "chunk",
    "cache",
    "vector",
    "index",
)
_RU_WORDS = (
    "ошибка",
    "предупреждение",
    "запрос",
    "ответ",
    "трассировка",
    "токен",
    "конфигурация",
    "сервис",
    "воркер",
    "таймаут",
    "пейлоад",
    "сообщение",
    "обработчик",
    "парсер",
    "поток",
    "чанк",
    "кэш",
    "индекс",
)


def _filler_words(lang: Language, rng: Random, count: int, *, dense: bool) -> str:
    if dense:
        # High token density to stress chunk-capping at smaller char budgets.
        token = "a" if lang == "en" else "а"
        return (token + " ") * max(1, count)
    words = _EN_WORDS if lang == "en" else _RU_WORDS
    return " ".join(rng.choice(words) for _ in range(max(1, count)))


def _choose_insert_count(
    *,
    rng: Random,
    remaining_values: int,
    remaining_chars: int,
    avg_line_len: int,
    max_insert: int,
) -> int:
    if remaining_values <= 0:
        return 0
    avg_line_len = max(1, int(avg_line_len))
    remaining_lines = max(1, int(remaining_chars) // avg_line_len)
    min_insert = max(0, remaining_values - (remaining_lines - 1) * max_insert)
    min_insert = min(min_insert, max_insert)
    extra_cap = min(max_insert - min_insert, remaining_values - min_insert)
    extra = 0
    if extra_cap > 0 and rng.random() < 0.7:
        extra = rng.randint(0, extra_cap)
    return min_insert + extra


def _pop_values(
    values_q: list[_ValueSpec],
    count: int,
    *,
    unique_labels: bool,
) -> list[_ValueSpec]:
    if count <= 0 or not values_q:
        return []
    if not unique_labels:
        out = values_q[:count]
        del values_q[:count]
        return out
    inserted: list[_ValueSpec] = []
    seen_labels: set[str] = set()
    spins = 0
    while values_q and len(inserted) < count and spins < 64:
        spins += 1
        cand = values_q.pop(0)
        if cand.label in seen_labels:
            values_q.append(cand)
            continue
        seen_labels.add(cand.label)
        inserted.append(cand)
    return inserted


def _pop_values_with_late(
    values_q: list[_ValueSpec],
    late_q: list[_ValueSpec],
    count: int,
    *,
    allow_late: bool,
    unique_labels: bool,
) -> list[_ValueSpec]:
    if count <= 0:
        return []
    inserted: list[_ValueSpec] = []
    if allow_late and late_q:
        take = min(count, len(late_q))
        inserted.extend(_pop_values(late_q, take, unique_labels=unique_labels))
    if len(inserted) < count and values_q:
        inserted.extend(_pop_values(values_q, count - len(inserted), unique_labels=unique_labels))
    return inserted


def _shrink_to_fit(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len]


def _emit_logs_plain(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    ts_base = date(2026, 2, 18).isoformat()

    while builder.length < ctx.target_len:
        remaining_chars = ctx.target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=120,
            max_insert=3,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 120 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=False)

        lvl = rng.choice(levels)
        prefix = f"{ts_base}T12:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z {lvl} "
        filler = _filler_words(lang, rng, rng.randint(6, 18), dense=dense_tokens)
        values_prefix_len = sum(len(f" | {v.label}=") for v in inserted)
        values_len = sum(len(v.value) for v in inserted)
        min_len = len(prefix) + values_prefix_len + values_len + 1
        if min_len > remaining_chars:
            # If we are running out of space, drop filler entirely.
            filler = ""
        else:
            allow = max(0, remaining_chars - min_len)
            filler = _shrink_to_fit(filler, allow)
        builder.append(prefix)
        builder.append(filler)
        for v in inserted:
            builder.append(f" | {v.label}=")
            spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
        if builder.length < ctx.target_len:
            builder.append("\n")

        if builder.length >= ctx.target_len and not values_q and not late_q:
            break

    return spans


def _emit_logs_jsonl(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)
    ts_base = date(2026, 2, 18).isoformat()

    while builder.length < ctx.target_len:
        remaining_chars = ctx.target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=180,
            max_insert=3,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 180 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=True)

        msg_words = rng.randint(8, 20)
        payload: dict[str, Any] = {
            "ts": f"{ts_base}T12:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z",
            "level": rng.choice(["INFO", "WARN", "ERROR", "DEBUG"]),
            "service": rng.choice(["guardrails", "api", "worker", "billing", "auth"]),
            "msg": _filler_words(lang, rng, msg_words, dense=dense_tokens),
            "trace_id": _random_hex(rng, 8),
            "request_id": _random_hex(rng, 6),
        }

        for item in inserted:
            key = {
                "person": "user_name",
                "email": "user_email",
                "phone": "user_phone",
                "ip": "client_ip",
                "url": "url",
                "identifier": "user_id",
                "date": "event_date",
                "location": "address",
                "secret": "api_key",
                "payment_card": "payment_card",
            }.get(item.label, item.label)
            payload[key] = item.value

        line = json.dumps(payload, ensure_ascii=False)
        # Shrink msg if we are close to the limit.
        if len(line) + 1 > remaining_chars and msg_words > 2:
            for _ in range(6):
                msg_words = max(2, int(msg_words * 0.7))
                payload["msg"] = _filler_words(lang, rng, msg_words, dense=dense_tokens)
                line = json.dumps(payload, ensure_ascii=False)
                if len(line) + 1 <= remaining_chars:
                    break
        if len(line) + 1 > remaining_chars:
            # Truncate msg to fit if still too long.
            overflow = len(line) + 1 - remaining_chars
            msg = str(payload.get("msg", ""))
            payload["msg"] = _shrink_to_fit(msg, max(0, len(msg) - overflow - 5))
            line = json.dumps(payload, ensure_ascii=False)

        for item in inserted:
            local = line.find(item.value)
            if local >= 0:
                start = builder.length + local
                end = start + len(item.value)
                spans.append(
                    Span(
                        start=start,
                        end=end,
                        label=item.label,
                        value_type=item.value_type,
                        placement=item.placement,
                        expected_in_chunk_windows=True,
                    )
                )
        builder.append(line)
        if builder.length < ctx.target_len:
            builder.append("\n")

        if builder.length >= ctx.target_len and not values_q and not late_q:
            break

    return spans


def _emit_code_python(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    _ = dense_tokens
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)

    if ctx.is_first:
        builder.append("# synthetic module\n")
        builder.append("CONFIG = {\n")
        config_count = max(1, min(3, max(1, len(values_q) // 5))) if values_q else 0
        config_vals = _pop_values(values_q, config_count, unique_labels=False)
        for v in config_vals:
            builder.append(f"    '{v.label}': '")
            spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
            builder.append("',\n")
        builder.append("}\n\n")

    fn = 0
    while builder.length < ctx.target_len:
        remaining_chars = ctx.target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=120,
            max_insert=2,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 120 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=False)

        fn += 1
        builder.append(f"def handler_{fn}(payload: dict) -> dict:\n")
        builder.append('    """synthetic handler"""\n')
        builder.append("    # ")
        builder.append(_filler_words(lang, rng, rng.randint(8, 18), dense=False))
        builder.append("\n")
        for v in inserted:
            builder.append("    note = '")
            spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
            builder.append("'\n")
        builder.append("    return {'ok': True}\n\n")
        if builder.length >= ctx.target_len and not values_q and not late_q:
            break

    for v in values_q + late_q:
        builder.append("# fixture: ")
        spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
        builder.append("\n")
    return spans


def _emit_code_js(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    _ = dense_tokens
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)

    if ctx.is_first:
        builder.append("// synthetic module\n")
        builder.append("export const CONFIG = {\n")
        config_count = max(1, min(3, max(1, len(values_q) // 5))) if values_q else 0
        config_vals = _pop_values(values_q, config_count, unique_labels=False)
        for v in config_vals:
            builder.append(f"  {v.label}: '")
            spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
            builder.append("',\n")
        builder.append("};\n\n")

    fn = 0
    while builder.length < ctx.target_len:
        remaining_chars = ctx.target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=120,
            max_insert=2,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 120 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=False)

        fn += 1
        builder.append(f"export function handler{fn}(payload) {{\n")
        builder.append("  // ")
        builder.append(_filler_words(lang, rng, rng.randint(8, 20), dense=False))
        builder.append("\n")
        for v in inserted:
            builder.append("  const note = '")
            spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
            builder.append("';\n")
        builder.append("  return { ok: true };\n}\n\n")
        if builder.length >= ctx.target_len and not values_q and not late_q:
            break

    for v in values_q + late_q:
        builder.append("// fixture: ")
        spans.append(builder.add_value(label=v.label, value=v.value, value_type=v.value_type, placement=v.placement))
        builder.append("\n")
    return spans


def _emit_dump_csv(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    _ = dense_tokens
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)
    headers = [
        "row_id",
        "person",
        "email",
        "phone",
        "ip",
        "url",
        "identifier",
        "date",
        "location",
        "secret",
        "payment_card",
    ]

    if ctx.is_first:
        builder.append(",".join(headers) + "\n")

    def csv_safe(value: str) -> str:
        return str(value).replace("\n", " ").replace("\r", " ").replace(",", " ")

    row_id = 0
    while builder.length < ctx.target_len:
        remaining_chars = ctx.target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=140,
            max_insert=4,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 140 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=True)

        row_id += 1
        fields: dict[str, str] = {"row_id": str(row_id)}
        inserted_safe: list[tuple[_ValueSpec, str]] = []
        for v in inserted:
            safe = csv_safe(v.value)
            inserted_safe.append((v, safe))
            fields[v.label] = safe

        for h in headers[1:]:
            if h in fields:
                continue
            if h == "secret":
                fields[h] = f"SYN_{_random_hex(rng, 8)}"
            elif h == "location":
                fields[h] = "N/A"
            elif h == "date":
                fields[h] = "N/A"
            elif h == "person":
                fields[h] = "N/A"
            else:
                fields[h] = f"X{_random_hex(rng, 3)}"

        line = ",".join(fields[h] for h in headers) + "\n"
        for v, safe in inserted_safe:
            start = builder.length + line.find(safe)
            end = start + len(safe)
            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=v.label,
                    value_type=v.value_type,
                    placement=v.placement,
                    expected_in_chunk_windows=True,
                )
            )
        builder.append(line)
        if builder.length >= ctx.target_len and not values_q and not late_q:
            break

    return spans


def _emit_dump_json(
    builder: _TextBuilder,
    *,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_values: list[_ValueSpec],
) -> list[Span]:
    spans: list[Span] = []
    values_q = list(values)
    late_q = list(late_values)

    if ctx.is_first:
        builder.append("[\n")

    row_id = 0
    close_suffix = "{}\n]\n" if ctx.is_last else ""
    target_len = ctx.target_len - len(close_suffix)

    while builder.length < target_len:
        remaining_chars = target_len - builder.length
        remaining_values = len(values_q) + len(late_q)
        insert_now = _choose_insert_count(
            rng=rng,
            remaining_values=remaining_values,
            remaining_chars=remaining_chars,
            avg_line_len=200,
            max_insert=4,
        )
        allow_late = bool(late_q) and (
            builder.length >= int(ctx.total_len * 0.95) or remaining_chars // 200 <= len(late_q)
        )
        inserted = _pop_values_with_late(values_q, late_q, insert_now, allow_late=allow_late, unique_labels=True)

        row_id += 1
        msg_words = rng.randint(10, 26)
        obj: dict[str, Any] = {
            "row_id": row_id,
            "msg": _filler_words(lang, rng, msg_words, dense=dense_tokens),
            "trace_id": _random_hex(rng, 8),
        }
        for item in inserted:
            obj[item.label] = item.value

        line = json.dumps(obj, ensure_ascii=False)
        if len(line) + 2 > remaining_chars and msg_words > 2:
            for _ in range(6):
                msg_words = max(2, int(msg_words * 0.7))
                obj["msg"] = _filler_words(lang, rng, msg_words, dense=dense_tokens)
                line = json.dumps(obj, ensure_ascii=False)
                if len(line) + 2 <= remaining_chars:
                    break
        if len(line) + 2 > remaining_chars:
            overflow = len(line) + 2 - remaining_chars
            msg = str(obj.get("msg", ""))
            obj["msg"] = _shrink_to_fit(msg, max(0, len(msg) - overflow - 5))
            line = json.dumps(obj, ensure_ascii=False)

        for item in inserted:
            start = builder.length + line.find(item.value)
            end = start + len(item.value)
            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=item.label,
                    value_type=item.value_type,
                    placement=item.placement,
                    expected_in_chunk_windows=True,
                )
            )
        builder.append(line)
        if builder.length < target_len:
            builder.append(",\n")

        if builder.length >= target_len and not values_q and not late_q:
            break

    if ctx.is_last:
        builder.append(close_suffix)
    return spans


def _emit_section(
    builder: _TextBuilder,
    *,
    fmt: FormatName,
    lang: Language,
    rng: Random,
    ctx: SectionCtx,
    values: list[_ValueSpec],
    dense_tokens: bool,
    late_ratio: float,
) -> list[Span]:
    late_values: list[_ValueSpec] = []
    if late_ratio > 0 and values:
        rng.shuffle(values)
        late_count = max(1, min(len(values), int(len(values) * late_ratio)))
        late_values = values[-late_count:]
        values = values[:-late_count]

    if fmt == "logs_plain":
        return _emit_logs_plain(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)
    if fmt == "logs_jsonl":
        return _emit_logs_jsonl(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)
    if fmt == "code_python":
        return _emit_code_python(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)
    if fmt == "code_js":
        return _emit_code_js(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)
    if fmt == "dump_csv":
        return _emit_dump_csv(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)
    return _emit_dump_json(builder, lang=lang, rng=rng, ctx=ctx, values=values, dense_tokens=dense_tokens, late_values=late_values)


def _compute_expected_coverage(*, text: str, spans: list[Span]) -> list[Span]:
    _ = text
    # Guardrails chunking is intended to be fully covering,
    # so spans are expected to be contained within the scanned windows.
    return [dataclasses.replace(sp, expected_in_chunk_windows=True) for sp in spans]


def _split_values_by_placement(values: list[_ValueSpec]) -> dict[Placement, list[_ValueSpec]]:
    out: dict[Placement, list[_ValueSpec]] = {"head": [], "middle": [], "tail": []}
    for v in values:
        out[v.placement].append(v)
    return out


def _section_fracs(bucket: LengthBucket, placement_profile: PlacementProfile) -> tuple[float, float, float]:
    if placement_profile in {"head", "tail"}:
        return (0.15, 0.70, 0.15)
    if placement_profile in {"middle", "middle_only"}:
        return (0.35, 0.30, 0.35)
    # spread
    return (0.20, 0.60, 0.20)


def generate_sample(
    *,
    sample_id: str,
    bucket: BucketSpec,
    lang: Language,
    fmt: FormatName,
    placement_profile: PlacementProfile,
    is_negative: bool,
    seed: int,
) -> Sample:
    rng = Random(int(seed))
    target_len = int(bucket.target_chars)
    builder = _TextBuilder()

    values: list[_ValueSpec]
    if is_negative:
        values = []
    else:
        values = _build_value_plan(bucket=bucket.name, lang=lang, placement_profile=placement_profile, rng=rng)

    values_by_place = _split_values_by_placement(values)

    head_frac, mid_frac, tail_frac = _section_fracs(bucket.name, placement_profile)
    head_target = max(0, int(target_len * head_frac))
    mid_target = max(0, int(target_len * mid_frac))
    tail_target = max(0, target_len - head_target - mid_target)

    dense_tokens = placement_profile == "middle_only" and bucket.name in {"100k", "250k", "1m"}

    spans: list[Span] = []
    sections = [
        ("head", head_target, True, False),
        ("middle", head_target + mid_target, False, False),
        ("tail", head_target + mid_target + tail_target, False, True),
    ]
    for idx, (section_name, section_target, is_first, is_last) in enumerate(sections):
        ctx = SectionCtx(
            start_len=builder.length,
            target_len=section_target,
            total_len=target_len,
            is_first=is_first,
            is_last=is_last,
        )
        late_ratio = 0.2 if section_name == "tail" else 0.0
        spans.extend(
            _emit_section(
                builder,
                fmt=fmt,
                lang=lang,
                rng=rng,
                ctx=ctx,
                values=values_by_place[section_name],
                dense_tokens=dense_tokens,
                late_ratio=late_ratio,
            )
        )

    text = builder.build()
    if len(text) > target_len:
        text = text[:target_len]
        spans = [sp for sp in spans if sp.end <= target_len]

    spans = _compute_expected_coverage(text=text, spans=spans)
    return Sample(
        sample_id=sample_id,
        source_text=text,
        spans=spans,
        language=lang,
        fmt=fmt,
        length_bucket=bucket.name,
        placement_profile=placement_profile,
        seed=seed,
    )


def _validate_sample(sample: Sample) -> None:
    text = sample.source_text
    for sp in sample.spans:
        if sp.start < 0 or sp.end <= sp.start or sp.end > len(text):
            raise ValueError(f"invalid span bounds: {sp}")
        if sp.label not in CANONICAL_LABELS:
            raise ValueError(f"non-canonical label: {sp.label}")

    spans = sorted(sample.spans, key=lambda s: (s.start, s.end))
    for a, b in zip(spans, spans[1:], strict=False):
        if b.start < a.end:
            raise ValueError(f"overlapping spans: {a} and {b}")


def _sample_to_row(sample: Sample) -> dict[str, Any]:
    length_chars = len(sample.source_text)
    entity_count = sample.entity_count
    density = 0.0 if length_chars <= 0 else (entity_count / (length_chars / 10_000.0))
    return {
        "id": sample.sample_id,
        "source_text": sample.source_text,
        "privacy_mask": [
            {
                "start": int(sp.start),
                "end": int(sp.end),
                "label": sp.label,
                "value_type": sp.value_type,
                "placement": sp.placement,
                "expected_in_chunk_windows": bool(sp.expected_in_chunk_windows),
            }
            for sp in sample.spans
        ],
        "language": sample.language,
        "script_profile": classify_script_profile(sample.source_text),
        "format": sample.fmt,
        "length_chars": length_chars,
        "length_bucket": sample.length_bucket,
        "entity_count": entity_count,
        "pii_density_per_10k": round(density, 6),
        "placement_profile": sample.placement_profile,
        "generator_version": "v2",
        "seed": int(sample.seed),
    }


def generate_split(*, split: SplitName, total_rows: int, seed: int, buckets: tuple[BucketSpec, ...]) -> list[dict[str, Any]]:
    rng = Random(int(seed))
    bucket_counts = _alloc_bucket_counts(total_rows=total_rows, buckets=buckets)
    neg_total = int(round(total_rows * 0.10))
    neg_by_bucket = _alloc_negatives(bucket_counts, neg_total)

    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        count = int(bucket_counts.get(bucket.name, 0))
        if count <= 0:
            continue
        neg_count = int(neg_by_bucket.get(bucket.name, 0))

        for idx_in_bucket in range(count):
            is_negative = idx_in_bucket < neg_count
            lang = _pick_language(idx_in_bucket)
            fmt = _pick_format(idx_in_bucket)
            placement_profile = _pick_placement_profile(bucket.name, idx_in_bucket)
            row_seed = rng.randint(0, 2**31 - 1)
            sample_id = f"gr_longctx::{split}::{bucket.name}::{idx_in_bucket:04d}"
            sample: Sample | None = None
            for attempt in range(6):
                sample = generate_sample(
                    sample_id=sample_id,
                    bucket=bucket,
                    lang=lang,
                    fmt=fmt,
                    placement_profile=placement_profile,
                    is_negative=is_negative,
                    seed=row_seed + attempt,
                )
                if is_negative or sample.entity_count > 0:
                    break
            if sample is None:
                raise RuntimeError("failed to generate sample")
            _validate_sample(sample)
            rows.append(_sample_to_row(sample))

    if len(rows) != total_rows:
        raise RuntimeError(f"split generation produced wrong row count: got={len(rows)} want={total_rows}")
    return rows


def _audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = 0
    tail_95 = 0
    tail_total = 0
    total_spans = 0

    for row in rows:
        prof = row.get("placement_profile")
        bucket = row.get("length_bucket")
        length = int(row.get("length_chars") or 0)
        if length <= 0:
            continue
        margin = 0.03 if bucket in {"10k", "50k"} else 0.02
        head_max = (0.20 if bucket in {"10k", "50k"} else 0.15) + margin
        tail_min = (0.80 if bucket in {"10k", "50k"} else 0.85) - margin
        mid_low = (0.30 if bucket in {"10k", "50k"} else 0.35) - margin
        mid_high = (0.70 if bucket in {"10k", "50k"} else 0.65) + margin
        for sp in row.get("privacy_mask", []):
            total_spans += 1
            frac = float(sp.get("start", 0)) / float(length)
            if prof == "tail":
                tail_total += 1
                if frac >= 0.95:
                    tail_95 += 1
                if frac < tail_min:
                    violations += 1
            elif prof == "head":
                if frac > head_max:
                    violations += 1
            elif prof in {"middle", "middle_only"}:
                if frac < mid_low or frac > mid_high:
                    violations += 1

    tail_ratio = 0.0 if tail_total == 0 else tail_95 / tail_total
    return {
        "placement_violations": violations,
        "total_spans": total_spans,
        "tail_95_ratio": round(tail_ratio, 6),
        "tail_95_count": tail_95,
        "tail_total": tail_total,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _counter(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            val = r.get(key)
            if val is None:
                continue
            out[str(val)] = out.get(str(val), 0) + 1
        return out

    label_counts: dict[str, int] = {}
    for r in rows:
        for sp in r.get("privacy_mask", []):
            label = sp.get("label")
            if not label:
                continue
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1

    placement_hist: dict[str, list[int]] = {}
    for r in rows:
        prof = str(r.get("placement_profile"))
        placement_hist.setdefault(prof, [0] * 10)
        length = int(r.get("length_chars") or 0)
        if length <= 0:
            continue
        for sp in r.get("privacy_mask", []):
            frac = float(sp.get("start", 0)) / float(length)
            idx = min(9, max(0, int(frac * 10)))
            placement_hist[prof][idx] += 1

    neg = sum(1 for r in rows if not r.get("privacy_mask"))
    return {
        "rows": len(rows),
        "negatives": neg,
        "length_bucket_counts": _counter("length_bucket"),
        "format_counts": _counter("format"),
        "language_counts": _counter("language"),
        "placement_profile_counts": _counter("placement_profile"),
        "label_counts": label_counts,
        "placement_histogram_deciles": placement_hist,
    }


def _dataset_card_text(*, fast_rows: int, full_rows: int) -> str:
    return f"""---
pretty_name: Guardrails Long-Context PII (Synthetic v2)
license: apache-2.0
task_categories:
  - token-classification
language:
  - en
  - ru
tags:
  - pii
  - synthetic
  - benchmark
---

# Guardrails Long-Context PII (Synthetic v2)

This dataset is **fully synthetic** and was generated by Antimatter Guardrails for **stress-testing PII guardrails** on **very large requests** with more realistic format structure.

## Key Properties

- Up to **1,000,000 characters** per sample ("1m" bucket)
- Formats: logs, code-like text, and data dumps
- RU + EN coverage
- Span supervision via `privacy_mask` (character offsets)
- Includes **person** labels

## Splits

- `fast`: {fast_rows} rows (for quick profiling)
- `full`: {full_rows} rows (for deeper benchmarking)

## Schema

Required fields:

- `source_text: string`
- `privacy_mask: list[{{start:int, end:int, label:string, ...}}]`

Labels (canonical):

`person`, `email`, `phone`, `ip`, `url`, `identifier`, `date`, `location`, `secret`, `payment_card`

### Placement profiles

- `head`: spans placed within the first ~15% of the text
- `middle`: spans placed within the 35–65% window
- `tail`: spans placed within the last ~15% of the text (with some spans in the final 5%)
- `spread`: spans distributed across head/middle/tail
- `middle_only`: dense tokens + spans in the middle window

### Decoy policy

Unlabeled decoys are limited to timestamps and trace/request IDs. Address-like strings and date values are labeled when present.

### Coverage marker

`expected_in_chunk_windows` is set to `true` for all spans to reflect the intended fully-covering chunking behavior.

## Safety

All values are synthetic (no real PII is used).
"""


def main() -> int:
    args = _parse_args()
    _configure_hf_cache(args.cache_dir)

    if args.max_chars < 10_000:
        raise ValueError("--max-chars must be >= 10_000")

    buckets = _default_buckets(max_chars=int(args.max_chars))

    fast_rows = int(args.fast_rows)
    full_rows = int(args.full_rows)
    fast = generate_split(split="fast", total_rows=fast_rows, seed=int(args.seed), buckets=buckets)
    full = generate_split(split="full", total_rows=full_rows, seed=int(args.seed) + 1, buckets=buckets)

    audit_fast = _audit_rows(fast) if args.audit else {}
    audit_full = _audit_rows(full) if args.audit else {}
    if args.audit:
        fast_limit = max(5, int((audit_fast.get("total_spans") or 0) * 0.005))
        full_limit = max(5, int((audit_full.get("total_spans") or 0) * 0.005))
        if int(audit_fast.get("placement_violations") or 0) > fast_limit:
            raise RuntimeError(f"fast split placement violations: {audit_fast}")
        if int(audit_full.get("placement_violations") or 0) > full_limit:
            raise RuntimeError(f"full split placement violations: {audit_full}")
        if audit_full.get("tail_95_ratio", 0.0) < 0.05 and audit_full.get("tail_total", 0) > 0:
            raise RuntimeError(f"tail placement lacks late spans: {audit_full}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": str(args.repo_id),
        "seed": int(args.seed),
        "fast_rows": fast_rows,
        "full_rows": full_rows,
        "max_chars": int(args.max_chars),
        "generator_version": "v2",
        "fast_summary": _summarize_rows(fast),
        "full_summary": _summarize_rows(full),
        "fast_audit": audit_fast,
        "full_audit": audit_full,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not bool(args.push):
        (out_dir / "fast.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fast) + "\n", encoding="utf-8")
        (out_dir / "full.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in full) + "\n", encoding="utf-8")
        print(f"[ok] generated locally at {out_dir}")
        return 0

    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required (install with guardrails-service[eval]).") from exc

    try:
        from huggingface_hub import HfApi, get_token  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub package is required.") from exc

    token = get_token()
    if not token:
        raise RuntimeError("No HF auth token found. Run `hf auth login` or set HF_TOKEN.")

    ds_fast = datasets.Dataset.from_list(fast)
    ds_full = datasets.Dataset.from_list(full)
    dd = datasets.DatasetDict({"fast": ds_fast, "full": ds_full})

    print(f"[push] repo_id={args.repo_id} private={bool(args.private)}")
    dd.push_to_hub(
        repo_id=str(args.repo_id),
        private=bool(args.private),
        token=token,
        commit_message="Publish synthetic long-context PII dataset (v2)",
    )

    api = HfApi(token=token)
    card_path = out_dir / "README.md"
    card_path.write_text(_dataset_card_text(fast_rows=fast_rows, full_rows=full_rows), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=str(args.repo_id),
        repo_type="dataset",
        commit_message="Add dataset card (synthetic long-context PII v2)",
    )

    fresh = out_dir / "_verify_cache"
    fresh.mkdir(parents=True, exist_ok=True)
    ds1 = datasets.load_dataset(str(args.repo_id), split="fast", token=True, cache_dir=str(fresh))
    ds2 = datasets.load_dataset(str(args.repo_id), split="full", token=True, cache_dir=str(fresh))
    print(f"[verify] fast={len(ds1)} full={len(ds2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
