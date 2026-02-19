from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re

from app.eval.types import EvalSample, EvalSpan, MetricCounts


def _spans_overlap(a: EvalSpan, b: EvalSpan) -> bool:
    return a.start < b.end and b.start < a.end


def _is_match(pred: EvalSpan, gold: EvalSpan, require_label: bool, allow_overlap: bool) -> bool:
    if require_label and pred.canonical_label != gold.canonical_label:
        return False

    if allow_overlap:
        return _spans_overlap(pred, gold)
    return pred.start == gold.start and pred.end == gold.end


def match_counts(
    gold_spans: list[EvalSpan],
    predicted_spans: list[EvalSpan],
    *,
    require_label: bool,
    allow_overlap: bool,
) -> MetricCounts:
    # Exact matching can be done with multiset intersections and is much faster than pairwise scanning.
    if not allow_overlap:
        if require_label:
            gold_keys = Counter(
                (span.start, span.end, span.canonical_label)
                for span in gold_spans
                if span.canonical_label is not None
            )
            pred_keys = Counter(
                (span.start, span.end, span.canonical_label)
                for span in predicted_spans
                if span.canonical_label is not None
            )
        else:
            gold_keys = Counter((span.start, span.end) for span in gold_spans)
            pred_keys = Counter((span.start, span.end) for span in predicted_spans)

        true_positives = sum(min(count, gold_keys.get(key, 0)) for key, count in pred_keys.items())
        return MetricCounts(
            true_positives=true_positives,
            false_positives=len(predicted_spans) - true_positives,
            false_negatives=len(gold_spans) - true_positives,
        )

    used_gold: set[int] = set()
    true_positives = 0

    for prediction in predicted_spans:
        for gold_idx, gold in enumerate(gold_spans):
            if gold_idx in used_gold:
                continue
            if _is_match(prediction, gold, require_label=require_label, allow_overlap=allow_overlap):
                used_gold.add(gold_idx)
                true_positives += 1
                break

    false_positives = len(predicted_spans) - true_positives
    false_negatives = len(gold_spans) - true_positives
    return MetricCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


@dataclass(slots=True)
class EvaluationAggregate:
    exact_agnostic: MetricCounts
    overlap_agnostic: MetricCounts
    exact_canonical: MetricCounts
    overlap_canonical: MetricCounts
    char_canonical: MetricCounts
    token_canonical: MetricCounts
    per_label_exact: dict[str, MetricCounts]
    per_label_char: dict[str, MetricCounts]


def _sum_counts(items: list[MetricCounts]) -> MetricCounts:
    return MetricCounts(
        true_positives=sum(item.true_positives for item in items),
        false_positives=sum(item.false_positives for item in items),
        false_negatives=sum(item.false_negatives for item in items),
    )


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((start, end) for start, end in ranges if end > start)
    if not cleaned:
        return []

    merged: list[tuple[int, int]] = []
    cur_start, cur_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _range_len(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _intersection_len(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        left = max(a_start, b_start)
        right = min(a_end, b_end)
        if right > left:
            total += right - left
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return total


def _ranges_to_counts(gold_ranges: list[tuple[int, int]], pred_ranges: list[tuple[int, int]]) -> MetricCounts:
    gold_merged = _merge_ranges(gold_ranges)
    pred_merged = _merge_ranges(pred_ranges)
    tp = _intersection_len(gold_merged, pred_merged)
    gold_len = _range_len(gold_merged)
    pred_len = _range_len(pred_merged)
    return MetricCounts(
        true_positives=tp,
        false_positives=max(0, pred_len - tp),
        false_negatives=max(0, gold_len - tp),
    )


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _ranges_to_token_ids(ranges: list[tuple[int, int]], tokens: list[tuple[int, int]]) -> set[int]:
    merged = _merge_ranges(ranges)
    token_ids: set[int] = set()
    if not merged or not tokens:
        return token_ids

    r_idx = 0
    for idx, (tok_start, tok_end) in enumerate(tokens):
        while r_idx < len(merged) and merged[r_idx][1] <= tok_start:
            r_idx += 1
        if r_idx >= len(merged):
            break
        cur_start, cur_end = merged[r_idx]
        if cur_start < tok_end and tok_start < cur_end:
            token_ids.add(idx)
    return token_ids


def _token_counts(
    *,
    text: str,
    gold_ranges: list[tuple[int, int]],
    pred_ranges: list[tuple[int, int]],
) -> MetricCounts:
    tokens = _token_spans(text)
    gold_tokens = _ranges_to_token_ids(gold_ranges, tokens)
    pred_tokens = _ranges_to_token_ids(pred_ranges, tokens)
    tp = len(gold_tokens & pred_tokens)
    return MetricCounts(
        true_positives=tp,
        false_positives=max(0, len(pred_tokens) - tp),
        false_negatives=max(0, len(gold_tokens) - tp),
    )


def evaluate_samples(
    dataset_samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
    *,
    include_overlap: bool = True,
    include_per_label: bool = True,
) -> EvaluationAggregate:
    exact_agnostic_items: list[MetricCounts] = []
    overlap_agnostic_items: list[MetricCounts] = []
    exact_canonical_items: list[MetricCounts] = []
    overlap_canonical_items: list[MetricCounts] = []
    char_canonical_items: list[MetricCounts] = []
    token_canonical_items: list[MetricCounts] = []

    all_labels: set[str] = set()
    for sample in dataset_samples:
        for span in sample.gold_spans:
            if span.canonical_label:
                all_labels.add(span.canonical_label)
        for span in predictions_by_id.get(sample.sample_id, []):
            if span.canonical_label:
                all_labels.add(span.canonical_label)

    per_label_counts: dict[str, list[MetricCounts]] = {label: [] for label in sorted(all_labels)} if include_per_label else {}
    per_label_char_counts: dict[str, list[MetricCounts]] = (
        {label: [] for label in sorted(all_labels)} if include_per_label else {}
    )

    for sample in dataset_samples:
        predicted = predictions_by_id.get(sample.sample_id, [])
        gold = sample.gold_spans

        exact_agnostic_items.append(
            match_counts(
                gold_spans=gold,
                predicted_spans=predicted,
                require_label=False,
                allow_overlap=False,
            )
        )
        if include_overlap:
            overlap_agnostic_items.append(
                match_counts(
                    gold_spans=gold,
                    predicted_spans=predicted,
                    require_label=False,
                    allow_overlap=True,
                )
            )

        canonical_gold = [item for item in gold if item.canonical_label is not None]
        canonical_pred = [item for item in predicted if item.canonical_label is not None]
        exact_canonical_items.append(
            match_counts(
                gold_spans=canonical_gold,
                predicted_spans=canonical_pred,
                require_label=True,
                allow_overlap=False,
            )
        )
        if include_overlap:
            overlap_canonical_items.append(
                match_counts(
                    gold_spans=canonical_gold,
                    predicted_spans=canonical_pred,
                    require_label=True,
                    allow_overlap=True,
                )
            )
        gold_ranges = [(item.start, item.end) for item in canonical_gold]
        pred_ranges = [(item.start, item.end) for item in canonical_pred]
        char_canonical_items.append(_ranges_to_counts(gold_ranges, pred_ranges))
        token_canonical_items.append(
            _token_counts(
                text=sample.text,
                gold_ranges=gold_ranges,
                pred_ranges=pred_ranges,
            )
        )

        if include_per_label:
            for label in per_label_counts:
                per_label_gold = [item for item in canonical_gold if item.canonical_label == label]
                per_label_pred = [item for item in canonical_pred if item.canonical_label == label]
                per_label_counts[label].append(
                    match_counts(
                        gold_spans=per_label_gold,
                        predicted_spans=per_label_pred,
                        require_label=True,
                        allow_overlap=False,
                    )
                )
                per_label_char_counts[label].append(
                    _ranges_to_counts(
                        [(item.start, item.end) for item in per_label_gold],
                        [(item.start, item.end) for item in per_label_pred],
                    )
                )

    zero = MetricCounts(true_positives=0, false_positives=0, false_negatives=0)

    return EvaluationAggregate(
        exact_agnostic=_sum_counts(exact_agnostic_items),
        overlap_agnostic=_sum_counts(overlap_agnostic_items) if include_overlap else zero,
        exact_canonical=_sum_counts(exact_canonical_items),
        overlap_canonical=_sum_counts(overlap_canonical_items) if include_overlap else zero,
        char_canonical=_sum_counts(char_canonical_items),
        token_canonical=_sum_counts(token_canonical_items),
        per_label_exact={label: _sum_counts(items) for label, items in per_label_counts.items()} if include_per_label else {},
        per_label_char={label: _sum_counts(items) for label, items in per_label_char_counts.items()} if include_per_label else {},
    )
