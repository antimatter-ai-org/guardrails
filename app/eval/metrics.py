from __future__ import annotations

from dataclasses import dataclass

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
    per_label_exact: dict[str, MetricCounts]


def _sum_counts(items: list[MetricCounts]) -> MetricCounts:
    return MetricCounts(
        true_positives=sum(item.true_positives for item in items),
        false_positives=sum(item.false_positives for item in items),
        false_negatives=sum(item.false_negatives for item in items),
    )


def evaluate_samples(
    dataset_samples: list[EvalSample],
    predictions_by_id: dict[str, list[EvalSpan]],
) -> EvaluationAggregate:
    exact_agnostic_items: list[MetricCounts] = []
    overlap_agnostic_items: list[MetricCounts] = []
    exact_canonical_items: list[MetricCounts] = []
    overlap_canonical_items: list[MetricCounts] = []

    all_labels: set[str] = set()
    for sample in dataset_samples:
        for span in sample.gold_spans:
            if span.canonical_label:
                all_labels.add(span.canonical_label)
        for span in predictions_by_id.get(sample.sample_id, []):
            if span.canonical_label:
                all_labels.add(span.canonical_label)

    per_label_counts: dict[str, list[MetricCounts]] = {label: [] for label in sorted(all_labels)}

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
        overlap_canonical_items.append(
            match_counts(
                gold_spans=canonical_gold,
                predicted_spans=canonical_pred,
                require_label=True,
                allow_overlap=True,
            )
        )

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

    return EvaluationAggregate(
        exact_agnostic=_sum_counts(exact_agnostic_items),
        overlap_agnostic=_sum_counts(overlap_agnostic_items),
        exact_canonical=_sum_counts(exact_canonical_items),
        overlap_canonical=_sum_counts(overlap_canonical_items),
        per_label_exact={label: _sum_counts(items) for label, items in per_label_counts.items()},
    )
