from __future__ import annotations

from typing import Any

from app.eval.datasets.base import DatasetAdapter
from app.eval.labels import canonicalize_scanpatch_gold_label
from app.eval.types import EvalSample, EvalSpan


def _reconstruct_text_from_tokens(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for index, token in enumerate(tokens):
        if index > 0:
            text_parts.append(" ")
            cursor += 1

        start = cursor
        text_parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))

    return "".join(text_parts), offsets


def _build_spans_from_bio_tags(
    tokens: list[str],
    tags: list[str],
) -> tuple[str, list[EvalSpan]]:
    text, offsets = _reconstruct_text_from_tokens(tokens)

    spans: list[EvalSpan] = []
    current_label: str | None = None
    current_start_token: int | None = None
    current_end_token: int | None = None

    def flush() -> None:
        nonlocal current_label, current_start_token, current_end_token
        if current_label is None or current_start_token is None or current_end_token is None:
            return
        start = offsets[current_start_token][0]
        end = offsets[current_end_token][1]
        spans.append(
            EvalSpan(
                start=start,
                end=end,
                label=current_label,
                canonical_label=canonicalize_scanpatch_gold_label(current_label),
            )
        )
        current_label = None
        current_start_token = None
        current_end_token = None

    for idx, raw_tag in enumerate(tags):
        tag = str(raw_tag)
        if tag == "O" or tag == "":
            flush()
            continue

        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = "B", tag

        if prefix == "B":
            flush()
            current_label = label
            current_start_token = idx
            current_end_token = idx
            continue

        if prefix == "I":
            if current_label == label and current_end_token is not None:
                current_end_token = idx
            else:
                flush()
                current_label = label
                current_start_token = idx
                current_end_token = idx
            continue

        flush()

    flush()
    return text, spans


class ScanpatchSyntheticControlledAdapter(DatasetAdapter):
    @property
    def dataset_name(self) -> str:
        return "scanpatch/pii-ner-corpus-synthetic-controlled"

    @staticmethod
    def _extract_tag_names(dataset: Any) -> list[str] | None:
        try:
            feature = dataset.features["ner_tags"]
        except Exception:
            return None

        names = getattr(getattr(feature, "feature", None), "names", None)
        if isinstance(names, list):
            return [str(item) for item in names]
        return None

    def load_samples(
        self,
        split: str,
        cache_dir: str,
        hf_token: str | None,
        max_samples: int | None = None,
    ) -> list[EvalSample]:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("datasets package is required. Install with guardrails-service[eval].") from exc

        dataset = load_dataset(
            self.dataset_name,
            split=split,
            token=hf_token,
            cache_dir=cache_dir,
        )
        tag_names = self._extract_tag_names(dataset)

        samples: list[EvalSample] = []
        for idx, row in enumerate(dataset):
            row_id = str(row.get("id", idx))

            if "tokens" in row and "ner_tags" in row:
                tokens = [str(item) for item in row["tokens"]]
                raw_tags = row["ner_tags"]
                tags: list[str] = []
                for tag_item in raw_tags:
                    if isinstance(tag_item, int) and tag_names is not None and 0 <= tag_item < len(tag_names):
                        tags.append(tag_names[tag_item])
                    else:
                        tags.append(str(tag_item))

                text, spans = _build_spans_from_bio_tags(tokens, tags)
                samples.append(
                    EvalSample(
                        sample_id=row_id,
                        text=text,
                        gold_spans=spans,
                        metadata={
                            "source": row.get("source"),
                            "noisy": row.get("noisy"),
                            "__split__": split,
                        },
                    )
                )
            elif {"text", "entity_starts", "entity_ends", "entity_labels"}.issubset(row.keys()):
                text = str(row["text"])
                starts = row["entity_starts"]
                ends = row["entity_ends"]
                labels = row["entity_labels"]

                spans: list[EvalSpan] = []
                for start, end, label in zip(starts, ends, labels, strict=False):
                    label_str = str(label)
                    spans.append(
                        EvalSpan(
                            start=int(start),
                            end=int(end),
                            label=label_str,
                            canonical_label=canonicalize_scanpatch_gold_label(label_str),
                        )
                    )
                samples.append(
                    EvalSample(
                        sample_id=row_id,
                        text=text,
                        gold_spans=spans,
                        metadata={
                            "source": row.get("source"),
                            "noisy": row.get("noisy"),
                            "__split__": split,
                        },
                    )
                )
            elif "text" in row and "entities" in row:
                text = str(row["text"])
                spans: list[EvalSpan] = []
                for item in row["entities"]:
                    label = str(item.get("label", "UNKNOWN"))
                    spans.append(
                        EvalSpan(
                            start=int(item["start"]),
                            end=int(item["end"]),
                            label=label,
                            canonical_label=canonicalize_scanpatch_gold_label(label),
                        )
                    )
                samples.append(
                    EvalSample(
                        sample_id=row_id,
                        text=text,
                        gold_spans=spans,
                        metadata={
                            "source": row.get("source"),
                            "noisy": row.get("noisy"),
                            "__split__": split,
                        },
                    )
                )
            else:
                raise RuntimeError("Unsupported dataset row format for Scanpatch adapter")

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples
