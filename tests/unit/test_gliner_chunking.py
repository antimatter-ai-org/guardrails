from __future__ import annotations

from app.runtime.gliner_chunking import GlinerChunkingConfig, build_chunk_windows, run_chunked_inference


def test_build_chunk_windows_short_text_single_window() -> None:
    text = "one two three"
    windows = build_chunk_windows(
        text=text,
        config=GlinerChunkingConfig(enabled=True, max_tokens=32, overlap_tokens=8),
    )

    assert len(windows) == 1
    assert windows[0].text_start == 0
    assert windows[0].text_end == len(text)
    assert windows[0].token_start == 0
    assert windows[0].token_end == 3


def test_build_chunk_windows_overlap_and_tail_coverage() -> None:
    tokens = [f"t{idx}" for idx in range(20)]
    text = " ".join(tokens)
    windows = build_chunk_windows(
        text=text,
        config=GlinerChunkingConfig(
            enabled=True,
            max_tokens=8,
            overlap_tokens=3,
            max_chunks=16,
            boundary_lookback_tokens=0,
        ),
    )

    assert len(windows) >= 3
    assert windows[0].token_start == 0
    assert windows[0].token_end == 8
    assert windows[1].token_start == 5  # 8 - overlap(3)
    assert windows[-1].token_end == 20

    for left, right in zip(windows, windows[1:], strict=False):
        assert right.token_start < right.token_end
        assert right.token_start <= left.token_end


def test_run_chunked_inference_dedupes_and_keeps_max_score() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    target_gamma_start = text.index("gamma")
    target_delta_start = text.index("delta")

    def _predict_batch(texts: list[str], labels: list[str], threshold: float) -> list[list[dict[str, object]]]:
        outputs: list[list[dict[str, object]]] = []
        for chunk in texts:
            preds: list[dict[str, object]] = []
            gamma_idx = chunk.find("gamma")
            if gamma_idx >= 0:
                score = 0.6 if chunk.startswith("alpha") else 0.9
                preds.append(
                    {
                        "start": gamma_idx,
                        "end": gamma_idx + len("gamma"),
                        "label": "person",
                        "score": score,
                    }
                )
            delta_idx = chunk.find("delta")
            if delta_idx >= 0:
                preds.append(
                    {
                        "start": delta_idx,
                        "end": delta_idx + len("delta"),
                        "label": "organization",
                        "score": 0.7,
                    }
                )
            outputs.append(preds)
        return outputs

    merged = run_chunked_inference(
        text=text,
        labels=["person", "organization"],
        threshold=0.5,
        chunking=GlinerChunkingConfig(
            enabled=True,
            max_tokens=4,
            overlap_tokens=2,
            max_chunks=8,
            boundary_lookback_tokens=0,
        ),
        predict_batch=_predict_batch,
    )

    assert len([item for item in merged if item["label"] == "person"]) == 1
    gamma = next(item for item in merged if item["label"] == "person")
    assert gamma["start"] == target_gamma_start
    assert gamma["end"] == target_gamma_start + len("gamma")
    assert gamma["score"] == 0.9

    delta = next(item for item in merged if item["label"] == "organization")
    assert delta["start"] == target_delta_start
    assert delta["end"] == target_delta_start + len("delta")
