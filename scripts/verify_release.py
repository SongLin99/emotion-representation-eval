from __future__ import annotations

import csv
import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_JSONL = {
    "data/vector_construction/emotion_stories.jsonl": 6000,
    "data/vector_construction/matched_neutral.jsonl": 1000,
    "data/vector_construction/pca_neutral.jsonl": 1128,
    "data/specificity/stories.jsonl": 350,
    "data/sensitivity/intensity_ladders.jsonl": 600,
    "data/invariance/paraphrase_pairs.jsonl": 600,
    "data/functionality/prompts.jsonl": 12,
    "results/functionality/responses.jsonl": 300,
    "results/functionality/judged_responses.jsonl": 300,
    "results/crosslingual_functionality/responses.jsonl": 600,
    "results/crosslingual_functionality/judged_responses.jsonl": 600,
}


def rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    for relative, expected in EXPECTED_JSONL.items():
        actual = rows(ROOT / relative)
        assert actual == expected, (relative, actual, expected)
        records = [json.loads(line) for line in (ROOT / relative).read_text(encoding='utf-8').splitlines() if line.strip()]
        ids = [row.get('id', row.get('task_id')) for row in records]
        assert len(ids) == len(set(ids)), ('duplicate IDs', relative)
        if relative.startswith('data/') and 'text_en' in records[0]:
            assert all(row['text_en'].strip() and row['text_zh'].strip() for row in records)
            assert all('\ufffd' not in row['text_en']+row['text_zh'] for row in records)

    stories = [json.loads(line) for line in (ROOT/'data/vector_construction/emotion_stories.jsonl').read_text().splitlines()]
    neutral = [json.loads(line) for line in (ROOT/'data/vector_construction/matched_neutral.jsonl').read_text().splitlines()]
    reference_topics = {row['topic_id'] for row in neutral}
    assert len(reference_topics) == 50
    assert set(Counter(row['topic_id'] for row in neutral).values()) == {20}
    emotions = ['happiness', 'surprise', 'fear', 'disgust', 'anger', 'sadness']
    assert set(row['emotion'] for row in stories) == set(emotions)
    for emotion in emotions:
        counts = Counter(row['topic_id'] for row in stories if row['emotion'] == emotion)
        assert set(counts) == reference_topics and set(counts.values()) == {20}

    for language in ("en", "zh"):
        folder = ROOT / "vectors/olmo3_7b_instruct" / language
        tensors = load_file(str(folder / "emotion_vectors.safetensors"))
        assert tensors["unit_vectors"].shape == (33, 6, 4096)
        assert all(np.isfinite(value).all() for value in tensors.values())
        norms = np.linalg.norm(tensors["unit_vectors"], axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=2e-5, rtol=2e-5)
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["counts"] == {
            "constructs": 6000,
            "matched_neutral": 1000,
            "pca_legacy": 1128,
        }

    for relative, expected in {'results/specificity/item_scores_l24.csv':700,
                              'results/sensitivity/item_scores_l24.csv':1200,
                              'results/invariance/item_scores_l24.csv':600}.items():
        with (ROOT/relative).open(newline='', encoding='utf-8') as handle:
            assert len(list(csv.DictReader(handle))) == expected

    manifest_lines = (ROOT/'MANIFEST.sha256').read_text().splitlines()
    for line in manifest_lines:
        expected, relative = line.split('  ', 1)
        path = ROOT/relative
        assert path.is_file(), ('missing file', relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, ('checksum mismatch', relative)

    with (ROOT / "results/functionality/contrasts.csv").open(newline="") as handle:
        contrasts = list(csv.DictReader(handle))
    assert len(contrasts) == 72
    assert any(float(row["target_target_delta"]) <= 0 for row in contrasts)

    print(f"PASS: data counts, paired topics, all-layer vectors, negative results and {len(manifest_lines)} checksums verified")


if __name__ == "__main__":
    main()
