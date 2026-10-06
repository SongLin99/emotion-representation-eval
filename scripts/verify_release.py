from __future__ import annotations

import csv
import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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

    revised=ROOT/'data/revisions/20261004'
    if revised.exists():
        sens=[json.loads(x) for x in (revised/'intensity_ladders.jsonl').read_text().splitlines()]
        para=[json.loads(x) for x in (revised/'paraphrase_pairs.jsonl').read_text().splitlines()]
        assert len(sens)==len(para)==600
        lookup={x['id']:x for x in sens};assert len(lookup)==600
        for x in para:
            assert all(x['text_'+l]==lookup[x['id']]['text_'+l] for l in ('en','zh'))
        for t in {x['topic_id'] for x in sens}:
            group=[x for x in sens if x['topic_id']==t]
            assert sorted(x['level'] for x in group)==list(range(5))
            assert all(len({x['shared_setup_'+l] for x in group})==1 for l in ('en','zh'))
        retest=ROOT/'results/retest_20261004'
        for name,n in [('functionality_judged.jsonl',300),('crosslingual_judged.jsonl',600),
                       ('small_dose_judged.jsonl',1176)]:
            rs=[json.loads(x) for x in (retest/name).read_text().splitlines()]
            assert len(rs)==n and len({x['id'] for x in rs})==n
            assert all(x['judging_protocol']=='anonymous_shuffled_quality_v2' for x in rs)
            assert all(x['judge']['id'].startswith('item_') for x in rs)
        with (retest/'readouts_l24.csv').open() as handle:
            scores=list(csv.DictReader(handle))
        assert len(scores)==5000 and {x['layer'] for x in scores}=={'24'}
        assert Counter(x['pooling'] for x in scores)=={'token50':3100,'continuation':1900}

    qwen=ROOT/'results/qwen25_14b_20261004'
    if qwen.exists():
        config_path=ROOT/'configs/qwen25_14b_l36_20261004.json'
        config=json.loads(config_path.read_text())
        summary=json.loads((qwen/'summary.json').read_text())
        assert summary['protocol_sha256']==hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert config['model']=='Qwen/Qwen2.5-14B-Instruct' and config['primary_layer']==36
        for language in ('en','zh'):
            folder=ROOT/'vectors/qwen25_14b_instruct'/language
            vector_file=folder/'emotion_vectors.safetensors'
            tensors=load_file(str(vector_file))
            assert set(tensors)=={'means','matched_neutral_means','raw','vectors','unit_vectors'}
            assert all(x.shape==(49,6,5120) and np.isfinite(x).all() for x in tensors.values())
            np.testing.assert_allclose(np.linalg.norm(tensors['unit_vectors'],axis=-1),1,atol=2e-5,rtol=2e-5)
            metadata=json.loads((folder/'manifest.json').read_text())
            assert metadata['packed_sha256']==hashlib.sha256(vector_file.read_bytes()).hexdigest()
            assert metadata['model_revision']==config['revision']
            assert metadata['constructs']==emotions
        norms=json.loads((ROOT/'vectors/qwen25_14b_instruct/en/neutral_token_norms.json').read_text())
        assert norms['all_real_token_mean_norm_by_layer'][36]==config['norm']
        exclusions=json.loads((ROOT/'data/vector_construction/qwen25_14b_pca_exclusions_20261004.json').read_text())
        assert exclusions['retained_count']==858 and exclusions['input_count']==1128
        controls=load_file(str(qwen/'random_controls.safetensors'))['random_controls']
        units=load_file(str(ROOT/'vectors/qwen25_14b_instruct/en/emotion_vectors.safetensors'))['unit_vectors'][36]
        assert controls.shape==(6,5120)
        np.testing.assert_allclose(np.linalg.norm(controls,axis=1),1,atol=2e-6)
        np.testing.assert_allclose(controls@units.T,0,atol=2e-6)
        with (qwen/'readouts_l36.csv').open() as handle:readouts=list(csv.DictReader(handle))
        assert len(readouts)==5000 and {r['layer'] for r in readouts}=={'36'}
        assert Counter(r['pooling'] for r in readouts)=={'token50':3100,'continuation':1900}
        missing=[r for r in readouts if r['valid']=='False']
        assert len(missing)==5 and all(r['language']=='zh' and int(r['token_count'])<50 for r in missing)
        assert all(all(r[k]=='' for k in r if k.startswith('cos_')) for r in missing)
        from evaluation.qwen_summary import usable_tables
        _,se,inv,excluded,complete=usable_tables(readouts)
        assert len(complete)==596
        assert len({r['topic_id'] for r in se if r['language']=='en'})==120
        assert len({r['topic_id'] for r in se if r['language']=='zh'})==117
        for name,n in (('functionality',300),('crosslingual',600),('small_dose',1176)):
            raw=[json.loads(x) for x in (qwen/(name+'_responses.jsonl')).read_text().splitlines()]
            judged=[json.loads(x) for x in (qwen/(name+'_judged.jsonl')).read_text().splitlines()]
            assert len(raw)==len(judged)==n
            assert len({r['id'] for r in raw})==len({r['id'] for r in judged})==n
            original={r['id']:r for r in raw}
            for row in judged:
                assert row['response']==original[row['id']]['response']
                assert row['generated_token_ids']==original[row['id']]['generated_token_ids']
                assert row['judging_protocol']=='anonymous_shuffled_quality_v2'
                assert row['judge']['id'].startswith('item_')
                for e in emotions:assert type(row['judge'][e]) is int and 0<=row['judge'][e]<=4
                for k in ('coherence','task_adherence','language_adherence'):
                    assert type(row['judge'][k]) is int and 1<=row['judge'][k]<=5

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
