"""Offline reproduction of frozen Qwen L36 metrics; no GPU or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from .utils import ROOT, EMOTIONS, read_csv, read_jsonl
from . import specificity, sensitivity, invariance
from .robustness import specificity_bootstrap, sensitivity_bootstrap, score_functionality, LEGACY_QUALITY_PROTOCOL
from .retest_summary import mapped_vector, pair_accuracy

EVAL = ROOT
OUT = ROOT / "results/qwen25_14b_20261004"
DATA = ROOT / "data/revisions/20261004"
CONFIG = ROOT / "configs/qwen25_14b_l36_20261004.json"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
LAYER = 36


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usable_tables(allrows):
    """Pre-score complete-case selection by native-token eligibility, never by values."""
    valid = lambda r: str(r.get("valid",True)).lower()=="true"
    primary = [r for r in allrows if r["pooling"]=="token50"]
    sp = [r for r in primary if r["task"]=="specificity" and valid(r)]
    original = [r for r in primary if r["task"]=="sensitivity"]
    missing_ladders = {(r["language"],r["topic_id"]) for r in original if not valid(r)}
    se = [r for r in original if valid(r) and (r["language"],r["topic_id"]) not in missing_ladders]
    candidates = [r for r in primary if r["task"] in ("sensitivity","invariance") and valid(r)]
    keys = {}
    for r in candidates:
        keys.setdefault(r["id"],set()).add((r["language"],r["version"]))
    complete = {i for i,k in keys.items() if k=={("en","original"),("zh","original"),("en","paraphrase"),("zh","paraphrase")}}
    inv = [r for r in candidates if r["id"] in complete]
    return sp,se,inv,missing_ladders,complete


def compute():
    protocol = json.loads(CONFIG.read_text())
    allrows = read_csv(OUT/"readouts_l36.csv")
    pick = lambda task,pool:[r for r in allrows if r["task"]==task and r["pooling"]==pool]
    sp,se,inv_rows,missing_ladders,complete_ids = usable_tables(allrows)
    primary = dict(specificity=specificity.analyze(sp),sensitivity=sensitivity.analyze(se),invariance=invariance.analyze(inv_rows))
    intervals = {}
    for lang in ("en","zh"):
        group = [r for r in sp if r["language"]==lang]
        primary["specificity"][lang]["matched_pair_accuracy"] = pair_accuracy(group)
        intervals[lang] = dict(specificity=specificity_bootstrap(group),sensitivity=sensitivity_bootstrap(primary["sensitivity"][lang]["records"]))
    robustness = dict(continuation_only=dict(specificity=specificity.analyze(pick("specificity","continuation")),
        sensitivity=sensitivity.analyze(pick("sensitivity","continuation"))),
        fixed_vector_language={vl:dict(specificity=specificity.analyze(mapped_vector(sp,vl)),invariance=invariance.analyze(mapped_vector(inv_rows,vl))) for vl in ("en","zh")},
        sensitivity_common_bilingual_ladders=sensitivity.analyze([r for r in se if r["topic_id"] not in {t for _,t in missing_ladders}]))
    result = dict(version="qwen25_14b_emotion_benchmark_20261004",model=MODEL_ID,revision=REVISION,primary_layer=LAYER,primary=primary,intervals=intervals,
        robustness=robustness,protocol_sha256=sha(CONFIG),protocol=protocol,
        coverage=dict(token50_missing=[{k:r[k] for k in ("id","task","language","version","token_count","missing_reason")} for r in allrows if str(r.get("valid",True)).lower()!="true"],
            sensitivity_excluded_ladders={lang:sorted(t for l,t in missing_ladders if l==lang) for lang in ("en","zh")},
            invariance_complete_pairs=len(complete_ids),invariance_expected_pairs=600,
            invariance_excluded_ids=sorted({r["id"] for r in pick("sensitivity","token50")}-complete_ids)),
        material_revision=json.loads((DATA/"final_materials_complete.json").read_text()),
        boundaries=protocol["boundaries"]+[protocol["missing_policy"],"Quality-valid subsets do not replace all-response effects","No subjective emotion or introspection claim"])
    for dataset,key in [("functionality","functionality"),("crosslingual","crosslingual_functionality"),("small_dose","small_dose_functionality")]:
        result[key] = score_functionality(read_jsonl(OUT/(dataset+"_judged.jsonl")), LEGACY_QUALITY_PROTOCOL)
    result["generation_quality"] = []
    for dataset in ("crosslingual","small_dose"):
        rows = read_jsonl(OUT/(dataset+"_judged.jsonl"))
        for lang in ("en","zh"):
            for dose in sorted({abs(r["alpha"]) for r in rows}):
                group = [r for r in rows if r["prompt_language"]==lang and abs(r["alpha"])==dose]
                result["generation_quality"].append(dict(prompt_language=lang,dose=dose,n=len(group),
                    replacement=sum("\ufffd" in r["response"] for r in group),truncated=sum(r["hit_token_limit"] for r in group),source=dataset))
    reference = json.loads((EVAL/"results/retest_20261004/summary.json").read_text())
    result["olmo_reference"] = dict(model=reference["model"],primary_layer=reference["primary_layer"],
        primary={k:reference["primary"][k] for k in ("specificity","sensitivity","invariance")},
        functionality=reference["functionality"]["conditions"],crosslingual_functionality=reference["crosslingual_functionality"]["conditions"],
        small_dose_functionality=reference["small_dose_functionality"]["conditions"],
        comparison_boundary="Same corrected benchmark materials and relative layer depth; different native chat defaults, model/vector/PCA fits and fixed norms; not isolated model-architecture causality")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new filename; never overwrite frozen results")
    result = compute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("QWEN SUMMARY COMPLETE", args.output)


if __name__ == "__main__":
    main()
