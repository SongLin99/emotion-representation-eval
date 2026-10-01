# Prompt provenance

Open-ended task prompts are preserved verbatim, paired in English and Chinese,
in `data/functionality/prompts.jsonl`. Layer, dosage, sampling and chat formatting
are documented in `data/functionality/functionality_protocol.json` and
`data/functionality/crosslingual_protocol.json`.

The optional new anonymous scoring rubric is `RUBRIC` in
`evaluation/judge_responses.py`. It is **not** the executed historical scoring
protocol: the archived judgments used condition-revealing IDs. Input stories and
paraphrases are frozen in `data/`; original vector/story construction scripts
are intentionally excluded from this evaluation-only release.
