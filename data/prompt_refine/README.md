# Prompt refine (GCS + realtime Gemini)

Iterate on the Odia grounding prompt **before** submitting a Vertex batch job.

## What is omitted from the prompt

These metadata ids are **never** sent as locate/transcribe targets:

| id | reason |
|----|--------|
| `registration_no` | identifier, not useful page text to ground |
| `book_no` / book number / `book_label` | book classification, not page text |
| `listed_on` | wrong id — value is promoted to `execution_date` instead |

If metadata only has `listed_on`, it is sent as `execution_date` (label
"Execution date"). The id `listed_on` never appears in the prompt.

## Files

- `prompt.py` — system instruction + `build_user_prompt`
- `grounding_realtime_gemini.py` — sample from GCS, call Gemini realtime
- `grounding_batch_gemini.py` — same prompt/targets for the later batch job
- `test_prompt_targets.py` — smoke checks (no cloud needed)

## Run (direct GCS — no Render)

```bash
export GCS_BUCKET=classification-vision
export GCS_CREDENTIALS_JSON='...'   # service-account JSON
export GCS_RAW_PREFIX=ocr_outputs/orissa_deeds
# Vertex: same SA (or ADC) must call Gemini on GCP_PROJECT
export GCP_PROJECT=vision-projects-463307

# Dry-run: sample 10 deeds, download pages, write prompts only
python3 grounding_realtime_gemini.py --n 10 --dry-run \
  --out-dir data/prompt_refine

# Realtime Gemini on those 10
python3 grounding_realtime_gemini.py --n 10 \
  --out-dir data/prompt_refine

# Re-run a fixed list after prompt edits
python3 grounding_realtime_gemini.py \
  --deeds data/prompt_refine/sample_reg_nos.txt \
  --out-dir data/prompt_refine
```

Outputs under `data/prompt_refine/`:

- `sample_reg_nos.txt` / `sample_groundings.json`
- `prompts/<reg>__cN.txt` — exact prompt text sent
- `realtime_results.jsonl` / `realtime_fields.csv` / `realtime_summary.json`

When found-rates look solid, switch to `grounding_batch_gemini.py` for the
full corpus.
