# Prompt refine → batch on the 2500 GCS sample

## Workflow

1. We already sampled **2500 diverse deeds from GCS**  
   (`data/mismatches/gcs_diverse_sample.json` + `…_reg_nos.txt`).
2. Pick **10 of those 2500**, run **realtime Gemini**, refine `prompt.py`.
3. When found-rates look good, run **batch Gemini on all 2500**.

Render is not involved — pages and grounding come from GCS.

## What is omitted from the prompt

| id | behaviour |
|----|-----------|
| `registration_no` | never sent |
| `book_no` / book number / `book_label` | never sent |
| `listed_on` | never sent as that id — value promoted to `execution_date` |

## Files

- `prompt.py` — system instruction + `build_user_prompt`
- `grounding_realtime_gemini.py` — 10-of-2500 realtime refine loop
- `grounding_batch_gemini.py` — batch job (use for the full 2500)
- `test_prompt_targets.py` — smoke checks (no cloud needed)

## 1) Refine on 10 (from the 2500)

```bash
export GCS_BUCKET=classification-vision
export GCS_CREDENTIALS_JSON='...'   # service-account JSON
export GCS_RAW_PREFIX=ocr_outputs/orissa_deeds
export GCP_PROJECT=vision-projects-463307

# Default: pick 10 diverse regs FROM data/mismatches/gcs_diverse_sample.json
python3 grounding_realtime_gemini.py --n 10 --out-dir data/prompt_refine

# After editing prompt.py, re-run the SAME 10:
python3 grounding_realtime_gemini.py \
  --deeds data/prompt_refine/sample_reg_nos.txt \
  --out-dir data/prompt_refine
```

Outputs under `data/prompt_refine/`:

- `sample_reg_nos.txt` — the 10 chosen from the 2500
- `prompts/<reg>__cN.txt` — exact prompt text sent
- `realtime_results.jsonl` / `realtime_fields.csv` / `realtime_summary.json`

## 2) Batch the full 2500

Once the prompt is solid:

```bash
python3 grounding_batch_gemini.py \
  --deeds data/mismatches/gcs_diverse_sample_reg_nos.txt \
  --tag gcs2500
```

(Plus whatever `--pages-manifest` / `--metadata` paths your local batch
setup uses — see `grounding_batch_gemini.py`.)
