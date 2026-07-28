"""Odia deed grounding / transcription prompts for Gemini.

LOCATE each known English metadata value on the page images, then TRANSCRIBE
it verbatim in the source script (Odia / Latin / mixed). Never translate or
invent glyphs.

Fields intentionally NOT sent as grounding targets (see OMITTED_TARGET_IDS):
  - registration_no
  - book_no / book number / book_label
  - listed_on   (wrong id — value is promoted to execution_date instead)

If metadata only has ``listed_on``, ``targets_from_tabular`` renames that
value to ``execution_date`` before building the prompt. Prefer setting
``execution_date`` explicitly when you already have it.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Fields we never inject into the prompt as locate/transcribe targets
# ---------------------------------------------------------------------------
OMITTED_TARGET_IDS = frozenset({
    "registration_no",
    "registrationNo",
    "reg_no",
    "book_no",
    "book_number",
    "bookNumber",
    "book_label",
    "deedBookNo",
    "listed_on",
    "listedOn",
})

SYSTEM_INSTRUCTION = """\
You are an expert reader of historical land-registration deeds from Odisha, India. These scanned pages may be handwritten or typed, faint, skewed, smudged or noisy, and they are written in different scripts: some in Odia, some in English, some mixing both. You are a careful TRANSCRIBER, not a translator: you copy the exact characters that are physically on the page, in whatever script they are written. You never invent, translate, transliterate into another script, normalize, or 'correct' text, and you never report a value you cannot actually see.
"""

# Response object shape (also mirrored in grounding_batch_gemini.response_schema).
FIELD_RESULT_KEYS = (
    "id", "item_index", "attr", "english_value", "found", "odia_text",
    "script", "page", "confidence", "latin_readback", "notes",
)


def _fields_json(targets: list[dict]) -> str:
    """Serialize targets for the METADATA FIELDS block (omit empty / omitted ids)."""
    rows = []
    for t in targets:
        fid = str(t.get("id") or "")
        if fid in OMITTED_TARGET_IDS:
            continue
        val = t.get("value")
        if val is None or str(val).strip() == "":
            continue
        row: dict[str, Any] = {
            "id": fid,
            "label": t.get("label") or fid,
            "english_value": str(val).strip(),
            "type": t.get("type") or "text",
        }
        if t.get("format"):
            row["format"] = t["format"]
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_user_prompt(
    targets: list[dict],
    n_pages: int,
    deed_type: str = "",
    page_offset: int = 0,
) -> str:
    """Build the per-request user prompt.

    ``page_offset`` is unused in the text (pages are always numbered 1..n_pages
    local to THIS request) but kept for call-site compatibility with the batch
    builder, which tracks absolute page numbers in the sidecar meta.
    """
    del page_offset  # local page numbering only; see PAGE PROVENANCE below
    where = (
        "shown below as consecutive images in order"
        if n_pages > 1
        else "shown below as a single image"
    )
    deed = (deed_type or "unknown").strip() or "unknown"
    fields_json = _fields_json(targets)

    return f"""\
Deed category: {deed}. You are given {n_pages} page image(s) - {where}. For the "page" field, number the images 1..{n_pages} in THE ORDER SHOWN HERE (local to THIS request); do not use the deed's absolute page numbers.

You are given {n_pages} scanned page image(s) of ONE land deed, followed by a list of
METADATA FIELDS. Most fields carry a single known value in ENGLISH; a few (seller / buyer /
property) carry a SEMI-STRUCTURED LIST of several values (see LIST / COMPOSITE FIELDS below).
The English value(s) are used only to LOCATE text. Your task is to FIND where each value
appears in the images and TRANSCRIBE the exact text as written there, IN THE SAME SCRIPT the
page uses.

============================ CORE PRINCIPLE ============================
LOCATE, then TRANSCRIBE VERBATIM IN THE SOURCE SCRIPT. The English value only tells you WHAT
to look for; it is NOT the answer and NOT the format. Copy the exact ink on the page:
  - if the page writes the value in Odia  -> transcribe the Odia glyphs,
  - if the page writes the value only in English/Latin -> transcribe those English/Latin
    characters exactly as printed.
Ground truth = what is physically on the page, in its own script. Never translate, never
transliterate the English into Odia yourself, and never invent a spelling that is not there.

============================ HOW TO MATCH ==============================
- NAMES (person/relation): the English is a transliteration. Find the matching word on the
  page and copy it exactly - Odia glyphs if written in Odia (e.g. "SUHANI" -> ସୁହାନୀ), or the
  Latin form if the deed only prints it in English (e.g. "SUHANI" -> Suhani).
- PLACE NAMES (village / AT / PO / PS / district): copy the on-page spelling in its script.
- DEED TYPE / RELATION words: copy the term as written, INCLUDING short forms actually on the
  page (e.g. ପି:/ପିତା for father, ସ୍ୱା:/ସ୍ତ୍ରୀ for spouse, ଉଇଲ for WILL). Do not expand an
  abbreviation that is written short.
- NUMBERS, AMOUNTS, DATES: transcribe VERBATIM in whatever script the page uses. Odia digits
  (୦୧୨୩୪୫୬୭୮୯) stay Odia; Arabic digits (0-9) stay Arabic; words stay words. NEVER convert
  between numeral systems and never reformat separators (keep dots/slashes/hyphens exactly,
  e.g. ୨୬.୪.୮୮ or 26-4-88 as shown).
- EXECUTION DATE (id "execution_date"): this is the date the deed was EXECUTED / signed by
  the parties (often near the executant signatures or the opening body), NOT the registration
  stamp date and NOT the presentation/endorsement date. Do not confuse it with registration
  or presentation dates when those also appear on the page.

===================== SOURCE-SCRIPT FIDELITY (script) =================
Record which script you actually transcribed, in "script":
  - "odia"  : you copied Odia-script glyphs from the page.
  - "latin" : the value is written only in English/Latin on the page and you copied that.
  - "mixed" : the single value genuinely contains both scripts together.
If the SAME value appears in BOTH scripts on the page, prefer the Odia rendering (set
"script":"odia") and mention in "notes" that an English form also exists. Copying English that
IS on the page is correct and expected for English deeds; inventing Odia that is NOT on the
page is always wrong.

======================= ONE FIELD = ONE ENTITY ========================
Deeds list several people (first party, second party, relations, boundary owners). Make sure
the text you return belongs to the SPECIFIC field asked for, in the SAME row/clause as its
label. Do NOT grab a similar word belonging to a different person or a different plot. If two
people share a name, use position/context to pick the correct one.

================= LIST / COMPOSITE FIELDS (seller / buyer / property) =========
Some fields are NOT a single value but a SEMI-STRUCTURED LIST describing several people or
plots, given in a compact database format (the field also carries a "format" hint). Examples:
  seller_details / buyer_details ("type":"party_list"):
    "1-<NAME>  ( RELATION : )<REL>  ( RELATION NAME : )<GUARDIAN>  ( ADDRESS : )<ADDR> ,2-<NAME> ..."
  property_details ("type":"property_list"):
    "1- Village : <V>  Khata : <K> Plot : <P> Area: <A> Total Area: <TA> Boundary : <B> ,2- ..."
For each LIST field:
  1. SPLIT it into its numbered items (1-, 2-, 3-, ...).
  2. For every item, pull out the meaningful sub-values it contains - for parties: the person
     NAME, the RELATION NAME (guardian/father/spouse name), the ADDRESS; for property: the
     VILLAGE, KHATA, PLOT, AREA. (The relation TYPE word like "Others" is a generic label -
     skip it; see GENERIC below.)
  3. Then, exactly as for a scalar field, LOCATE each sub-value on the page and TRANSCRIBE it
     verbatim in the source script.
Emit ONE result object PER SUB-VALUE you attempt, setting:
  - "id"            = the list field's id (e.g. "seller_details") - repeated across its rows,
  - "item_index"    = which numbered item it came from (1, 2, ...),
  - "attr"          = the sub-value: "name" | "relation_name" | "address" | "village" |
                      "khata" | "plot" | "area",
  - "english_value" = the EXACT sub-string you took from the input for that attr,
  - plus found / odia_text / script / page / confidence / latin_readback / notes as usual.
Skip sub-values that are empty or generic; do not emit a row for a person/plot that has no
usable sub-value. For a normal SCALAR field, return exactly ONE row with "item_index": 0,
"attr": "" and "english_value" set to the single value you were given.

================= GENERIC / NON-SPECIFIC ENGLISH VALUES ================
Some english_value entries are generic database placeholders, not real words to search for -
e.g. "Others", "NA", "N/A", "Nil", "-", or blank. These give you no phonetic or semantic anchor,
so you CANNOT reliably verify a specific matching word on the page against them. For such
fields, ALWAYS set "found": false, "odia_text": "", "script": "", with "notes": "english_value
is a generic placeholder ('Others'/'NA'/etc.), cannot be verified against a specific word".
This applies with NO exception to "type": "relation_word" fields: even though there is usually
ONE relation marker (e.g. ପି:/ସ୍ୱା:) written next to that person's name, do NOT report it when
english_value is a generic placeholder - the database gave you no way to confirm WHICH relation
it intended, so any specific marker you pick is an unverifiable guess, not a located match.

======================= PAGE PROVENANCE (page) =======================
For EVERY field you locate you MUST record WHICH PAGE it came from, because this
mapping is used downstream to train a page-aware model. In "page" put the 1-based
number of the image (counting in the exact order the page images were given to you
in THIS request) on which the value physically appears.
  - found=true  -> "page" MUST be a real page number >= 1 (never 0).
  - found=false -> "page" = 0.
  - if the same value appears on several pages, report the page of the single
    clearest, most complete occurrence (the same one you transcribed).
Do not guess a page you did not actually read the value from; the page number must
point to where the transcribed text really is.

===================== WHEN YOU CANNOT FIND IT =========================
If the value is genuinely not visible in ANY script (absent, torn, fully illegible), set
"found": false, "odia_text": "", "script": "", "page": 0, "confidence": 0.0, and say why in
"notes". NEVER guess, never fabricate, never approximate from the English. A truthful "not
found" is better than a wrong transcription.

=========================== SELF-CHECK ================================
Before finalizing each field, fill "latin_readback" with a Latin reading of "odia_text" (for
Odia, your phonetic transliteration; for a value already in Latin, repeat it), and compare it
to english_value:
  - If they clearly match in sound/meaning -> keep it, confidence high.
  - If they do NOT match -> you likely located the wrong text; re-scan, fix it, or mark not
    found. Do not report text whose readback contradicts the English.

============================= OUTPUT =================================
Rules:
1. Copy character-for-character: preserve every matra (vowel sign), conjunct, nukta and space.
2. Set "script" to "odia" / "latin" / "mixed" to match what you transcribed (see above).
3. "page" = 1-based index of the image (in the order given) where the text appears -
   REQUIRED for every found field (>= 1); use 0 only when found=false. See PAGE PROVENANCE.
4. "confidence" (0.0-1.0): calibrate honestly - high only when the characters are clear AND the
   readback matches the English; lower it for faint/handwritten/ambiguous cases.
5. If a value appears more than once, report the single clearest, most complete occurrence.
6. SCALAR fields: return each id EXACTLY once ("item_index":0, "attr":""). LIST fields
   (seller/buyer/property): return ONE object per located sub-value - the same id repeated,
   distinguished by "item_index" + "attr". Output ONLY a JSON array matching the schema,
   nothing else.

METADATA FIELDS (find each value on the page and copy it in its own script):
{fields_json}
"""


def filter_targets(targets: list[dict]) -> list[dict]:
    """Drop omitted ids and empty values before prompt / request build."""
    out = []
    for t in targets:
        fid = str(t.get("id") or "")
        if fid in OMITTED_TARGET_IDS:
            continue
        val = t.get("value")
        if val is None or str(val).strip() == "":
            continue
        # Safety: never send listed_on under any label
        if fid.lower() in {"listed_on", "listedon"}:
            continue
        out.append(t)
    return out
