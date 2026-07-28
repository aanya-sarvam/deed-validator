#!/usr/bin/env python3
"""Smoke checks for prompt target filtering (no GCS / Gemini required)."""

from __future__ import annotations

import sys

import grounding_batch_gemini as batch
import prompt as grounding_prompt


def test_omitted_ids_never_sent():
    meta = {
        "registration_no": "910010000401",
        "book_no": "1",
        "book_label": "SALE",
        "listed_on": "11-Feb-2000",
        "deed_type": "SALE IMMOVABLE",
        "district": "ANGUL",
        "office": "ANGUL",
        "registration_date": "22-May-2000",
        "presentation_date": "11-Feb-2000",
        "execution_date": "11-Feb-2000",
        "consideration_amount": "6600",
        "old_reg_no": "401",
    }
    raw = {
        "sellerDetails": "1-NIDRA PRADHAN   ( RELATION : )  Others  "
                         "(  RELATION NAME : )  Pandab Pradhan  "
                         "(  ADDRESS : )  Dhokuta ",
        "buyerDetails": "",
        "propertyDetails": "1- Village : KAHNEINAGAR  Khata : 38 Plot : 73 "
                           "Area: A 0.12 Total Area: A 0.12 Boundary : ",
    }
    targets = batch.build_targets(meta, raw)
    ids = {t["id"] for t in targets}
    leaked = ids & grounding_prompt.OMITTED_TARGET_IDS
    assert not leaked, f"omitted ids leaked: {leaked}"
    assert "registration_no" not in ids
    assert "book_no" not in ids
    assert "listed_on" not in ids
    assert "execution_date" in ids
    assert "deed_type" in ids
    assert "seller_details" in ids
    assert "property_details" in ids
    # Prompt text must not ask the model to locate omitted fields
    text = grounding_prompt.build_user_prompt(targets, n_pages=2, deed_type="SALE")
    assert "registration_no" not in text
    assert "listed_on" not in text
    assert "execution_date" in text
    assert "Execution date" in text
    print("ok: omitted ids filtered; execution_date present")


def test_listed_on_promoted_to_execution_date():
    """listed_on is omitted as an id; its value is sent as execution_date."""
    meta = {
        "listed_on": "11-Feb-2000",
        "deed_type": "SALE IMMOVABLE",
        "district": "ANGUL",
    }
    targets = batch.build_targets(meta, None)
    by_id = {t["id"]: t["value"] for t in targets}
    assert "listed_on" not in by_id
    assert by_id.get("execution_date") == "11-Feb-2000"
    print("ok: listed_on promoted to execution_date")


if __name__ == "__main__":
    test_omitted_ids_never_sent()
    test_listed_on_promoted_to_execution_date()
    print("all smoke checks passed")
    sys.exit(0)
