"""
sarvam_client.py — thin client for the Sarvam / IGR deed APIs from the
shared Postman collection:

  POST /api/User/GetUserAuthencate
  POST /api/Deed/GetDeedRegNoDetail
  POST /api/Deed/GetDeedInfoByRegNo
  POST /api/Deed/GetDeedScanCopy

Configure with env vars (or pass into SarvamClient):

  SARVAM_BASE_URL   e.g. https://example.gov.in
  SARVAM_LOGIN_ID   default from the shared Postman collection
  SARVAM_PASSWORD   default from the shared Postman collection
  SARVAM_USER_TYPE  default "SU"
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any

import requests
from urllib3.exceptions import InsecureRequestWarning

DEFAULT_LOGIN_ID = "$@rv@m@2026"
DEFAULT_PASSWORD = "Sarvam@2026"
DEFAULT_USER_TYPE = "SU"

# Envelope keys that are transport metadata, not deed fields.
_ENVELOPE_KEYS = {
    "data", "code", "message", "information", "status", "success",
    "error", "errors", "registration_no", "_raw", "_empty", "_http_status",
}


def is_empty_deed_payload(deed: Any) -> bool:
    """True when GetDeedInfoByRegNo has no usable deed fields.

    Live IGR shape:
      {"data": {serial, district, office, ...}, "code": 0, "information": null}
    Empty / removed deeds:
      {"data": null, "code": 0, "information": null}
    """
    if not isinstance(deed, dict) or not deed:
        return True
    if deed.get("_empty"):
        return True
    # Prefer nested data/information when present (full HTTP envelope).
    for key in ("data", "Data", "information", "Information", "result", "Result"):
        val = deed.get(key)
        if isinstance(val, dict) and val:
            deed = val
            break
        if isinstance(val, list) and val:
            return False
    usable = 0
    for k, v in deed.items():
        if str(k).lower() in _ENVELOPE_KEYS:
            continue
        if v in (None, "", [], {}):
            continue
        usable += 1
    return usable == 0


def _unwrap(payload: Any) -> Any:
    """Peel common ASP.NET / wrapper envelopes so callers see the deed
    object (or list) directly."""
    if not isinstance(payload, dict):
        return payload
    for key in ("data", "Data", "result", "Result", "response", "Response",
                "deedInfo", "DeedInfo", "deedDetails", "DeedDetails"):
        if key in payload and payload[key] not in (None, "", [], {}):
            return _unwrap(payload[key])
    return payload


def _pick(obj: Any, *names: str, default=None):
    """Case-insensitive / underscore-insensitive lookup across a few
    common response shapes."""
    if obj is None:
        return default
    if isinstance(obj, list):
        for item in obj:
            v = _pick(item, *names, default=None)
            if v not in (None, ""):
                return v
        return default
    if not isinstance(obj, dict):
        return default
    norm = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in obj.items()}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in norm and norm[key] not in (None, ""):
            return norm[key]
    return default


class SarvamClient:
    def __init__(
        self,
        base_url: str | None = None,
        login_id: str | None = None,
        password: str | None = None,
        user_type: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or os.environ.get("SARVAM_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise ValueError(
                "SARVAM_BASE_URL is required (the BaseURL from the Postman collection)")
        self.login_id = login_id or os.environ.get("SARVAM_LOGIN_ID", DEFAULT_LOGIN_ID)
        self.password = password or os.environ.get("SARVAM_PASSWORD", DEFAULT_PASSWORD)
        self.user_type = user_type or os.environ.get("SARVAM_USER_TYPE", DEFAULT_USER_TYPE)
        self.timeout = timeout
        self.token: str | None = None
        self.verify_ssl = os.environ.get("SARVAM_VERIFY_SSL", "1").lower() not in (
            "0", "false", "no")
        if not self.verify_ssl:
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json",
                                     "Accept": "application/json"})
        # Keep last raw JSON for --probe / debugging.
        self.last_raw: Any = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _auth_headers(self) -> dict:
        if not self.token:
            raise RuntimeError("Not authenticated — call authenticate() first")
        return {"Authorization": f"Bearer {self.token}"}

    def authenticate(self) -> str:
        r = self.session.post(
            self._url("/api/User/GetUserAuthencate"),
            json={
                "useR_LOGIN_ID": self.login_id,
                "useR_PWD": self.password,
                "useR_TYPE": self.user_type,
            },
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        r.raise_for_status()
        payload = r.json()
        token = _pick(payload, "token", "Token", "accessToken", "access_token",
                      "jwt", "jwtToken", "authToken")
        if not token and isinstance(payload, dict):
            # IGR Odisha ERP: { "information": { "token": "..." }, ... }
            info = payload.get("information") or payload.get("Information")
            if isinstance(info, dict):
                token = _pick(info, "token", "Token", "accessToken", "access_token")
        if not token and isinstance(payload, str):
            token = payload
        if not token and isinstance(payload, dict):
            # some APIs nest the token one level deeper after unwrap
            inner = _unwrap(payload)
            token = _pick(inner, "token", "Token", "accessToken", "access_token",
                          "jwt", "jwtToken", "authToken")
            if not token and isinstance(inner, str):
                token = inner
            if not token and isinstance(inner, dict):
                info = inner.get("information") or inner.get("Information")
                if isinstance(info, dict):
                    token = _pick(info, "token", "Token")
        if not token:
            raise RuntimeError(
                f"Authenticate succeeded but no token found in response: {payload!r}")
        self.token = str(token).strip()
        return self.token

    def get_reg_nos_by_date(self, from_date: str, to_date: str) -> list[dict]:
        """POST GetDeedRegNoDetail. Dates as in Postman: '15-Jan-2002'."""
        r = self.session.post(
            self._url("/api/Deed/GetDeedRegNoDetail"),
            headers=self._auth_headers(),
            json={"fromDate": from_date, "toDate": to_date},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        r.raise_for_status()
        data = _unwrap(r.json())
        if isinstance(data, dict):
            for key in ("list", "List", "items", "Items", "records", "Records",
                        "regNoDetails", "RegNoDetails"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            return []
        out = []
        for row in data:
            if not isinstance(row, dict):
                continue
            reg = _pick(row, "registrationNo", "registration_no", "regNo",
                        "reg_no", "RegistrationNo", "deedNo", "deed_no")
            if not reg:
                continue
            book = _pick(row, "bookNo", "book_no", "book", "BookNo", "bookNumber")
            out.append({
                "registration_no": str(reg).strip(),
                "book_no": _as_book_int(book),
                "raw": row,
            })
        return out

    def get_deed_info(self, registration_no: str) -> dict:
        """POST GetDeedInfoByRegNo — official metadata for one registration.

        Live IGR envelope puts the deed object in `data` (dict), with
        `information` usually null. Empty/removed deeds return data=null.
        """
        reg = str(registration_no).strip()
        bodies = [
            {"registrationNo": reg},
            {"RegistrationNo": reg},
        ]
        last_raw = None
        for body in bodies:
            r = self.session.post(
                self._url("/api/Deed/GetDeedInfoByRegNo"),
                headers=self._auth_headers(),
                json=body,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            if r.status_code >= 400:
                # alternate body shapes can 400 — try the next
                continue
            raw = r.json()
            last_raw = raw
            self.last_raw = raw
            candidate = None
            if isinstance(raw, dict):
                # Prefer `data` first — that's where IGR puts the deed object.
                for key in ("data", "Data", "information", "Information",
                            "result", "Result"):
                    val = raw.get(key)
                    if isinstance(val, dict) and val:
                        candidate = val
                        break
                    if isinstance(val, list) and val:
                        candidate = val[0] if isinstance(val[0], dict) else None
                        if candidate:
                            break
            if isinstance(candidate, dict) and not is_empty_deed_payload(candidate):
                out = dict(candidate)
                out.setdefault("registration_no", reg)
                out.setdefault(
                    "registrationNo",
                    out.get("registrationNo") or reg)
                out["_empty"] = False
                return out
        if isinstance(last_raw, dict):
            out = dict(last_raw)
        else:
            out = {"_raw": last_raw}
        out["registration_no"] = reg
        out["_empty"] = True
        return out

    def get_deed_scan_copy(self, registration_no: str) -> Any:
        """POST GetDeedScanCopy — scanned deed payload (often base64 / URL)."""
        r = self.session.post(
            self._url("/api/Deed/GetDeedScanCopy"),
            headers=self._auth_headers(),
            json={"registrationNo": str(registration_no).strip()},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        r.raise_for_status()
        self.last_raw = r.json()
        return self.last_raw


def _as_book_int(value) -> int | None:
    if value is None or value == "":
        return None
    s = str(value).strip()
    m = re.search(r"\d+", s)
    if not m:
        # Roman numerals occasionally appear as Book I / III / IV
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
        return roman.get(s.lower().replace("book", "").strip())
    try:
        return int(m.group(0))
    except ValueError:
        return None


def book_no_from_api_deed(deed: dict) -> int | None:
    return _as_book_int(_pick(
        deed, "deedBookNo", "bookNo", "book_no", "book", "BookNo", "bookNumber",
        "bookType", "book_type", "Book"))
