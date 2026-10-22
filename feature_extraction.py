from __future__ import annotations

import csv
import difflib
import email
import os
import re
import unicodedata
from dataclasses import dataclass
from email import policy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from docx import Document


FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "aol.com", "proton.me", "protonmail.com",
}
URGENCY_PATTERNS = [
    r"\burgent\b", r"\bimmediately\b", r"\basap\b", r"\baction required\b",
    r"\bfinal warning\b", r"\bwithin \d+ hours?\b", r"\bnow\b", r"\btoday\b",
]
SENSITIVE_PATTERNS = [
    r"\bpassword\b", r"\bcredentials\b", r"\bcredit card\b",
    r"\bbank(ing)? information\b", r"\bssn\b", r"\blogin details\b",
    r"\bverification code\b", r"\baccount number\b",
]
ACTION_PATTERNS = [
    r"\bclick here\b", r"\bverify\b", r"\bconfirm\b", r"\breset\b",
    r"\blog in\b", r"\bsign in\b", r"\bupdate\b", r"\bunlock\b", r"\benroll\b",
]
THREAT_PATTERNS = [
    r"\bsuspended\b", r"\blocked\b", r"\bdeactivated\b", r"\bterminated\b",
    r"\bdisabled\b", r"\block(ed|out)\b", r"\bunauthorized\b", r"\bcompromised\b",
]
GENERIC_GREETING_PATTERNS = [
    r"\bdear customer\b", r"\bdear user\b", r"\bdear valued customer\b",
    r"\battention\b",
]
EMOTIONAL_PATTERNS = [
    r"\bprotect your information\b", r"\bfor your safety\b", r"\bsecurity alert\b",
    r"\brisk\b", r"\bthreat\b", r"\bfraud\b", r"\bunauthorized\b",
]
FINANCIAL_PATTERNS = [
    r"\binvoice\b", r"\bpayment\b", r"\bpayroll\b", r"\breimbursement\b",
    r"\brefund\b", r"\bwire transfer\b", r"\bbank transfer\b", r"\boverdue\b",
    r"\btransaction\b",
]
IMPERSONATION_PATTERNS = [
    r"\bit (?:support|security|desk|administrator)\b", r"\bhelp\s?desk\b",
    r"\bsecurity team\b", r"\bhr (?:office|team|department)\b",
    r"\bfinance (?:office|team|department)\b", r"\bmail(?:box)? administrator\b",
    r"\bsystem administrator\b",
]
CREDENTIAL_REQUEST_PATTERNS = [
    r"\bre-enter\b.{0,35}\bcredentials?\b", r"\benter\b.{0,35}\bcredentials?\b",
    r"\bprovide\b.{0,35}\b(?:password|credentials?|verification code)\b",
    r"\bsign in\b", r"\blog in\b", r"\bidentity check\b",
    r"\bauthentication check\b", r"\bvalidate\b.{0,25}\b(?:account|identity|mailbox)\b",
]
POLITE_ACTION_PATTERNS = [
    r"\bplease (?:review|open|confirm|verify|sign in|log in)\b",
    r"\bwhen convenient\b.{0,50}\b(?:sign in|log in|open|review)\b",
    r"\bquick (?:sign in|login|review|verification|identity check)\b",
    r"\bto (?:view|preview|access|open)\b.{0,40}\b(?:sign in|log in)\b",
]
REASSURANCE_PATTERNS = [
    r"\bno (?:immediate |user )?action is (?:needed|required)\b",
    r"\bthrough the (?:normal|official) (?:course |institutional )?portal\b",
    r"\bwill contact you if\b", r"\bis being handled by\b",
]
# Cue-layer action patterns. Authentication verbs (sign in, log in) belong to
# the credential and polite cues, so a single sign-in request raises one cue
# instead of two for the same words. Feature extraction keeps ACTION_PATTERNS.
CUE_ACTION_PATTERNS = [
    r"\bclick here\b", r"\bverify\b", r"\bconfirm\b", r"\breset\b",
    r"\bupdate\b", r"\bunlock\b", r"\benroll\b",
]

URL_REGEX = re.compile(r"https?://[^\s'\">)]+|www\.[^\s'\">)]+", re.I)
IP_URL_REGEX = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|\b)", re.I
)
# Bait stems taken from the hyphenated compounds in the supplied rule set
# (secure-, verify-, account-, billing-, auth-update, transaction-review) and
# their inflections. Host labels are matched as whole tokens, so the rule keys on
# an account-security or billing context rather than on enumerated host strings.
# Brand misspellings such as paypa1 and micr0soft are not listed here; the
# lookalike check covers them, and repeating them would double-count one link.
SUSPICIOUS_DOMAIN_STEMS = {
    "secure", "security", "verify", "verified", "verification",
    "account", "accounts", "billing", "auth", "authentication",
    "update", "transaction", "review",
}


def _host_tokens(host: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", host.lower()) if token]
SUSPICIOUS_TLDS = {"zip", "mov", "top", "xyz", "click", "work", "support", "info", "buzz"}
KNOWN_BRANDS = {
    "microsoft", "paypal", "google", "apple", "docusign", "dropbox", "outlook",
    "university",
}
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "rb.gy"}

# The supplied lab's own pattern lists and URL rules, kept verbatim. The control
# arms and the supplied-detector reconstruction score with these, so a before/
# after comparison never mixes a rule edit into the feature delta. The duplicate
# threat pattern and the greeting match on any "hello" are supplied behaviour.
SUPPLIED_THREAT_PATTERNS = [
    r"\bsuspended\b", r"\blocked\b", r"\bdeactivated\b", r"\bterminated\b",
    r"\blocked\b", r"\bdisabled\b", r"\block(ed|out)\b", r"\bunauthorized\b",
    r"\bcompromised\b",
]
SUPPLIED_GENERIC_GREETING_PATTERNS = [
    r"\bdear customer\b", r"\bdear user\b", r"\bdear valued customer\b",
    r"\bhello\b", r"\battention\b",
]
SUPPLIED_SUSPICIOUS_DOMAIN_REGEX = re.compile(
    r"(?:paypa1|micr0soft|secure-|verify-|account-|billing-|auth-update|"
    r"transaction-review)",
    re.I,
)
SUPPLIED_SHORTENER_DOMAINS = ("bit.ly", "tinyurl.com", "t.co", "rb.gy")


@dataclass
class ParsedEmail:
    subject: str = ""
    from_addr: str = ""
    to_addr: str = ""
    reply_to: str = ""
    return_path: str = ""


def sanitize_text(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode(
        "utf-8", "ignore"
    ).decode("utf-8", "ignore")


def _read_html_from_string(raw: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(" ", strip=True)
    urls = [a["href"].strip() for a in soup.find_all("a", href=True)]
    urls.extend(URL_REGEX.findall(raw))
    return text, sorted(set(urls))


def _read_html(path: str) -> tuple[str, list[str]]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return _read_html_from_string(raw)


def _read_docx(path: str) -> tuple[str, list[str]]:
    text = "\n".join(p.text for p in Document(path).paragraphs)
    return text, extract_urls_from_text(text)


def _read_eml(path: str) -> tuple[str, list[str], ParsedEmail]:
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    body_parts: list[str] = []
    urls: list[str] = []
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            try:
                body_parts.append(part.get_content())
            except Exception:
                pass
        elif content_type == "text/html":
            try:
                text, found = _read_html_from_string(part.get_content())
                body_parts.append(text)
                urls.extend(found)
            except Exception:
                pass

    if not body_parts:
        try:
            payload = msg.get_body(preferencelist=("plain", "html"))
            if payload:
                content = payload.get_content()
                if payload.get_content_type() == "text/html":
                    text, found = _read_html_from_string(content)
                    body_parts.append(text)
                    urls.extend(found)
                else:
                    body_parts.append(content)
        except Exception:
            pass

    parsed = ParsedEmail(
        subject=str(msg.get("Subject", "") or ""),
        from_addr=str(msg.get("From", "") or ""),
        to_addr=str(msg.get("To", "") or ""),
        reply_to=str(msg.get("Reply-To", "") or ""),
        return_path=str(msg.get("Return-Path", "") or ""),
    )
    full_text = "\n".join(filter(None, [parsed.subject, *body_parts]))
    urls.extend(extract_urls_from_text(full_text))
    return full_text, sorted(set(urls)), parsed


def extract_content(path: str) -> dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext == ".html":
        text, urls = _read_html(path)
        headers = ParsedEmail()
    elif ext == ".docx":
        text, urls = _read_docx(path)
        headers = ParsedEmail()
    elif ext == ".eml":
        text, urls, headers = _read_eml(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {"text": sanitize_text(text), "urls": urls, "headers": headers}


def _count_matches(text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, re.I))


def extract_urls_from_text(text: str) -> list[str]:
    """Extract and de-duplicate URLs from plain text CSV records."""
    return sorted({url.rstrip(".,;:!?]") for url in URL_REGEX.findall(text or "")})


def extract_domain(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"@([A-Za-z0-9.-]+)", value)
    return match.group(1).lower() if match else ""


def get_header_anomalies(headers: ParsedEmail) -> list[str]:
    headers = headers or ParsedEmail()
    anomalies: list[str] = []
    from_domain = extract_domain(headers.from_addr)
    reply_domain = extract_domain(headers.reply_to)
    return_domain = extract_domain(headers.return_path)

    if from_domain and from_domain in FREE_EMAIL_DOMAINS:
        anomalies.append("Sender uses a free email provider despite claiming an organizational role.")
    if from_domain and reply_domain and from_domain != reply_domain:
        anomalies.append("Reply-To domain differs from the visible From domain.")
    if from_domain and return_domain and from_domain != return_domain:
        anomalies.append("Return-Path domain differs from the visible From domain.")
    return anomalies


def _url_host(url: str) -> str:
    parsed = urlparse(url if re.match(r"^https?://", url, re.I) else f"http://{url}")
    return (parsed.hostname or "").lower().rstrip(".")


def _looks_like_brand_spoof(host: str) -> bool:
    """Detect close brand misspellings while excluding exact brand tokens."""
    translation = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t"})
    for token in re.split(r"[^a-z0-9]+", host.lower()):
        if not token:
            continue
        normalized = token.translate(translation)
        for brand in KNOWN_BRANDS:
            if token != brand and (
                normalized == brand
                or (
                    len(token) >= 5
                    and difflib.SequenceMatcher(None, normalized, brand).ratio() >= 0.86
                )
            ):
                return True
    return False


def _url_indicators(url: str) -> dict[str, bool]:
    host = _url_host(url)
    parsed = urlparse(url if re.match(r"^https?://", url, re.I) else f"http://{url}")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    labels = [label for label in host.split(".") if label]
    return {
        "ip": bool(IP_URL_REGEX.search(url)),
        "suspicious_domain": bool(
            set(_host_tokens(host)) & SUSPICIOUS_DOMAIN_STEMS
        ),
        "shortener": host in SHORTENER_DOMAINS,
        "suspicious_tld": tld in SUSPICIOUS_TLDS,
        "brand_similarity": _looks_like_brand_spoof(host),
        "complex_url": (
            "@" in parsed.netloc
            or host.startswith("xn--")
            or len(host) > 45
            or len(labels) > 4
            or host.count("-") >= 3
        ),
    }


def compute_supplied_features(
    text: str,
    urls: list[str] | None = None,
    headers: ParsedEmail | None = None,
) -> dict[str, float]:
    """The supplied lab's 15 features, implemented exactly as the lab defined them."""
    urls = urls or []
    headers = headers or ParsedEmail()
    lowered = text.lower()
    words = re.findall(r"\b\w+\b", lowered)

    ip_urls = 0
    suspicious_domains = 0
    shorteners = 0
    for url in urls:
        parsed = urlparse(url if url.startswith("http") else f"http://{url}")
        host = (parsed.netloc or parsed.path).lower()
        if IP_URL_REGEX.search(url):
            ip_urls += 1
        if SUPPLIED_SUSPICIOUS_DOMAIN_REGEX.search(host):
            suspicious_domains += 1
        if any(short in host for short in SUPPLIED_SHORTENER_DOMAINS):
            shorteners += 1

    return {
        "char_count": float(len(text)),
        "word_count": float(len(words)),
        "url_count": float(len(urls)),
        "ip_url_count": float(ip_urls),
        "suspicious_domain_count": float(suspicious_domains),
        "url_shortener_count": float(shorteners),
        "urgency_count": float(_count_matches(lowered, URGENCY_PATTERNS)),
        "sensitive_count": float(_count_matches(lowered, SENSITIVE_PATTERNS)),
        "action_count": float(_count_matches(lowered, ACTION_PATTERNS)),
        "threat_count": float(_count_matches(lowered, SUPPLIED_THREAT_PATTERNS)),
        "generic_greeting_count": float(
            _count_matches(lowered, SUPPLIED_GENERIC_GREETING_PATTERNS)
        ),
        "emotional_tone_count": float(_count_matches(lowered, EMOTIONAL_PATTERNS)),
        "exclamation_count": float(text.count("!")),
        "uppercase_ratio": float(
            round(sum(1 for char in text if char.isupper()) / max(len(text), 1), 5)
        ),
        "header_anomaly_count": float(len(get_header_anomalies(headers))),
    }


def compute_handcrafted_features(
    text: str,
    urls: list[str] | None = None,
    headers: ParsedEmail | None = None,
) -> dict[str, float]:
    """Enhanced feature set: the supplied 15 features plus eight new ones.

    URLs are extracted from every text record, so the URL features carry values
    during training as well as in the application.
    """
    effective_urls = sorted(set((urls or []) + extract_urls_from_text(text)))
    features = compute_supplied_features(text, effective_urls, headers)
    indicators = [_url_indicators(url) for url in effective_urls]
    features.update(
        {
            "suspicious_tld_count": float(
                sum(item["suspicious_tld"] for item in indicators)
            ),
            "domain_similarity_count": float(
                sum(item["brand_similarity"] for item in indicators)
            ),
            "complex_url_count": float(sum(item["complex_url"] for item in indicators)),
            "credential_request_count": float(
                _count_matches(text, CREDENTIAL_REQUEST_PATTERNS)
            ),
            "financial_pressure_count": float(
                _count_matches(text, FINANCIAL_PATTERNS)
            ),
            "impersonation_signal_count": float(
                _count_matches(text, IMPERSONATION_PATTERNS)
            ),
            "polite_action_count": float(_count_matches(text, POLITE_ACTION_PATTERNS)),
            "reassurance_count": float(_count_matches(text, REASSURANCE_PATTERNS)),
        }
    )
    return features


def _imperative_action_present(text: str) -> bool:
    """True when an action verb appears outside any softened polite phrase.

    A verb inside a polite phrase raises the polite cue instead, so one
    phrasing cannot raise two cues; a separate imperative in the same message
    still raises the action cue.
    """
    masked = text
    for pattern in POLITE_ACTION_PATTERNS:
        masked = re.sub(
            pattern, lambda match: " " * len(match.group(0)), masked, flags=re.I
        )
    return _count_matches(masked, CUE_ACTION_PATTERNS) > 0


def phishing_cues(
    text: str,
    urls: list[str] | None = None,
    headers: ParsedEmail | None = None,
) -> list[str]:
    effective_urls = sorted(set((urls or []) + extract_urls_from_text(text)))
    headers = headers or ParsedEmail()
    cues: list[str] = []

    if _count_matches(text, URGENCY_PATTERNS):
        cues.append("Urgency: the message pressures the recipient to act quickly.")
    if _imperative_action_present(text):
        cues.append(
            "Requested action: the message asks the recipient to click, verify, "
            "confirm, or update."
        )
    if _count_matches(text, SENSITIVE_PATTERNS):
        cues.append(
            "Sensitive information: passwords, credentials, banking data, or "
            "verification codes are mentioned."
        )
    if _count_matches(text, THREAT_PATTERNS):
        cues.append(
            "Threat: access loss, suspension, or another penalty is used as pressure."
        )
    if _count_matches(text, GENERIC_GREETING_PATTERNS):
        cues.append(
            "Generic greeting: the message does not identify the recipient personally."
        )
    if _count_matches(text, EMOTIONAL_PATTERNS):
        cues.append(
            "Emotional manipulation: security or fraud language is used to create fear."
        )
    if _count_matches(text, FINANCIAL_PATTERNS):
        cues.append(
            "Financial pressure: the message invokes payroll, invoices, payments, "
            "refunds, or reimbursements."
        )
    if _count_matches(text, IMPERSONATION_PATTERNS):
        cues.append(
            "Authority impersonation: the sender presents itself as IT, security, "
            "HR, or finance."
        )
    if _count_matches(text, CREDENTIAL_REQUEST_PATTERNS):
        cues.append(
            "Credential request: the recipient is asked to sign in, re-enter "
            "credentials, or complete an identity check."
        )
    if _count_matches(text, POLITE_ACTION_PATTERNS):
        cues.append(
            "Polite social engineering: a low-pressure request still directs the "
            "recipient to open, review, or authenticate."
        )

    indicators = [_url_indicators(url) for url in effective_urls]
    if any(item["ip"] for item in indicators):
        cues.append(
            "IP-based destination: the link uses a numeric address instead of a "
            "recognizable domain."
        )
    if any(item["suspicious_domain"] for item in indicators):
        cues.append(
            "Suspicious domain wording: the link contains security, verification, "
            "account, or billing bait."
        )
    if any(item["suspicious_tld"] for item in indicators):
        cues.append(
            "High-risk top-level domain: a link uses a TLD frequently abused for "
            "deceptive downloads or landing pages."
        )
    if any(item["brand_similarity"] for item in indicators):
        cues.append(
            "Possible brand lookalike: a domain token closely resembles a known "
            "service or institution."
        )
    if any(item["complex_url"] for item in indicators):
        cues.append(
            "Obfuscated URL structure: a link uses excessive subdomains, hyphens, "
            "length, or user-info syntax."
        )
    cues.extend(get_header_anomalies(headers))
    return list(dict.fromkeys(cues))


FEATURE_LABELS = {
    "ip_url_count": "IP-based destination",
    "suspicious_domain_count": "Suspicious domain wording",
    "url_shortener_count": "Shortened destination",
    "suspicious_tld_count": "High-risk top-level domain",
    "domain_similarity_count": "Brand-lookalike domain",
    "complex_url_count": "Obfuscated URL structure",
    "credential_request_count": "Credential or identity request",
    "financial_pressure_count": "Financial subject pressure",
    "impersonation_signal_count": "Authority impersonation language",
    "polite_action_count": "Polite social-engineering request",
    "urgency_count": "Urgency language",
    "threat_count": "Threat or access-loss language",
    "header_anomaly_count": "Email-header mismatch",
    "reassurance_count": "Legitimate reassurance language",
}

POSITIVE_WEIGHTS = {
    "ip_url_count": 0.18,
    "suspicious_domain_count": 0.13,
    "url_shortener_count": 0.09,
    "suspicious_tld_count": 0.11,
    "domain_similarity_count": 0.15,
    "complex_url_count": 0.09,
    "credential_request_count": 0.09,
    "financial_pressure_count": 0.04,
    "impersonation_signal_count": 0.06,
    "polite_action_count": 0.05,
    "urgency_count": 0.04,
    "threat_count": 0.05,
    "header_anomaly_count": 0.08,
}

MODEL_WEIGHT = 0.70
HEURISTIC_WEIGHT = 0.30
# Derived from the false-alarm budgets recorded in model/threshold_calibration.csv:
# the review cut is the 80th percentile and the high cut the 95th percentile of
# the legitimate scores in the selection corpus, scored by the selected arm on
# its training-split fit, which never trains on that corpus. Regenerate that
# file after retraining and re-read the cuts before changing them by hand.
REVIEW_RISK_THRESHOLD = 0.3162
HIGH_RISK_THRESHOLD = 0.4457


def calculate_risk_breakdown(
    model_probability: float,
    feature_map: dict[str, float],
) -> dict:
    """Blend classifier confidence with weighted indicator evidence."""
    contributions = []
    heuristic_score = 0.0
    for feature, weight in POSITIVE_WEIGHTS.items():
        value = float(feature_map.get(feature, 0))
        if value <= 0:
            continue
        contribution = min(0.30, value * weight)
        heuristic_score += contribution
        contributions.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "value": value,
                "effect": round(contribution, 3),
                "direction": "increase",
            }
        )

    reassurance = float(feature_map.get("reassurance_count", 0))
    if reassurance > 0:
        reduction = min(0.16, reassurance * 0.08)
        heuristic_score -= reduction
        contributions.append(
            {
                "feature": "reassurance_count",
                "label": FEATURE_LABELS["reassurance_count"],
                "value": reassurance,
                "effect": round(reduction, 3),
                "direction": "decrease",
            }
        )

    heuristic_score = round(max(0.0, min(1.0, heuristic_score)), 4)
    model_component = round(MODEL_WEIGHT * float(model_probability), 4)
    heuristic_component = round(HEURISTIC_WEIGHT * heuristic_score, 4)
    final_score = round(min(1.0, model_component + heuristic_component), 4)
    contributions.sort(key=lambda item: item["effect"], reverse=True)
    return {
        "model_probability": round(float(model_probability), 4),
        "model_component": model_component,
        "heuristic_score": heuristic_score,
        "heuristic_component": heuristic_component,
        "final_score": final_score,
        "contributions": contributions,
    }


def build_baseline_numeric_feature_frame(series):
    """Replicate supplied training, where URL and header inputs were always empty."""
    return __import__("pandas").DataFrame(
        [
            compute_supplied_features(str(text), urls=[], headers=None)
            for text in series
        ]
    )


def build_urlfix_numeric_feature_frame(series):
    """Original 15 features with URLs read from each record.

    Isolates the extraction repair from the eight new features, which the enhanced
    frame bundles with it.
    """
    return __import__("pandas").DataFrame(
        [
            compute_supplied_features(
                str(text), extract_urls_from_text(str(text)), None
            )
            for text in series
        ]
    )


def build_numeric_feature_frame(series):
    """Build enhanced features, including URLs extracted from each text record."""
    return __import__("pandas").DataFrame(
        [compute_handcrafted_features(str(text), urls=None, headers=None) for text in series]
    )


def log_result(log_path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp", "filename", "prediction", "risk_score",
                "model_probability", "cue_count", "header_anomaly_count",
                "triggered_cues",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
