"""Transparent, lexical SMS risk indicators.

These rules surface cues for review; they do not determine whether a message is
fraudulent. URL checks are string-only and never make a network request.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Signal:
    code: str
    title: str
    explanation: str
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["details"] = list(self.details)
        return result


_CREDENTIAL_TERMS = r"(?:\b(?:otp|one[- ]time\s+(?:password|code)|upi\s*pin|pin|cvv|password|passcode)\b|ओटीपी|पिन)"
_REQUEST_VERBS = r"(?:\b(?:share|send|tell|provide|enter|confirm|verify|update|reply with|submit)\b|भेजें|बताएं|दर्ज करें|शेयर करें|साझा करें)"
_CREDENTIAL_REQUEST = re.compile(
    rf"(?:{_REQUEST_VERBS}).{{0,48}}(?:{_CREDENTIAL_TERMS})|"
    rf"(?:{_CREDENTIAL_TERMS}).{{0,48}}(?:{_REQUEST_VERBS})",
    re.IGNORECASE,
)

_URGENCY_PATTERNS = (
    re.compile(r"\b(?:act now|immediately|urgent(?:ly)?|right away|within\s+\d+\s+(?:minutes?|hours?)|last chance|limited time)\b", re.IGNORECASE),
    re.compile(r"\b(?:account|card|service)\s+(?:will be\s+)?(?:blocked|suspended|closed|deactivated)\b", re.IGNORECASE),
    re.compile(r"\b(?:kyc|verification)\s+(?:expires?|pending|required)\b", re.IGNORECASE),
    re.compile(r"(?:तुरंत|अभी|शीघ्र|जल्द|खाता\s*(?:बंद|ब्लॉक)|केवाईसी\s*(?:समाप्त|पूरा))", re.IGNORECASE),
)

_ADVANCE_FEE_PATTERNS = (
    re.compile(r"\b(?:registration|processing|security|advance|joining|refundable)\s+(?:fee|deposit|charge)\b", re.IGNORECASE),
    re.compile(r"\b(?:pay|transfer|send)\b.{0,48}\b(?:fee|deposit|charge)\b", re.IGNORECASE),
    re.compile(r"\b(?:fee|deposit|charge)\b.{0,48}\b(?:pay|transfer|send)\b", re.IGNORECASE),
    re.compile(r"(?:रजिस्ट्रेशन\s*फीस|प्रोसेसिंग\s*फीस|सुरक्षा\s*जमा|शुल्क\s*जमा|शुल्क\s*भुगतान)", re.IGNORECASE),
)

_PRIZE_PATTERNS = (
    re.compile(r"\b(?:you have won|you won|winner|lottery|prize|claim your reward|claim prize)\b", re.IGNORECASE),
    re.compile(r"\b(?:inaam|jeeta|jeet gaye|mubarak|congratulations)\b", re.IGNORECASE),
    re.compile(r"(?:इनाम|जीत गए|जीते हैं|लॉटरी|पुरस्कार|बधाई)", re.IGNORECASE),
)

_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}"
_SHORTENER_HOSTS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "cutt.ly",
    "rb.gy",
    "is.gd",
    "shorturl.at",
}


def _contains(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _has_credential_request(text: str) -> bool:
    for match in _CREDENTIAL_REQUEST.finditer(text):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if re.search(r"(?:\bnever|\bdon't|\bdont|\bdo\s+not)\s*$", prefix, re.IGNORECASE):
            continue
        return True
    return False


def _extract_urls(text: str) -> list[str]:
    urls = []
    for match in _URL_PATTERN.findall(text):
        candidate = match.rstrip(_URL_TRAILING_PUNCTUATION)
        if candidate:
            urls.append(candidate)
    return list(dict.fromkeys(urls))


def _url_reasons(url: str) -> list[str]:
    normalized = url if re.match(r"(?i)^https?://", url) else f"http://{url}"
    parsed = urlsplit(normalized)
    host = (parsed.hostname or "").rstrip(".").lower()
    reasons: list[str] = []

    if parsed.scheme.lower() == "http":
        reasons.append("The link uses unencrypted HTTP rather than HTTPS.")
    if host in _SHORTENER_HOSTS:
        reasons.append("The link uses a URL-shortening service, which hides its final destination.")
    if "@" in parsed.netloc:
        reasons.append("The link contains user-info syntax that can disguise the actual host.")
    if any(label.startswith("xn--") for label in host.split(".")):
        reasons.append("The host contains an internationalized/punycode label that may be hard to visually distinguish.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        reasons.append("The link uses a numeric IP address instead of a recognizable domain name.")
    return reasons


def analyze_message(message: str) -> dict[str, object]:
    """Return explainable SMS indicators without a numeric risk score.

    The message is not sent anywhere. The output intentionally avoids echoing
    matched text, which could include private information or one-time codes.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    signals: list[Signal] = []
    if _has_credential_request(message):
        signals.append(
            Signal(
                code="credential_request",
                title="Request for a sensitive code or credential",
                explanation="The message appears to ask the recipient to share or enter a one-time code, PIN, password, or similar credential.",
            )
        )
    if _contains(_URGENCY_PATTERNS, message):
        signals.append(
            Signal(
                code="urgency_or_threat",
                title="Urgency or account-pressure language",
                explanation="The message uses time pressure or threatens account/service restriction, which can be used to rush a decision.",
            )
        )
    if _contains(_ADVANCE_FEE_PATTERNS, message):
        signals.append(
            Signal(
                code="upfront_fee_or_deposit",
                title="Up-front fee or deposit language",
                explanation="The message mentions paying or transferring a fee or deposit; verify the request independently before paying.",
            )
        )
    if _contains(_PRIZE_PATTERNS, message):
        signals.append(
            Signal(
                code="prize_or_lottery_lure",
                title="Prize or lottery claim language",
                explanation="The message claims a prize, lottery win, or reward; unexpected winnings can be used to solicit money or personal information.",
            )
        )

    url_details = tuple(
        dict.fromkeys(reason for url in _extract_urls(message) for reason in _url_reasons(url))
    )
    if url_details:
        signals.append(
            Signal(
                code="suspicious_link_characteristics",
                title="Link characteristic needs checking",
                explanation="The message contains a link with one or more characteristics worth checking. This check does not visit the link or confirm that it is malicious.",
                details=url_details,
            )
        )

    return {
        "signals": [signal.to_dict() for signal in signals],
        "signal_count": len(signals),
        "disclaimer": "These rule-based indicators are clues for review, not proof of fraud. No URL is opened: this checker does not visit links or confirm them as malicious.",
    }
