from __future__ import annotations

import pytest

from scamshield.rules.sms import analyze_message


def codes(message: str) -> set[str]:
    result = analyze_message(message)
    return {signal["code"] for signal in result["signals"]}


def test_detects_sensitive_code_request_without_echoing_message() -> None:
    result = analyze_message("Share your OTP 839201 immediately to keep your account active.")

    assert {signal["code"] for signal in result["signals"]} == {
        "credential_request",
        "urgency_or_threat",
    }
    assert "839201" not in str(result)
    assert result["signal_count"] == 2


def test_detects_upfront_fee_and_prize_lure() -> None:
    assert codes("Congratulations, you won a prize. Pay the registration fee to claim it.") == {
        "upfront_fee_or_deposit",
        "prize_or_lottery_lure",
    }


def test_detects_common_hindi_and_hinglish_cues() -> None:
    assert "credential_request" in codes("अपना OTP भेजें अभी")
    assert "upfront_fee_or_deposit" in codes("job ke liye registration fee pay karo")
    assert "prize_or_lottery_lure" in codes("Aap lottery jeet gaye")


def test_reports_shortened_and_unencrypted_link_without_fetching_it() -> None:
    result = analyze_message("Verify now: http://bit.ly/example123.")
    signal = next(s for s in result["signals"] if s["code"] == "suspicious_link_characteristics")

    assert "unencrypted HTTP" in " ".join(signal["details"])
    assert "URL-shortening service" in " ".join(signal["details"])
    assert "does not visit" in result["disclaimer"]


def test_reports_numeric_ip_and_punycode_hosts() -> None:
    assert "suspicious_link_characteristics" in codes("http://192.0.2.1/login")
    assert "suspicious_link_characteristics" in codes("https://xn--example-9d0b.test/login")


def test_benign_message_has_no_signals() -> None:
    result = analyze_message("Your appointment is confirmed for 4 pm tomorrow.")

    assert result["signals"] == []
    assert result["signal_count"] == 0


def test_unrelated_credential_mention_is_not_a_request() -> None:
    assert "credential_request" not in codes("Never share your OTP with anyone.")


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="message must be a string"):
        analyze_message(None)  # type: ignore[arg-type]
