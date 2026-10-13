# SMS rule indicators

The first explainability slice is implemented in `src/scamshield/rules/sms.py` as `analyze_message(message)`. It returns named signals and plain-language explanations; it does **not** return a numeric score or declare that a message is fraudulent.

## Signals

| Signal code | Examples of cues | What the explanation says |
|---|---|---|
| `credential_request` | A request to share/enter an OTP, PIN, CVV, or password; common Hindi terms and Hindi-script request verbs are included. | A sensitive code or credential appears to be requested. |
| `urgency_or_threat` | “Act now,” a short deadline, account-blocking language, or selected Hindi urgency/account phrases. | Time pressure or a threat could be rushing a decision. |
| `upfront_fee_or_deposit` | Registration, processing, security, or advance fee/deposit language, including a few Hindi/Hinglish phrases. | Verify the payment request independently before paying. |
| `prize_or_lottery_lure` | Unexpected prize, lottery, reward, and selected Hindi/Hinglish win phrases. | Unexpected winnings can be used to solicit money or personal information. |
| `suspicious_link_characteristics` | Plain-string checks for HTTP, known URL shorteners, user-info syntax, numeric IP hosts, or punycode host labels. | The specific characteristic is described; the link is not opened or reputation-checked. |

The patterns are deliberately small and reviewable. They may miss scams and may flag legitimate messages that use the same wording. Hindi/Hinglish coverage is partial; these rules are not a language detector. Absence of signals does not establish safety.

## Example

```python
from scamshield.rules import analyze_message

result = analyze_message("Share your OTP immediately: http://bit.ly/example")
for signal in result["signals"]:
    print(signal["title"], "—", signal["explanation"])
```

The function does not echo matched text, so it will not repeat a one-time code from the message. URL analysis is lexical only and makes no network request.

Run the tests from the project root with:

```powershell
python -m pytest -q -p no:cacheprovider
```
