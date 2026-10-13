# Dataset selection and source review

**Status:** Source-file and preliminary grouping diagnostics for UCI SMS, the Hinglish supplementary dataset, and PhiUSIIL URLs completed on 2026-10-12. The English UCI SMS baseline and locked evaluation were run using the documented group-aware split. The Hinglish and URL datasets were not used for model training or performance claims. Manual review of message-template candidates, Hinglish provenance, the UCI archive licence wording, and URL registrable-domain grouping remains required before those sources are used in further modeling.

Re-run the local checks with `python scripts/audit_datasets.py` from the project root. Add `--output reports/data_audit.json` to save the summary; the script reports counts and hashes without printing message or URL values.

## 1. UCI SMS Spam Collection — selected for the English baseline

- Official source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Size: 5,574 messages.
- Labels: `ham` and `spam`.
- Licence shown by UCI: CC BY 4.0; provide attribution in the repository and reports.
- Use: reproducible English spam-vs-ham baseline to compare the classical model and transformer on the same split.
- Important limitation: UCI describes a mixed corpus assembled from several sources, including UK forum spam claims and legitimate messages from a Singapore university SMS corpus. It is not an India-specific fraud dataset. `spam` does not always mean fraud, phishing, or financial scam. Results must be described as SMS spam classification, not as proof of UPI-scam detection. [UCI source description](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Local audit (2026-10-12): official UCI archive saved as `data/raw/uci_sms_spam_collection.zip` (SHA-256 `1587EA43E58E82B14FF1F5425C88E17F8496BFCDB67A583DBFF9EEFAF9963CE3`). Archive contains `SMSSpamCollection` (477,907 bytes) and `readme` (5,868 bytes). The data file has 5,574 tab-separated rows, 4,827 `ham` and 747 `spam`; no malformed or empty rows. Exact-text duplicates beyond first occurrence: 403; normalized duplicates after strip/case-fold: 415; no conflicting labels among normalized duplicates. Do not random-split these rows independently; deduplicate or keep duplicate groups together before splitting. This is an initial duplicate check, not a complete near-duplicate or source-overlap audit.
- Template-key check replaces URLs and standalone digit sequences, then normalizes punctuation and spacing. It found 305 repeated signature groups (434 rows beyond the first), with no groups spanning both labels. Pairwise `difflib.SequenceMatcher` comparison of case-folded raw messages across those groups found 999 pairs, minimum similarity 0.864, median 1.0, and all pairs at or above 0.8. This is strong near-duplicate/template evidence, not a semantic test or automatic deduplication decision. Keep each candidate group together or manually review it before splitting; the audit intentionally does not print message text.
- Licence note: the current UCI record identifies CC BY 4.0 and permits sharing/adaptation with attribution. The bundled legacy `readme` separately describes the corpus as free with copyright and warranty terms and does not itself state CC BY 4.0. The official page and archive text are therefore inconsistent; preserve UCI attribution, retain the archive note, and avoid publishing the raw archive or derived message text until the rights status is clarified. [Current UCI licence statement](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## 2. Hinglish & English Financial Scam/Fraud Text Dataset — supplementary only

- Source: https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset
- Listed size: 3,787 examples; listed licence: CC BY 4.0.
- Its card says that the English portion is derived in part from the UCI SMS Spam Collection, and that it has 35 hand-written Hinglish scam examples plus 28 hand-written Hinglish benign examples.
- Use: inspect as a supplementary language/pattern-coverage source and potential challenge set only after exact-overlap checks.
- Do not merge it naively with UCI data: source overlap can leak near-identical messages across train and test. The 63 Hinglish examples are far too few and too hand-authored to establish reliable real-world Hindi/Hinglish performance.
- Preserve attribution and document that the Hinglish items are hand-written examples, not a representative sample of real incoming messages.
- Local audit (2026-10-12): downloaded `data.json` and `README.md` from immutable Hugging Face commit `743799229923fc80fd7e9f2c764b5e9bd4ab3c9a`. `data.json` is a JSON object with a `rows` list and `text`/`label` fields; 3,787 rows, labels 0=benign (3,040) and 1=scam (747), no malformed objects, non-string texts, or empty texts. SHA-256: `data.json` `4C2EA6D6C9303F6B7E27740CB3BBEBDE57EC3CC10EC7B7F9E565E99B5D483F16`; `README.md` `2B9975D2C5959CEA5DB7CDBDEAC77C1D61B31FE6295C8820D2D59F61C3D9468A`.
- Exact overlap audit against UCI used Unicode NFKC normalization, trim, and case-fold: 3,700 rows (3,471 distinct normalized texts) match UCI; all 3,700 labels agree when mapping UCI `ham` to 0 and `spam` to 1. There are 229 normalized duplicate rows within this dataset. Only 87 rows do not match UCI (47 label 1, 40 label 0); none contain Devanagari code points. A small manually selected Romanized-Hindi cue list matched 37 of those 87, but this is only a heuristic and not language identification. The card describes 63 hand-written Hinglish items, so the remaining unmatched rows' provenance/composition needs clarification. Do not use this dataset as an independent test set or mix it into UCI splits. At most, consider the 87 unmatched rows as a small, manually reviewed challenge set after resolving their provenance and labels.
- Provenance limitation: the card labels the dataset CC BY 4.0 and says its non-UCI Hinglish items mirror patterns reported by Indian banks/cyber cells, but it does not provide row-level sources or citations. The Hugging Face viewer currently reports that it cannot parse the split names; local JSON inspection was used instead. Treat it as curated supplementary material, not representative evidence of real-world performance.
- Current card check (2026-10-12): the page still lists CC BY 4.0 and the same composition/provenance claims; the viewer still fails to parse split names. No independent row-level provenance was located, so the 87 unmatched examples remain unsuitable as an independent performance set without author/source clarification. [Dataset card](https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset)

## 3. PhiUSIIL Phishing URL (Website) — selected candidate for URL-string evaluation

- Official source: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Size: 235,795 examples: 134,850 legitimate and 100,945 phishing, according to UCI.
- Licence shown by UCI: CC BY 4.0; cite the dataset paper and source.
- Label mapping in source: label `1` = legitimate; label `0` = phishing. Normalize this carefully in our own schema (e.g. `is_phishing=1`).
- Important limitation: the dataset includes the URL plus features derived from the webpage/source code. For the paste-a-URL product, start with the raw URL string and features computable from that string alone. Do not train on webpage-derived columns unless the production service also derives them using a documented, safe process.
- Split policy: group by normalized hostname/registrable domain and deduplicate URLs before splitting, to reduce leakage from many similar URLs on the same site.
- Local audit (2026-10-12): official archive saved as `data/raw/phiusiil_phishing_url.zip` (SHA-256 `0A639FD03AEA6308C5B1C10C92AA23C2CE1505447A9137271865CD0BADC9A59A`). It contains one CSV (56 columns including `FILENAME` and `label`) with 235,795 rows. Observed labels: 100,945 `0` and 134,850 `1`, matching UCI's mapping. No missing cells; 425 raw URL values are duplicated (850 rows total), with no duplicate URL assigned conflicting labels. Parsing URL hostnames locally found 220,077 distinct hostnames; 54 hostnames occur with both labels. The supplied `Domain` value equals the parsed URL hostname in 235,759 rows and differs in 36; it has 220,086 distinct values, so it is not a registrable-domain key. Do not use row-random splits. Before freezing splits, deduplicate URLs and use a Public Suffix List-aware registrable-domain grouping method to reduce subdomain/campaign leakage.
- For the first URL-string benchmark, use the `URL` column as input and `label` as target, with label mapping normalized to `is_phishing=1`. Exclude webpage/source-code and other-derived columns unless their computation is reproduced at inference. UCI describes instances as URLs and corresponding webpages and says features are extracted from webpage source code and URLs; the 54-feature table therefore is not directly a URL-only input set.

## 4. Employment Scam Aegean Dataset (EMSCAD) — deferred pending licence/access review

- Research description: https://www.mdpi.com/1999-5903/9/1/6
- Public research descriptions report 17,880 job postings (17,014 legitimate and 866 fraudulent), manually annotated, from 2012–2014.
- Limitation: it is an older, non-India-specific long-form job-ad dataset and is commonly encountered via mirrors. The initial source review did not establish a clear redistribution licence from the original host. Before using it, verify the actual download source, terms, and permitted use; do not redistribute raw records by default.
- Keep job-ad classification as a later, separate component and evaluate it separately from SMS messages.

## UPI / Indian scam evidence gap

At kickoff, no sufficiently documented, representative open dataset of real Indian UPI-scam SMS/chat messages was verified for this project. A public GitHub project called FraudLens reports a curated Indian UPI cybercrime corpus, but the repository page reviewed did not make the independent source provenance and data-use licence clear enough for us to adopt it without further audit. Candidate page: https://github.com/twinkleghangare/FraudLens_AIML. We will not assume its reported results are transferable or that its corpus is licensed for reuse. We will not invent labels or present generated examples as real data. We can later create a small, clearly labelled synthetic challenge set for rule/unit tests and pattern coverage, but its results must never be presented as real-world model performance.

## Data governance checklist (complete before training)

- [ ] Keep the original dataset download and its source URL/date/license note.
- [ ] Record the exact file hash and dataset version when possible.
- [ ] Inspect row count, columns, encoding, missing text, empty text, label counts, and text-length distribution.
- [ ] Verify label mapping and preserve original label alongside normalized label.
- [ ] Remove exact duplicates and investigate near-duplicates across all sources.
- [ ] Track dataset/source provenance at row level where practical.
- [ ] Do not put private messages, secrets, or unlicensed raw records in Git.
- [ ] Create and freeze splits only after source overlap review.
