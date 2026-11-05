# Prototype retrieval and limitations report

Status: prototype, using fictional fixture documents. These numbers are **not** a measure of real-world accuracy. Rerun the evaluation with the business's documents and labeled questions before customer rollout.

## Setup

- **Documents** (`fixtures/documents/`):
  - `product.md`: Markdown with headings.
  - `policy.txt`: plain text.
  - `faq.docx`: headings and a table.
  - `security.pdf`: searchable, two pages.
  - `onboarding-scan.pdf`: image-only; text comes from OCR.
  - `legacy-hours.html`: an outdated 2023 contact page whose support hours deliberately conflict with `policy.txt`.

  Regenerate the binary files with `python scripts/make_fixtures.py`.
- **Questions** (`fixtures/questions.json`): 12 labeled cases.
  - 4 direct: one per binary format, including the scan.
  - 3 multi-passage.
  - 1 ambiguous.
  - 2 absent: SAML SSO; an incident SLA that sits next to related but insufficient text.
  - 1 conflicting.
  - 1 follow-up that depends on conversation history.

## How retrieval works

1. **Parse.** Extract text with document order preserved, and record title, version, page (PDF), and heading path (Markdown, HTML, DOCX). Pages with fewer than 40 letters of extractable text go to OCR (Poppler at 300 dpi, then Tesseract). A page is flagged `low_ocr_quality` when OCR still yields fewer than 40 letters or its mean word confidence is below 60. Flagged pages are still indexed, and their warnings are returned at upload.
2. **Chunk.** Paragraphs are packed into chunks of at most 1,300 characters, with about 180 characters of overlap. Chunks never cross a page or section, and each keeps its character offsets. The title and section are prepended to the text that is embedded.
3. **Search.** Exact cosine search in pgvector over active chunks in shared or same-account documents. The account filter is part of the SQL query, so other accounts' chunks are never candidates. It returns 12 candidates.
4. **Rerank.** By default, 75% vector similarity and 25% BM25 over the candidate set. With `.[local]` installed, a cross-encoder can be used instead. The top 6 go to the model.
5. **Answer or abstain.**
   - If the best similarity is below `MIN_RETRIEVAL_SCORE` (0.24), abstain without calling the model.
   - Otherwise the model must return `answered` with citation numbers, `insufficient`, or `conflicting`.
   - Malformed output, out-of-range citations, and provider errors all abstain.
   - Cited chunks are rechecked against the database for active status and account access before they are returned.
6. **Follow-ups.** When a session has history, the model rewrites the message as a standalone query. If that fails, the service falls back to a heuristic. Evidence is retrieved fresh, with the same account filter, on every turn.

## Results

### Offline pipeline run (deterministic test providers)

[`fixture-eval-offline.md`](fixture-eval-offline.md) was produced with the hashing embedder and keyword-matching generator from `tests/conftest.py`. It shows that the whole pipeline (parsing every format including OCR, retrieval, scoring, report storage, and cleanup) runs end to end and that the scorer catches failures. Summary: **8/12 passed**.

- Retrieval hit 10/10: an expected document was retrieved every time.
- The four failures came from the keyword fake, not from a real model:
  - q06: cited the wrong passage on word overlap.
  - q08: "seats" doesn't match "agents" without semantic embeddings.
  - q09, q10: the fake can't tell related-but-insufficient text apart from an answer.
- The direct answers from all five formats passed, as did the conflicting case (abstained as `conflicting`) and the follow-up.

### Real-model run

Not run in this environment, because no provider credentials are configured. To produce it:

```
cp .env.example .env   # add OPENAI_API_KEY (or Anthropic settings)
rag-evaluate --documents fixtures/documents --questions fixtures/questions.json --report reports/eval --judge
```

Paste the summary table here and review every failure by hand. The deterministic checks (key facts, cited documents) are strict on wording, so review them next to the `--judge` verdicts.

## Observations

- `policy.txt` comes out as a single chunk: it is short, and its paragraphs all fit within the size limit. The refund, hours, and retention facts therefore always arrive together. That doesn't hurt correctness, but it inflates the context and can make conflict detection noisier. If real documents are similar, consider a smaller `CHUNK_MAX_CHARS` (for example 600).
- The conflicting case depends on the outdated page still being indexed. In production, the better fix is content governance: replace or delete superseded pages. Abstention is the safety net.
- `MIN_RETRIEVAL_SCORE` is an uncalibrated starting value. Set it from the score distribution of answerable versus absent questions in the real labeled set.

## Limitations

- **Claim support is not proven automatically.** Citations are checked for existence, range, and access, but whether each sentence is entailed by its cited passage is only assessed by the optional `--judge` evaluation, not on live traffic.
- **OCR quality varies.** Low-confidence pages are flagged, not rejected, so staff should review upload warnings. Handwriting, multi-column layouts, and tables in scans are likely to degrade. A Poppler build that cannot decode JPEG images will render pages blank; this was seen with an old conda `pdftoppm` 0.81. Use a current Poppler.
- **Ingestion is synchronous.** Long scanned PDFs can take minutes, so put a long-timeout proxy in front of the API or move ingestion to a job queue before bulk loads.
- **Search is exact.** That is appropriate for about 10,000 documents at moderate traffic. Benchmark latency before adding an HNSW index, and measure recall with the account filter applied, because filtered approximate search can drop relevant results.
- **Ambiguous questions may abstain rather than ask a clarifying question.** That is the intended conservative behavior, but customers may prefer a clarifying question.
- **Retention and logging defaults need business sign-off.** Message text is stored in the database for session history (90-day default) and is never written to application logs.
