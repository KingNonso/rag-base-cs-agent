# Evaluation report

Run `2bb7c821-12dc-4b43-b5b4-b429683bf915` · embedding `test:hash-256` · generation `test:scripted` · rerank `lexical` · min score 0.1 · git `29a34fc`

**8/12 questions passed.** retrieval hit: 10/10 · citation support: 7/9 · abstention correct: 9/12 · answer correct: 7/9

| ID | Category | Retrieval | Citations | Abstention | Answer | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | direct | ✓ | ✓ | ✓ | ✓ | pass | cited: product.md |
| q02 | direct | ✓ | ✓ | ✓ | ✓ | pass | cited: faq.docx |
| q03 | direct | ✓ | ✓ | ✓ | ✓ | pass | cited: security.pdf |
| q04 | direct | ✓ | ✓ | ✓ | ✓ | pass | cited: onboarding-scan.pdf |
| q05 | multi_passage | ✓ | ✓ | ✓ | ✓ | pass | cited: product.md |
| q06 | multi_passage | ✓ | ✗ | ✓ | ✗ | **fail** | cited: policy.txt; failed: citation_support, answer_correct |
| q07 | multi_passage | ✓ | ✓ | ✓ | ✓ | pass | cited: policy.txt |
| q08 | ambiguous | ✓ | ✗ | ✗ | ✗ | **fail** | abstained (low_similarity); failed: citation_support, abstention_correct, answer_correct |
| q09 | absent | – | – | ✗ | – | **fail** | cited: product.md, security.pdf; failed: abstention_correct |
| q10 | absent | – | – | ✗ | – | **fail** | cited: policy.txt; failed: abstention_correct |
| q11 | conflicting | ✓ | – | ✓ | – | pass | abstained (conflicting) |
| q12 | followup | ✓ | ✓ | ✓ | ✓ | pass | cited: product.md |

## Answers

- **q01** How many agents does the Basic plan allow?
  - expected: Three support agents.
  - answer: Acme Desk has Basic and Pro plans. Basic allows three support agents. Pro allows twenty support agents.
- **q02** Which plans include the Salesforce integration?
  - expected: Only the Pro plan.
  - answer: The table below lists which plans include each integration. Integration | Plans Slack | Basic and Pro Salesforce | Pro only Integrations are configured from Settings > Integrations.
- **q03** How long are backups kept?
  - expected: 14 days; backups are taken every 24 hours.
  - answer: Backups Backups are taken every 24 hours and kept for 14 days. Restores are performed by the Acme Desk operations team on request.
- **q04** How long is the free trial for new accounts?
  - expected: 21 days, on the Pro plan (scanned onboarding checklist).
  - answer: Onboarding checklist New accounts include a 21-day free trial of the Pro plan. During the trial, up to five agents can be invited from Settings > Team.
- **q05** Can a regular agent export tickets, and what file format does the export use?
  - expected: No, only account administrators can export; exports are CSV.
  - answer: Customers can export tickets as CSV from Reports > Export. Only account administrators can export tickets.
- **q06** How is my account and data protected? Mention sign-in options and encryption.
  - expected: Two-factor authentication (admins can require it) and AES-256 encryption at rest with TLS in transit.
  - answer: Annual subscriptions can be refunded within 30 days of purchase. Monthly subscriptions are not refundable. Support hours are Monday through Friday, 09:00 to 17:00 UTC. Critical incidents can be reported at any time using the emergency form in the help center. Customer data is retained for 30 days after account closure, then deleted.
- **q07** I paid for an annual plan two weeks ago. Can I get a refund, and how long is my data kept after I close the account?
  - expected: Yes, annual plans are refundable within 30 days; data is kept 30 days after closure, then deleted.
  - answer: Annual subscriptions can be refunded within 30 days of purchase. Monthly subscriptions are not refundable. Support hours are Monday through Friday, 09:00 to 17:00 UTC. Critical incidents can be reported at any time using the emergency form in the help center. Customer data is retained for 30 days after account closure, then deleted.
- **q08** How many seats do I get?
  - expected: Depends on the plan: Basic has three agents, Pro has twenty.
  - answer: I couldn't find a reliable answer in the available documents. Please contact support for help.
- **q09** Does Acme Desk support SAML single sign-on?
  - expected: Abstain; not in the documents.
  - answer: Acme Desk has Basic and Pro plans. Basic allows three support agents. Pro allows twenty support agents.
- **q10** What is the guaranteed response time SLA for critical incidents?
  - expected: Abstain; the policy only says critical incidents can be reported at any time.
  - answer: Annual subscriptions can be refunded within 30 days of purchase. Monthly subscriptions are not refundable. Support hours are Monday through Friday, 09:00 to 17:00 UTC. Critical incidents can be reported at any time using the emergency form in the help center. Customer data is retained for 30 days after account closure, then deleted.
- **q11** What are your support hours?
  - expected: Abstain; policy says 09:00–17:00 UTC but the 2023 contact page says 08:00–20:00 UTC.
  - answer: The available documents give conflicting information about this, so I can't answer reliably. Please contact support for help.
- **q12** How many agents does the bigger one allow?
  - expected: Pro allows twenty agents.
  - answer: Acme Desk has Basic and Pro plans. Basic allows three support agents. Pro allows twenty support agents.
