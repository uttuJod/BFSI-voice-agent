# Security Model

## Scope

This repository is a portfolio/research implementation of a BFSI voice-support architecture. It demonstrates deterministic privacy and business controls around an LLM router, but it should not be treated as a production banking security boundary without additional controls and independent review.

## Security principles

### 1. Sensitive account data is verification-gated

Protected tools do not return account data to an unverified voice session. The verification layer exists outside the LLM so a model prediction cannot bypass it.

### 2. Third-party privacy requests are blocked

Requests for another person's account information are treated as privacy-sensitive even when the caller claims permission.

### 3. Restricted credentials are out of scope

The assistant is designed not to request or expose:

- OTP
- CVV
- PIN
- password
- full card number
- another customer's private information

### 4. Dynamic facts come from tools

Balances, account state, payment state, and similar dynamic information must come from the business-tool layer rather than model memory.

### 5. Policy facts come from retrieval

Policy answers are grounded in the policy corpus and carry citations rather than being invented by the routing model.

## Demo verification mechanism

The repository's local/demo verification flow uses a stored salted PBKDF2 hash and a session-scoped challenge for the last four digits of a registered mobile number.

This is useful for demonstrating:

- protected-tool gating
- verification state
- attempt limiting
- pending-request resume
- voice-specific digit handling

It is **not** bank-grade authentication or MFA.

A real deployment should integrate an approved authentication provider and should not use mobile-number last-four digits as the primary production identity factor.

## Production requirements not implemented here

Before a real BFSI deployment, add or validate at minimum:

- approved authentication/MFA
- KYC and consent integration
- encryption in transit and at rest
- secure key/secret management
- least-privilege service identities
- database access controls
- audit logging and retention policy
- PII redaction in logs and traces
- rate limiting and abuse controls
- replay protection
- fraud/risk controls
- secure session/token handling
- dependency and container scanning
- vulnerability management
- incident-response processes
- regional data-residency controls
- regulatory/legal/compliance review
- penetration testing and threat modeling

## Logging caution

Voice/STT logs can contain sensitive text. Verification inputs should be redacted or suppressed in production observability pipelines.

## External services

The voice stack uses external STT/TTS services. Their API credentials must be stored outside source control. Usage quotas, service retention policies, data-processing terms, and regional availability should be reviewed before production use.

## Secrets

Never commit:

```text
.env
API keys
access tokens
service credentials
real customer databases
raw production call recordings
production PII exports
private model credentials
```

Rotate a secret immediately if it is accidentally committed.
