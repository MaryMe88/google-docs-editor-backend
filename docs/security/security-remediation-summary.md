# Security Remediation Summary

**Period:** 2026-10-09 — 2026-10-12  
**Tracking issue:** #35  
**Next review:** 2027-04-10

## Overview

This document summarizes the residual security remediation roadmap (iterations 0–9) performed between October 9 and October 12, 2026.

## Iterations

| # | Focus | Branch | Outcome |
|---|---|---|---|
| 0 | Baseline snapshot | `security/residual-risk-baseline` | `residual-risk-baseline.md` |
| 1 | Production auth hardening | `security/enforce-production-auth` | API key required in production |
| 2 | Backend request limits | `security/enforce-request-limits` | text length & body size enforced |
| 3 | API abuse controls | `security/api-abuse-controls` | rate limiting added |
| 4 | Reproducible Torch install | `security/reproducible-torch-install` | `torch==2.5.0+cpu` pinned |
| 5 | Torch risk assessment | `security/document-torch-risk` | `torch-risk-assessment.md` |
| 6 | Semantic model hardening | `security/harden-semantic-model-loading` | revision pinned, `trust_remote_code=False` |
| 7 | Final runtime verification | `security/final-runtime-verification` | `final-security-verification.md` |
| 8 | CI enforcement | `security/enforce-residual-risk-checks` | gitleaks + dependency-review |
| 9 | Close-out | `docs/close-security-remediation` | this document |

## Dependency versions (final)

| Package | Version |
|---|---|
| Python | 3.11, 3.12 |
| torch | 2.5.0+cpu (CPU-only) |
| transformers | 4.57.6 |
| sentence-transformers | 5.6.1 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |
| fastapi | 0.142.2 |
| starlette | 1.3.1 |
| pydantic | 2.9.2 |
| httpx | 0.27.2 |
| anyio | 4.15.1 |
| slowapi | 0.1.9 |
| python-dotenv | 1.2.4 |
| uvicorn | 0.30.1 |
| numpy | 2.1.3 |

## Advisory status

### Resolved (by upgrade)

| Package | Advisory | Fixed in | Iteration |
|---|---|---|---|
| python-dotenv | PYSEC-2026-2270 | 1.2.2 | 1 |
| fastapi/starlette | PYSEC-2026-161, -248, -249, -2280, -2281 | fastapi 0.142.2 / starlette 1.3.1 | 1, 6 |
| pytest | PYSEC-2026-1845 | 9.0.3 | 4 |
| anyio | GHSA-82r6-8w77-94w6 | 4.15.1 | 4 |
| sentence-transformers | CVE-2026-68770 | 5.6.1 | 5 |

### Accepted risks (unreachable in configuration)

| Package | Advisory | Reason |
|---|---|---|
| transformers 4.57.6 | 6 unique advisories | Require `trust_remote_code=True` or untrusted model loading |
| torch 2.5.0+cpu | 9 advisories (CVE-2025-32434, CVE-2025-46148, CVE-2025-2999, CVE-2025-2998, CVE-2025-3000, CVE-2025-2149, CVE-2025-2148, CVE-2025-3730, CVE-2025-63396) | Require `torch.load`, RNN ops, JIT, profiler, or CTC loss — none used |
| setuptools 78.1.0 | PYSEC-2025-49, PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w | Build-time only, not imported at runtime; Windows-only path escape |

### Update 2026-10-25

PyPA/OSV advisory database added 5 new entries between 2026-10-12 and 2026-10-25. Actions:

- **certifi** upgraded `2026.7.22 -> 2026.10.19` (PYSEC-2026-4320, compromised intermediate CA).
- **filelock** upgraded `3.32.3 -> 4.2.1` (CVE-2026-112940, CVE-2026-112913, CVE-2026-112907).
- **setuptools** `GHSA-xmj3-9gf7-h52w` accepted as risk (build-time only, upgrade blocked by torch).

Full classification: `docs/security/advisory-exclusions-audit.md`.

### Tooling limitations

- **torch** is not covered by `pip-audit` (installed from `download.pytorch.org/whl/cpu`). Separate `verify_torch.py` + manual assessment.

## CI enforcement

| Check | Tool | Trigger |
|---|---|---|
| Runtime audit | `pip-audit` | push, PR, weekly |
| Dev audit | `pip-audit` | push, PR, weekly |
| Torch version | `verify_torch.py` | push, PR, weekly |
| Dependency consistency | `pip check` | push, PR, weekly |
| Dependency review | `actions/dependency-review-action@v4` | PR only |
| Secret scanning | `gitleaks` (OSS binary) | push, PR, weekly |

## Dependabot alerts

GitHub Dependabot shows **28 open alerts** (14 unique × 2 files: `requirements.txt` + `requirements-dev.txt`). All correspond to advisory entries already classified in this document and in `torch-risk-assessment.md` / `2026-10-transformers-audit.md`. They are **not applicable** to this project's runtime configuration.

These alerts should be dismissed with the rationale "not applicable — see torch-risk-assessment.md / transformers-audit.md" via the GitHub Security UI. The dismiss action does not suppress future alerts for new advisories.

## Next review

**2027-04-10**, or earlier if a new critical advisory appears. The weekly Security workflow will catch new advisories automatically.

## Related documents

- `docs/security/torch-risk-assessment.md`
- `docs/security/2026-10-transformers-audit.md`
- `docs/security/2026-10-dependency-audit.md`
- `docs/security/2026-10-final-audit.md`
- `docs/security/residual-risk-baseline.md`
- `docs/security/reproducible-torch-install.md`
- `docs/security/final-security-verification.md`
