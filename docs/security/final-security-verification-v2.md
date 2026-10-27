# Final Security Verification v2 — 2026-10-27

**SHA:** `cd32d6c08a8a4de007b5b365906b8856eaf1c36f`  
**Branch:** `security/final-runtime-verification-v2`  
**Date:** 2026-10-27  
**Tracking issue:** #35  
**Reviewer:** clean environment `.venv-final-verify` (created from scratch)

## Scope

This document supersedes `final-security-verification.md` (iteration 7, 2026-10-12). Verification performed in a **fresh virtual environment** (`.venv-final-verify`), not in the long-lived `.venv-baseline` or `.venv-security-final` — to prove the project installs and passes all checks on a clean machine.

## Versions

| Component | Version |
|---|---|
| Python | 3.12.2 |
| pip | 26.2.2 |
| torch | 2.5.0+cpu |
| torch.version.cuda | None |
| torch.cuda.is_available() | False |
| fastapi | 0.142.2 |
| starlette | 1.3.1 |
| uvicorn | 0.30.1 |
| pydantic | 2.14.0 |
| python-dotenv | 1.2.4 |
| httpx | 0.27.2 |
| certifi | 2026.10.19 |
| anyio | 4.15.1 |
| slowapi | 0.1.9 |
| filelock | 4.2.1 |
| sentence-transformers | 5.6.1 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |
| numpy | 2.1.3 |
| setuptools | 78.1.0 (transitive, build-time only) |
| ruff | 0.8.4 |
| pytest | 9.1.1 |
| mypy | 2.3.1 |
| pip-audit | 2.10.1 |
## Installation sequence

```powershell
python -m venv .venv-final-verify
.venv-final-verify\Scripts\python.exe -m pip install --upgrade pip
.venv-final-verify\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.0+cpu"
.venv-final-verify\Scripts\python.exe -m pip install -r requirements.txt
.venv-final-verify\Scripts\python.exe -m pip install -r requirements-dev.txt
Torch installed before runtime dependencies, in a separate step from the official CPU index. This is the exact installation flow for Render (buildCommand in render.yaml).

## Verification results

| Check | Result |
|---|---|
| `pip check` (runtime + dev) | No broken requirements found |
| `import torch` | 2.5.0+cpu, cuda=None, cuda_available=False |
| `ruff check src tests scripts` | All checks passed |
| `ruff format --check` | 96 files already formatted |
| `mypy` (5 CI-scope files) | Success: no issues found |
| `pytest tests/ -q` | **617 passed, 9 skipped** |
| `scripts/validate_kb_tags.py` | passed |
| `scripts/audit_tags.py` | informational |
| OpenAPI (`app.openapi()`) | 3.1.0, valid |
| `pip-audit` runtime + dev | No known vulnerabilities found, 14 ignored |
| `torch` in pip-audit | skipped (external index) |

## Advisory status

### Resolved by upgrade (this roadmap, 2026-10-09 — 2026-10-27)

| Package | Advisory | Fixed in | Iteration |
|---|---|---|---|
| python-dotenv | PYSEC-2026-2270 | 1.2.2 | 1 |
| fastapi/starlette | PYSEC-2026-161, -248, -249, -2280, -2281 | fastapi 0.142.2 / starlette 1.3.1 | 1, 6 |
| pytest | PYSEC-2026-1845 | 9.0.3 | 4 |
| anyio | GHSA-82r6-8w77-94w6 | 4.15.1 | 4 |
| sentence-transformers | CVE-2026-68770 | 5.6.1 | 5 |
| certifi | PYSEC-2026-4320 | 2026.10.19 | out-of-plan 2026-10-25 |
| filelock | CVE-2026-112940, -112913, -112907 | 4.2.1 | out-of-plan 2026-10-25 |

### Accepted risks (unreachable / blocked)

| Package | Advisory | Reason |
|---|---|---|
| transformers 4.57.6 | 6 unique advisories | Require `trust_remote_code=True`, untrusted model loading, or `save_pretrained`. Not used. |
| torch 2.5.0+cpu | 10 advisories | Require `torch.load`, RNN ops, JIT, profiler, CTC loss, or LSTM. None used. |
| setuptools 78.1.0 | PYSEC-2025-49, PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w | Build-time only; upgrade blocked by torch 2.5.0+cpu. |
| torch in pip-audit | n/a (tooling limitation) | Installed from external CPU index; see `torch-risk-assessment.md`. |

Full classification: `advisory-exclusions-audit.md`.

### Transitive version differences (vs baseline 2026-10-12)

Minor patch bumps from PyPI resolution (all within semver, not pinned in `requirements*.txt`):

- `websockets` 17.2 → 17.3
- `wrapt` 2.5.0 → 2.5.1
- `pluggy` 1.6.0 → 1.7.0
- `coverage` 7.16.2 → 7.17.0
- `ast-serialize` 0.12.1 → 0.13.1
- `platformdirs` 4.13.0 → 4.13.6
- `msgpack` 1.2.3 → 1.2.4
- `pip` 26.2.1 → 26.2.2

These do not affect runtime code, CI checks, or test outcomes (confirmed: 617 passed in clean env).

## Production smoke test (2026-10-27, Render)

| Request | Expected | Result |
|---|---|---|
| `GET /livez` without key | 200 | ✅ 200 |
| `POST /api/edit` without key | 401 | ✅ 401 |
| `POST /api/edit` with wrong key | 401 | ✅ 401 |
| `POST /api/edit` with correct key | 200 | ✅ 200 (LLM call succeeded) |

This confirms `ENV=production` from `render.yaml` works and the auth fail-fast is active.

## Remaining risks

None high or critical. All advisory-classified as accepted risks with review date.

## Next review

**2027-04-10**, or earlier if:
- new `critical` advisory appears (weekly `pip-audit` CI will catch);
- `torch >= 2.7` adopted (unlocks `setuptools` upgrade, `transformers 5.x`, closes several torch advisories).

## Related documents

- `docs/security/torch-risk-assessment.md`
- `docs/security/advisory-exclusions-audit.md`
- `docs/security/security-remediation-summary.md`
- `docs/security/residual-risk-baseline.md`
- `docs/security/reproducible-torch-install.md`
- `docs/security/2026-10-transformers-audit.md`
