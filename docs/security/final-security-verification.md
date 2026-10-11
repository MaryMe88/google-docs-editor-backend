# Final Runtime Security Verification

**Date:** 2026-10-11  
**Reviewer:** security/final-runtime-verification  
**Tracking issue:** #35

## Scope

Verification was performed in a clean virtual environment (`.venv-security-final`) to avoid contamination from the long-lived `.venv-baseline` dev environment.

## Versions

| Component | Version |
|---|---|
| Python | 3.12.2 |
| pip | 26.2.1 |
| torch | 2.5.0+cpu |
| torch.version.cuda | None |
| torch.cuda.is_available() | False |
| fastapi | 0.142.2 |
| starlette | 1.3.1 |
| uvicorn | 0.30.1 |
| pydantic | 2.9.2 |
| python-dotenv | 1.2.4 |
| httpx | 0.27.2 |
| anyio | 4.15.1 |
| slowapi | 0.1.9 |
| sentence-transformers | 5.6.1 |
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |
| numpy | 2.1.3 |
| setuptools | 78.1.0 (transitive, build-time only) |
| ruff | 0.8.4 |
| pytest | 9.1.1 |
| pytest-asyncio | 1.4.0 |
| pytest-cov | 7.1.0 |
| mypy | 2.3.1 |
| pip-audit | 2.10.1 |

## Installation

Torch is installed **before** runtime dependencies, in a separate step, from the official CPU index. All other packages come from PyPI via `--extra-index-url`.

```powershell
python -m venv .venv-security-final
.venv-security-final\Scripts\python.exe -m pip install --upgrade pip
.venv-security-final\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.0+cpu"
.venv-security-final\Scripts\python.exe -m pip install -r requirements.txt
```

## Verification results

| Check | Result |
|---|---|
| `pip check` (runtime only) | No broken requirements found |
| `pip check` (with dev) | No broken requirements found |
| Runtime `pip-audit` | No known vulnerabilities found, 13 ignored |
| Dev `pip-audit` | No known vulnerabilities found, 13 ignored |
| Torch (separate) | passed (2.5.0+cpu, CPU-only) |
| ruff check | All checks passed |
| ruff format --check | 93 files already formatted |
| pytest | 593 passed, 9 skipped |
| `scripts/validate_kb_tags.py` | passed |
| `scripts/audit_tags.py` | completed (informational) |
| `from src.main import app` | App OK |
| `app.openapi()` | OpenAPI 3.1.0 generated |

## Resolved advisories

These advisories were previously flagged and have been resolved by dependency upgrades in earlier iterations:

- `PYSEC-2025-217` — transformers X-CLIP checkpoint conversion RCE
- `PYSEC-2026-2288` — transformers Trainer `_load_rng_state()` RCE
- `PYSEC-2026-2289` — transformers config injection RCE
- `PYSEC-2026-2290` — transformers LightGlue model loading bypass
- `PYSEC-2026-3929` — transformers `save_pretrained()` path traversal
- `PYSEC-2026-4174` — transformers remote Python file write

**Status:** all six remain pinned to `--ignore-vuln` in `security.yml` because they are present in `transformers 4.57.6` and cannot be resolved until upgrade to `transformers 5.x`, which is blocked by incompatibility with `torch 2.5.0+cpu`. All are unreachable in this project's configuration (no `trust_remote_code=True`, no untrusted model loading, no `save_pretrained` calls). See `docs/security/2026-10-transformers-audit.md`.

- `PYSEC-2025-49` — setuptools (build-time)
- `PYSEC-2026-3447` — setuptools (build-time)

**Status:** ignored because `setuptools` is a **transitive build-time** dependency of `torch 2.5.0+cpu`. It is not imported at runtime, and the vulnerabilities affect package building, not runtime execution. Upgrading `setuptools` to 83.0.0 is a stop-condition because `torch` will reinstall its own pinned version, breaking `pip check`.

## Remaining advisories

None in runtime or dev dependencies after the above ignores.

## Accepted risks

| Risk | Mitigation | Tracking | Review |
|---|---|---|---|
| transformers 4.57.6 has 6 advisory entries | unreachable in our config; upgrade blocked by torch compat | #35 | 2026-04-10 |
| torch is not covered by pip-audit | separate `verify_torch.py` + manual assessment | #35 | 2026-04-10 |
| setuptools 78.1.0 build-time advisories | build-time only, not imported at runtime | #35 | 2026-04-10 |

## Dev dependencies in runtime

Confirmed that `pytest`, `ruff`, `mypy`, `pip-audit` are **not** required to run the application. They live only in `requirements-dev.txt` and are installed separately in CI.

## Next review date

**2026-04-10**

## Related documents

- `docs/security/torch-risk-assessment.md`
- `docs/security/2026-10-transformers-audit.md`
- `docs/security/2026-10-dependency-audit.md`
- `docs/security/2026-10-final-audit.md`
- `docs/security/residual-risk-baseline.md`
- `docs/security/reproducible-torch-install.md`
