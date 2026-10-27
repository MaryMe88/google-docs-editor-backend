# Advisory Exclusions Audit — 2026-10-25

**Date:** 2026-10-25  
**Trigger:** `pip-audit` found 5 new advisories (base updated since 2026-10-12)  
**Tracking issue:** #35  
**Next review:** 2027-04-10

## Summary

Between the final verification (2026-10-12) and 2026-10-25, the PyPA/OSV advisory database added 5 new entries affecting our dependency set. This document classifies each one and records the action taken.

## Classification

| Advisory | Package | Installed → Action | Fix | Action | Reason |
|---|---|---|---|---|---|
| PYSEC-2026-4320 | certifi | 2026.7.22 → **2026.10.19** | 2026.10.19 | **Upgraded** | TLS CA compromise; certifi used by httpx → OpenRouter, HuggingFace |
| CVE-2026-112940 | filelock | 3.32.3 → **4.2.1** | 4.2.1 | **Upgraded** | POSIX symlink attack on ReadWriteLock; fix available, no conflict with torch |
| CVE-2026-112913 | filelock | 3.32.3 → **4.2.1** | 4.2.1 | **Upgraded** | Windows-only ReadWriteLock; covered by upgrade |
| CVE-2026-112907 | filelock | 3.32.3 → **4.2.1** | 4.2.1 | **Upgraded** | Windows-only oplock; covered by upgrade |
| GHSA-xmj3-9gf7-h52w | setuptools | 78.1.0 (unchanged) | 84.1.0 | **Accepted risk** | Windows-only build-time path escape; setuptools pinned by torch 2.5.0+cpu, cannot upgrade |
## Changes applied

Both pins are explicit to prevent transitive downgrade:

- `certifi==2026.10.19` — pulled by `httpx` without upper bound; explicit pin guarantees the fixed version.
- `filelock==4.2.1` — pulled by `torch` without upper bound; `4.2.1` is the first fixed version, no conflict with `torch 2.5.0+cpu` (verified via `pip check`).

In `.github/workflows/security.yml`, added `--ignore-vuln GHSA-xmj3-9gf7-h52w` to both `runtime-audit` and `dev-audit` jobs, with comment explaining why.

## Verification

| Check | Result |
|---|---|
| `pip install -r requirements.txt` | clean |
| `pip check` | No broken requirements found |
| `pytest tests/ -q` | **617 passed, 9 skipped** |
| `pip-audit` runtime | **No known vulnerabilities found, 14 ignored** |
| `pip-audit` dev | same |
| `torch` | skipped (external index, see `torch-risk-assessment.md`) |

## Accepted risk: setuptools GHSA-xmj3-9gf7-h52w

- **Type:** Windows-only path escape via `namespace_packages.txt` during legacy wheel installation.
- **Runtime:** not applicable — `setuptools` is build-time only, not imported in runtime code.
- **Upgrade blocked:** `torch 2.5.0+cpu` declares `setuptools` as a direct dependency and installs `78.1.0`; cannot upgrade without breaking torch installation.
- **Review:** 2027-04-10, or when `torch` >= 2.7 is adopted.

## Next review

**2027-04-10**, or earlier if a new `critical` advisory appears (weekly `pip-audit` CI will catch it).

## Related documents

- `docs/security/residual-risk-baseline.md`
- `docs/security/torch-risk-assessment.md`
- `docs/security/security-remediation-summary.md`
- `docs/security/final-security-verification.md`

## Full inventory of --ignore-vuln in security.yml (2026-10-27)

All 9 advisory exclusions in `.github/workflows/security.yml` (both `runtime-audit` and `dev-audit` jobs) have:

- advisory ID (as `--ignore-vuln` flag);
- reason (as inline comment);
- reference to issue #35 (in header comment, line 24);
- review date 2027-04-10 (in header comment, line 24);
- documentation (in `docs/security/`).

| Advisory | Package | Reason | Documented in |
|---|---|---|---|
| PYSEC-2025-217 | transformers | X-CLIP conversion RCE (not used) | 2026-10-transformers-audit.md |
| PYSEC-2026-2288 | transformers | Trainer `_load_rng_state()` RCE (not used) | same |
| PYSEC-2026-2289 | transformers | config injection RCE (not used) | same |
| PYSEC-2026-2290 | transformers | LightGlue bypass `trust_remote_code` (not used) | same |
| PYSEC-2026-3929 | transformers | `save_pretrained()` path traversal (not used) | same |
| PYSEC-2026-4174 | transformers | remote Python write without consent (not used) | same |
| PYSEC-2025-49 | setuptools | build-time only, pulled by torch | final-security-verification.md |
| PYSEC-2026-3447 | setuptools | build-time only, pulled by torch | final-security-verification.md |
| GHSA-xmj3-9gf7-h52w | setuptools | Windows-only build-time path escape | this document, section "Accepted risk" |

**Note on `residual-risk-baseline.md`:** that document (iteration 0) lists accepted risks at package level (`transformers 4.57.6`, `torch 2.5.0+cpu`, `setuptools 78.1.0`) without individual advisory IDs. `security.yml` operates at advisory-ID level. This is not a conflict — both are consistent in intent, at different granularity.

**Audit result (2026-10-27): passed.** All exclusions documented, review date current, `pip-audit` state matches the ignore list.
