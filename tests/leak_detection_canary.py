"""Temporary file for manual gitleaks detection verification.

Not part of any test suite. Will be removed after the probe.
"""

# Intentionally fake GitHub-PAT-shaped string to trigger gitleaks `github-pat` rule.
# NOT a real token. Value is public on purpose.
_LEAK_CANARY = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
