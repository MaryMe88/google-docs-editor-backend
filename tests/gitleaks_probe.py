"""Temporary file for manual gitleaks detection verification.

Not part of any test suite. Will be removed after the probe.
"""

# Intentionally fake high-entropy string to trigger gitleaks generic-api-key.
# NOT a real secret. Value is public on purpose.
_PROBE_FAKE_KEY = "probe-fake-key-c4f8a2e1-9b7d-4e6a-8f3c-2a1b0d5e6f7a"
