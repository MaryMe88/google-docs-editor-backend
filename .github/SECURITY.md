# Security Policy

## Supported versions

Only the latest version on `main` is supported. Older commits are not
maintained.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues.**

If you find a vulnerability, report it privately via one of these channels:

1. **GitHub Security Advisories** — use the "Security" tab in this repository
   and click "Report a vulnerability". This is the preferred channel.
2. **Direct contact** — if the first option is not available, contact the
   repository owner directly. Do not include secrets, API keys or exploit
   payloads in the first message.

## What to include

- A short description of the issue.
- Steps to reproduce, if safe to share.
- The potential impact.
- Any suggested mitigation.

## What not to include

- Real API keys or credentials.
- User data or production text samples.
- Full exploit payloads in the first message — we will ask if needed.

## Response

We aim to acknowledge reports within 3 business days. Fix timelines depend
on severity and complexity.

## Scope

This project is a backend service for LLM-based text editing.

In scope:

- Authentication and API key handling.
- Rate limiting.
- Data leaks via logs or responses.
- Vulnerabilities in the request/response pipeline.

Out of scope:

- Vulnerabilities in third-party dependencies — please report them
  upstream (though we still want to know if they affect this project).
- Issues that require physical access to the deployment environment.
