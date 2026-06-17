# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.1.x | Yes |
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Send a report to **alexey1pechorin@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- The version of `llm-inference-benchmark` where you observed the issue

You will receive an acknowledgement within **48 hours** and a status update within **7 days**.

If the vulnerability is confirmed, a fix will be released as a patch version and credited to you
in the release notes (unless you prefer to remain anonymous).

## Scope

This project is a local benchmarking harness. It does not expose a network service, store user
data, or handle authentication in production deployments. The primary security-relevant surfaces are:

- **Config file parsing** — YAML deserialization of user-supplied config files.
- **Path handling** — run names and output paths are validated against path-traversal attacks.
- **API key handling** — the `openai` backend reads API keys from environment variables only;
  they are never written to config files, logs, CSV output, or run manifests.
- **Docker image** — the published ghcr.io image runs as a non-root user and does not expose ports.

## Non-Scope

- Vulnerabilities in third-party dependencies (report to the upstream project).
- Issues that require physical access to the machine running the benchmark.
- Denial-of-service from running large models on insufficient hardware.
