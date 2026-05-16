# Security Policy

## Supported versions

Synara MCP is pre-1.0 and ships from `main`. Security fixes land on the
latest commit of `main`; there are no backported release branches yet.

| Version | Supported |
| ------- | --------- |
| `main` (latest) | ✅ |
| Older commits / tags | ❌ |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Report privately to **coderdayton14@gmail.com**, or use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" under the repository's **Security** tab).

Please include:

- a description of the issue and its impact,
- the affected file(s) / tool(s) / config,
- steps or a proof-of-concept to reproduce,
- any suggested remediation.

## What to expect

- **Acknowledgement** within 5 business days.
- An initial assessment and severity triage within 10 business days.
- Coordinated disclosure: a fix is prepared privately, then released on
  `main` with credit to the reporter (unless you prefer to remain anonymous).

## Scope notes

Synara MCP runs locally and stores memories in a local vector database.
When configured with a remote OpenAI-compatible embedding endpoint
(`SYNARA_EMBEDDING_URL`), episode text is sent to that endpoint — treat
that endpoint as part of your trust boundary. Never commit API keys;
`SYNARA_EMBEDDING_API_KEY` is read from the environment only.
