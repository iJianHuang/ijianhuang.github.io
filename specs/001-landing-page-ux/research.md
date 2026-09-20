# Research: Landing Page UX Improvement

## Decision

Keep the landing experience as a lightweight static web page built with the existing HTML/CSS structure rather than introducing a framework or build tool.

## Rationale

- The repository is already organized as a set of browser-based demos with a single landing page entry point.
- The current UX problem is primarily one of discoverability and clarity, not application complexity.
- A static, simple structure matches the constitution’s requirements for browser-first learning, minimal dependencies, and project isolation.
- The target improvement is mostly content hierarchy, demo presentation, and readability, which can be achieved without changing the underlying demo architecture.

## Alternatives considered

- Framework-based landing page: Rejected because it adds setup complexity and is unnecessary for a static repo of demos.
- Minimal text-only homepage: Rejected because it would not improve project discovery or invite exploration enough.
- Rewriting all project pages: Rejected because the issue is specifically scoped to the entry experience and the demo logic is already functioning.

## Findings

The site currently exposes multiple machine learning demos through a single page with a basic card-like layout. The main opportunity is to make the purpose of the repository and each experiment clearer, more visually engaging, and easier to navigate at a glance.

No unresolved product or technical clarifications remain for this feature. The scope is limited to the user experience of the homepage and project discovery flow.
