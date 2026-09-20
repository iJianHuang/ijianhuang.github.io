# Jian's ML Code Camp Constitution

## Core Principles

### I. Browser-First Learning
This repository is organized around browser-based machine learning experiments and demos. Features should be easy to open, run, and inspect in a local web environment without requiring a heavy application framework or complex build pipeline.

### II. Mixed-Stack Compatibility
The repo contains multiple projects with different implementations and runtime patterns, including plain HTML/CSS/JavaScript, TensorFlow.js, canvas-based simulations, custom ML logic, and CSV-driven examples. New work must respect the local stack of the target project instead of forcing a single framework across the entire repository.

### III. Educational Clarity Over Abstraction
Code should remain readable and understandable for learning. Visualizations, model behavior, and data flow should be obvious to a developer inspecting the demo directly. Avoid adding unnecessary indirection when a small, explicit implementation is clearer.

### IV. Minimal Dependencies and Local Assets
Prefer self-contained browser code, local assets, and lightweight dependencies. When external libraries are used, they must be justified by the demo's learning goals and documented clearly. Reproducibility and low-friction execution are preferred over infrastructure-heavy setups.

### V. Project Isolation
Each project area is effectively its own mini-experiment. Shared code should be intentionally reused only when it adds clear value; otherwise, each demo should remain independently understandable and maintainable.

## Technology Context

This repository currently spans a range of browser- and JavaScript-oriented patterns:

- Static HTML/CSS pages for demos and navigation
- Vanilla JavaScript for model logic and UI behavior
- TensorFlow.js for browser-based training and inference
- Canvas 2D rendering for simulations and visual feedback
- CSV-based data examples for house sales and prediction tasks
- Custom algorithms such as KNN, linear regression, and multiclass classification

Changes should be aligned with the actual stack of the target project, not with a repo-wide abstraction that does not exist.

## Development Workflow

- Keep each demo or feature in its own directory with clear entry points and assets.
- Prefer local scripts and data files over introducing project-wide packaging unless the feature genuinely requires it.
- When adding a new project, document the stack, model type, data source, and browser entry page.
- Validate features by running the relevant page or demo in the browser and checking the actual learning behavior.
- Preserve the educational value of the examples: readable code, understandable training flow, and visible outputs are required.

## Governance

This constitution governs all work in the repository, including sub-projects that use different technical approaches. No single framework or stack may be imposed on unrelated demos without a clear project-specific reason. If a new feature introduces broader tooling or dependencies, the justification must be documented and balanced against the repository's emphasis on lightweight, browser-first experiments.

**Version**: 1.0.0 | **Ratified**: 2026-09-20 | **Last Amended**: 2026-09-20
