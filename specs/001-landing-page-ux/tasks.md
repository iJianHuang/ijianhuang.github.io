# Tasks: Landing Page UX Improvement

**Input**: Design documents from `/specs/001-landing-page-ux/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm repo structure and page entry points before UX changes

- [X] T001 Audit the current landing page structure in index.html and Content/Site.css
- [X] T002 Review the project entry links and demo destinations in index.html
- [X] T003 [P] Verify the README and homepage messaging align with the existing project purpose in README.md and index.html

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the content hierarchy and page structure needed before implementing story-specific UX improvements

**Checkpoint**: Foundation ready - user story implementation can begin

- [X] T004 Define the homepage narrative and primary callouts for the landing-page experience in index.html
- [X] T005 [P] Map the project cards and visual hierarchy needed for quick discovery in Content/Site.css
- [X] T006 [P] Decide the final copy structure for the key sections and demo summaries in index.html
- [X] T007 Capture the supporting documentation notes for each demo in README.md and index.html

---

## Phase 3: User Story 1 - Discover the project quickly (Priority: P1) 🎯 MVP

**Goal**: Help a first-time visitor understand the repository purpose and find the main demos immediately

**Independent Test**: Open the homepage and confirm that a visitor can identify the repo as a collection of ML demos and spot the key project links without confusion.

### Implementation for User Story 1

- [X] T008 [P] [US1] Rewrite the landing-page introduction in index.html so the site purpose is immediately clear
- [X] T009 [P] [US1] Add a stronger hero or summary section in index.html to frame the repo as a learning playground for ML experiments
- [X] T010 [US1] Improve the arrangement of the most important project cards in index.html so high-priority demos stand out clearly
- [X] T011 [US1] Update the project card copy in index.html to explain each demo in plain language and encourage click-through
- [X] T012 [US1] Adjust spacing and card layout in Content/Site.css to support easier scanning and stronger visual hierarchy

**Checkpoint**: User Story 1 should be fully functional and independently testable

---

## Phase 4: User Story 2 - Understand each experiment before selecting it (Priority: P2)

**Goal**: Help visitors understand the value of each demo before they click into it

**Independent Test**: Review each project card and confirm that the purpose of the experiment is understandable without reading supporting docs.

### Implementation for User Story 2

- [X] T013 [P] [US2] Refresh the WebCam project description in index.html to explain the demo's value and use case clearly
- [X] T014 [P] [US2] Refresh the Self Driving Car project description in index.html to emphasize the learning goal and demo experience
- [X] T015 [P] [US2] Refresh the House Sales regression and classification descriptions in index.html for clarity and consistency
- [X] T016 [US2] Standardize card copy and text tone across the landing page in index.html
- [X] T017 [US2] Refine visual emphasis for descriptions and labels in Content/Site.css so each card reads clearly at a glance

**Checkpoint**: User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Navigate the page comfortably across devices (Priority: P3)

**Goal**: Keep the landing page readable and usable on common desktop and mobile viewport sizes

**Independent Test**: View the homepage on different screen widths and confirm the content remains readable and the primary actions are still easy to reach.

### Implementation for User Story 3

- [X] T018 [P] [US3] Review the current responsive behavior in Content/Site.css and identify layout constraints for narrower screens
- [X] T019 [US3] Update card sizing and layout rules in Content/Site.css for mobile-friendly readability
- [X] T020 [US3] Adjust the landing page spacing and alignment in index.html so the layout remains balanced across viewport sizes
- [X] T021 [US3] Ensure body and section spacing remain readable and visually consistent across the page in Content/Site.css

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final UX refinement and consistency cleanup across the landing page

- [X] T022 [P] Review the homepage copy for tone consistency and readability in index.html
- [X] T023 [P] Final polish the visual styling in Content/Site.css for spacing, contrast, and hierarchy
- [X] T024 [P] Align the landing page messaging with the repo summary in README.md
- [X] T025 Check all demo links from the landing page still point to valid destinations
- [X] T026 Validate the homepage experience end-to-end in a browser and confirm the project discovery flow feels inviting and clear

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories
- **User Story phases (Phase 3-5)**: Depend on Foundational completion
- **Polish (Phase 6)**: Depends on all main UX work being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can begin after Foundational completion; no dependency on other stories
- **User Story 2 (P2)**: Can begin after Foundational completion and can be developed independently
- **User Story 3 (P3)**: Can begin after Foundational completion and can be developed independently

### Parallel Opportunities

- T001, T002, and T003 can run in parallel during Setup
- T005 and T006 can run in parallel during Foundational work
- Within User Story 1, T008 and T009 can be done in parallel
- Within User Story 2, T013 through T015 can be completed in parallel
- Within User Story 3, T018 and T019 can be completed in parallel
- Final polish tasks in Phase 6 can be done in parallel when the main story work is complete

---

## Parallel Example: User Story 1

```bash
# Parallel review and iteration for discovery-focused improvements
Task: "Rewrite the landing-page introduction in index.html"
Task: "Add a stronger hero or summary section in index.html"
Task: "Adjust spacing and card layout in Content/Site.css"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate the landing-page entry experience in the browser
5. Stop and confirm the homepage clearly communicates project purpose and main demos

### Incremental Delivery

1. Complete Setup + Foundational
2. Implement User Story 1 to improve discovery and clarity
3. Implement User Story 2 to improve understanding of each demo
4. Implement User Story 3 to improve responsiveness and readability
5. Finish with polish and cross-cutting UX validation

### Parallel Team Strategy

With multiple contributors:

1. One person can handle content and copy updates in index.html
2. Another can tune layout and spacing in Content/Site.css
3. A third can validate links and overall usability across screen sizes

This keeps the work aligned while preserving the independence of each user story.
