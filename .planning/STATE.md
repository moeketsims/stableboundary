# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-08-24)

**Core value:** Users can fit a near-Gaussian stable model and trust that fast reduced inference is never presented without an explicit scope, numerical status, and fallback decision.  
**Current focus:** Phase 1 — Working Theorem-Faithful Package

## Current Position

Phase: 1 of 9 (Working Theorem-Faithful Package)  
Plan: 0 of TBD in current phase  
Status: Ready to plan  
Last activity: 2026-08-24 — Created a fine-grained roadmap with all 43 v1 requirements mapped exactly once.

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: No execution data

*Updated after each plan completion.*

## Accumulated Context

### Decisions

Decisions are logged in `PROJECT.md` Key Decisions.

- Phase 1 must end in an installed, fixed-seed exact finite-cell fit with known location and scale; metadata-only scaffolding is not acceptable.
- Exact finite-cell inference is the reduced default; limiting Poisson/Gamma-Beta inference remains an explicitly named approximation.
- Independent full-posterior comparison precedes automatic fallback and any reliability claim.
- Certification is a proof-and-enclosure kill gate, not a label obtainable from simulation or ordinary quadrature agreement.
- Four-parameter inference uses a recorded pilot likelihood and fixed raw main-sample bins; same-sample plug-in standardization is unsupported.

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 4: The finite-sample posterior-discrepancy bound and conservative numerical enclosure are unproved research dependencies; failure stops the certification claim.
- Phase 5-6: The four-parameter workflow has a sound likelihood outline but no validated bin design or production posterior engine yet.
- Phase 9: No scientifically qualified empirical dataset has been selected.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Interoperability | Thin R wrapper over the validated Python core | Deferred to v2 | Initialization |
| Extended models | Exact-Gaussian spike, one-sided boundary, regression/time-series adapters, and multiscale cells | Deferred to v2 | Initialization |

## Session Continuity

Last session: 2026-08-24  
Stopped at: Roadmap and initial state created; Phase 1 is ready for detailed planning.  
Resume file: None
