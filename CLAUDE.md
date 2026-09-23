# CLAUDE.md

## Mission

Implement the SIH 2026 marine oil-spill ISTE system according to `README.md`.

`README.md` is the scientific source of truth. Do not replace its methodology with a simpler deterministic backtracking approach.

## Workflow

Work on **ONE task at a time** from `tasks.md`.

Before each task:
1. Read `progress.md`.
2. Inspect the existing code related to the task.
3. Implement only that task.
4. Run focused tests.
5. Create/update the manual-check notebook or artifact required by the task.
6. Update `tasks.md`.
7. Update `progress.md`.
8. Update `memory.md` only when a durable scientific decision is discovered.
9. STOP.

Never start the next task automatically.

## Code rules

- `.py` = reusable/core scientific logic.
- `.ipynb` = experiments, plots, debugging, and manual validation.
- Keep modules small and focused.
- Do not implement future stages early.
- Prefer explicit parameters/configuration over hidden constants.
- Do not hard-code a particular NetCDF layout into scientific equations.

## Scientific guardrails

Follow the equations, assumptions, interpolation rules, boundary policy, validity rules, diagnostics, and interpretation limits defined in `README.md`.

Do not silently invent:
- physical terms,
- uncertainty models,
- thresholds,
- observation models,
- forcing perturbations,
- probability interpretations.

Do not claim mathematical or physical validation merely because code executes successfully.

## Manual verification

A task is not fully `DONE` until its required manual artifact has been inspected.

Use:
- `TODO` before work,
- `IN_PROGRESS` during implementation,
- `PENDING_REVIEW` after automated tests/manual artifact are ready,
- `DONE` only after manual approval.

## Response format

After each task:

```text
Task:
Status:
Files changed:
Tests:
Manual check:
What to inspect:
Known limitations:
STOPPED — waiting for approval
```
