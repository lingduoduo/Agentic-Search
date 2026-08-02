# Tool Invoke Form Implementation Plan

**Goal:** Replace the raw JSON textarea in the tool invoke dialog with labelled inputs generated from the tool's own JSON Schema.

**Architecture:** Pure schema→fields helpers in `web/src/toolSchema.ts`, consumed by a rewritten `InvokeModal`. Schemas a flat form cannot express fall back to the existing JSON editor.

**Tech Stack:** React 19, TypeScript, Vitest + @testing-library/react.

**Global Constraints**

- Work on branch `fix/tool-invoke-form`. Never commit to `main`.
- No backend changes; the invoke API and its validation are unchanged.
- No tool may become harder to invoke than it is today.
- `npm run typecheck` and `npx vitest run` pass before commit.

## Tasks

- [x] **Task 1 — Schema helpers.** `toolFormFromSchema`, `humanizeName`,
      `initialValues`, `buildArguments`, `validate` as pure functions.
      *Verify:* the real `web_search` schema yields one required `stringList`
      field; scalars map to their kinds; nested objects and arrays-of-objects
      report `supported: false`.

- [x] **Task 2 — Render the controls.** A `FieldControl` per kind, with
      description, required marker, and add/remove for string lists.
      *Verify:* the dialog shows a labelled input and the description; no JSON
      editor by default.

- [x] **Task 3 — Submit and validate.** Build arguments from values; block
      submission on missing required fields.
      *Verify:* typing "faiss" invokes with `{queries: ["faiss"]}`; empty
      required field shows an error and makes no request.

- [x] **Task 4 — Keep the escape hatch.** JSON mode for unsupported schemas and
      on demand, carrying form values across.
      *Verify:* a nested schema starts in JSON mode with an explanation; the
      toggle round-trips.

- [x] **Task 5 — Styles.**
      *Verify in a browser, not just tests* — this is where the global
      `label { display: grid }` collision surfaced.

## Verification

| Gate | Command | Result |
| --- | --- | --- |
| Frontend | `npx vitest run` | 186 passed (19 new) |
| Types | `npm run typecheck` | clean |
| Backend | `python3 -m pytest` | 2821 passed (untouched) |
| Visual | playwright against the running stack | form renders; validation blocks empty submit |
