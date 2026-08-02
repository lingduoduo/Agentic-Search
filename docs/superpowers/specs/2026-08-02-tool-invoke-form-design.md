# Schema-driven tool invoke form

**Date:** 2026-08-02
**Status:** Approved

## Problem

The tool invoke dialog asked the user to hand-write a JSON arguments object into
a bare textarea pre-filled with `{}`. To invoke `web_search` you had to already
know to type `{"queries": ["faiss"]}`.

Two distinct failures.

**The hint was dead code.** The dialog built a schema hint —
`{"queries": "..."}` — and passed it as the textarea's `placeholder`. But
`argsJson` is initialised to `"{}"`, and a placeholder only renders when the
field is empty. The hint has never been visible to anyone.

**The schema was treated as decoration.** `web_search` declares everything a
form needs:

```json
{"queries": {"type": "array", "items": {"type": "string"},
             "description": "One or more search queries to run in parallel."}}
required: ["queries"]
```

Name, type, description, required-ness — all present, none shown. The only
feedback was after clicking Invoke: "Invalid JSON", or a server-side validation
error.

A JSON Schema *is* a form definition. The dialog had one and ignored it.

## Goals

- Type a query into a labelled box; never hand-write JSON for ordinary tools.
- The schema's description and required-ness are visible before invoking.
- Problems are reported before the request, not after.
- No tool becomes harder to invoke than it is today.

## Non-goals

- No nested-object form builder. Schemas a flat form cannot express keep the
  JSON editor.
- No change to the invoke API or to server-side validation.
- The Dev Console's separate ToolsPanel is untouched; it does not invoke.

## Design

### Schema → fields

`web/src/toolSchema.ts` holds pure functions — `toolFormFromSchema`,
`initialValues`, `buildArguments`, `validate` — so the mapping is testable
without rendering.

| Schema | Control |
| --- | --- |
| `string` | text input (textarea when the name looks like free text) |
| `string` + `enum` | select |
| `number` / `integer` | number input |
| `boolean` | checkbox |
| `array` of `string` | repeatable inputs, add/remove |
| anything else | *unsupported* → JSON editor |

`supported: false` is returned for the whole schema rather than skipping a field.
Rendering a partial form would silently drop the arguments it could not
represent, which is worse than asking for JSON.

### Behaviour

- Blank optional fields are **omitted**, not sent as `""` — an empty string is a
  real value to a tool and would silently change what it does.
- Required-but-empty and non-numeric numbers are reported inline before the
  request is made.
- "Edit as JSON" stays available for supported schemas, and carries the current
  form values into the editor so nothing is retyped.
- A tool with no parameters says so instead of showing an empty form.

## Verification

Driven in a real browser against the running stack:

- `web_search` renders "Queries *", its description, one input, "+ Add another".
- Invoking empty shows "Queries is required." and makes no request.
- The accessibility tree exposes the control as `textbox "Queries required"`.

186 frontend tests (19 new), typecheck clean. Backend untouched: 2821 pass.

## Note

The first render put the required asterisk on its own line. Cause: a global
`label { display: grid }` rule, which makes every child of a label its own grid
row. Only visible by looking at the page — the tests passed. Fixed by making
`.tool-field > label` an explicit row flex.

## Risks

- Field-kind inference is heuristic: the textarea is chosen by a name pattern
  (`text`, `content`, `body`, `prompt`), so an unusual name gets a single-line
  input. Harmless — the value is still a string.
- `buildArguments` omits `false` booleans, treating unchecked as unset. Correct
  for optional flags; a tool with a required boolean defaulting true would need
  explicit tri-state handling.
