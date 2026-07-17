# Remove Dead Frontend Detail Fetchers — Implementation Plan

> **For agentic workers:** single mechanical task; verify with typecheck + build.

**Goal:** Delete the three zero-caller frontend symbols (`getTool`, `getConnector`, `ConnectorDetailView`).

## Global Constraints

- Frontend only. No backend change (the detail routes are test-covered / part of the REST surface).
- `npm run typecheck` and `npm run build` must pass (they are the gate — a lingering reference fails the type-check).

---

### Task 1: Delete the dead symbols

**Files:**
- Modify: `web/src/api.ts`
- Modify: `web/src/types.ts`

- [ ] **Step 1: Remove `getConnector` from api.ts**

Delete the block:
```ts
export function getConnector(
  connectorId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ConnectorDetailView> {
  return requestJson<ConnectorDetailView>(`/admin/connectors/${connectorId}`, {
    signal: init?.signal,
  });
}
```

- [ ] **Step 2: Remove `getTool` from api.ts**

Delete the block:
```ts
export function getTool(
  name: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ToolView> {
  return requestJson<ToolView>(`/admin/tools/${name}`, { signal: init?.signal });
}
```

- [ ] **Step 3: Remove the now-unused `ConnectorDetailView` import in api.ts**

In the type import block at the top of `web/src/api.ts`, remove the
`ConnectorDetailView,` line.

- [ ] **Step 4: Remove the `ConnectorDetailView` interface from types.ts**

Delete:
```ts
export interface ConnectorDetailView extends ConnectorView {
  attempts: IndexAttemptView[];
  document_count: number;
}
```

- [ ] **Step 5: Verify**

```bash
cd web && npm run typecheck && npm run build
grep -rn "getTool\|getConnector\|ConnectorDetailView" src && echo "STILL PRESENT" || echo "all removed"
```
Expected: typecheck + build pass; grep finds nothing (or only unrelated hits).

- [ ] **Step 6: Commit**

```bash
git add web/src/api.ts web/src/types.ts
git commit -m "chore: remove dead getTool/getConnector/ConnectorDetailView frontend code"
```

---

## Self-Review

- Spec coverage: all three symbols + the unused import → Task 1. ✓
- No backend change (constraint). ✓
- Gate is the typecheck/build (a missed reference fails it). ✓
