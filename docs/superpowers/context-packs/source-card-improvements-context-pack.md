# Generated Context Pack

# Source Card Improvements Design Spec

## Sources

- [Specification: 2026-06-16-source-card-improvements-design.md](../specs/2026-06-16-source-card-improvements-design.md)

## Specification Context

### 2. Architecture

Single-file change: `web/src/components/SourceGrid.tsx`. Each card becomes a small controlled component (`SourceCard`) that owns its own `expanded` and `copied` state. The parent `SourceGrid` remains a thin mapper.

---

### 6. Testing Strategy

**File:** `web/src/components/__tests__/SourceGrid.test.tsx`

- Render one card with long content — assert `.source-content--clamped` class present
- Click "show more ▾" — assert class removed, button text changes to "show less ▴"
- Click "show less ▴" — assert class re-applied
- Mock `navigator.clipboard.writeText` — click "⎘ copy" — assert button text becomes "copied ✓"; after 1.5 s assert resets to "⎘ copy"
- Assert card has `id="source-[1]"` when `document.citation === "[1]"`

---

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
