# figbuilder M2 — Interactive Layout Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the read-only M1 canvas into a layout editor — select, drag, resize, snap, align, distribute, nudge, group, undo — so a paper figure's geometry can be arranged by hand instead of by editing `subplots_adjust` numbers and re-running a notebook cell.

**Architecture:** All layout math is PURE TYPESCRIPT with no Python counterpart: the browser is the only place it runs, and Python's export consumes finished rects. Every operation is a pure function over `Rect`s, unit-tested without React. The canvas and panels are thin views over a single editor store; every mutation goes through a command so undo is uniform. Panel tiles keep coming from the M1 server unchanged — a resize re-requests one tile, a move re-requests nothing.

**Tech Stack:** TypeScript 6, React 19, Vite 8, vitest 4. No new runtime dependency. The existing FastAPI server (`figbuilder/server.py`) is reused; one route is added.

**Spec:** `docs/plans/2026-08-24-figbuilder-design.md` — see "Layout model" (absolute rects, the two-box rule, snapping, align/distribute, groups) and the M2 rows of the "Scope boundaries" table in `docs/plans/2026-08-24-figbuilder-m0-m1-plan.md`.

## Global Constraints

- TypeScript strict; the repo's tsconfig sets `erasableSyntaxOnly`, so **no TS parameter-property shorthand** (`constructor(private x: number)`) — declare fields explicitly. This already bit the M1 implementer.
- Layout math lives in `web/src/layout/` and must import NOTHING from React, the DOM, or `api.ts`. It is pure functions over plain data, tested with vitest alone.
- **No new npm runtime dependency.** No state-management library: use React's `useReducer` + context. Dev dependencies are likewise frozen; vitest is already present.
- Python: the only permitted change is to `figbuilder/server.py`. Do NOT touch `figbuilder/render.py`, `compose.py`, `annot.py`, `svgutil.py`, `figure.py`, `bundle.py`, or anything under `utils/` or `scripts/`.
- `utils/` remains consumed unmodified.
- Rects are `[x, y, w, h]` in **root-figure fractions**, y-up from the bottom-left, matching `fig.add_axes`. The canvas renders in SVG points with y-down; the flip happens in exactly ONE place (Task 2's `rectToScreen`). Do not flip anywhere else.
- Never commit `.h5`/`.png`/`.svg`/`.pdf` outputs; `figures/` is gitignored (rule anchored to `/figures/`). `web/node_modules` stays ignored.
- Run BOTH suites before every commit: `python -m pytest tests/test_figbuilder_*.py -q` (164 passing at M2 start) and `cd web && npx vitest run` (5 passing at M2 start).

## What M1 actually left behind (read this before Task 1)

```
web/src/
  api.ts                  getFigure/getBundle/getPanelTypes/renderPanel  (37 lines)
  canvas/Canvas.tsx       READ-ONLY: wheel zoom + drag pan, no selection (78 lines)
  canvas/namespaceIds.ts  mirrors figbuilder/svgutil.py (64 lines)
  annot/emit.ts           TS annotation emitter, conformance-tested vs Python
  App.tsx, main.tsx, index.css
```
`POST /api/panel` returns `{svg, ink_box, cache_hit, overflows}`. **`ink_box` is already returned per tile** — M2 consumes it for ink-box alignment and overlap detection; it does not need a server change for that.

## Two facts from M0/M1 that shape this plan

1. **A panel has two boxes.** The `rect` (axes box) and the `ink_box` (tight bbox including tick labels). They are never equal, and for a POLAR panel the ink box can be NARROWER than the rect (a circle inscribed in a square) — measured. Every align/snap/distribute operation therefore takes an explicit `by: 'axes' | 'ink'` and must not assume ink ⊇ axes.
2. **Panel-vs-panel overlap is currently undetected.** `overflows` tests only canvas bounds, so a neighbour's axis label sitting inside another panel is reported as fine. This was found by rendering, not by tests. Task 6 closes it.

---

## File Structure

| File | Responsibility |
|---|---|
| `web/src/layout/rect.ts` | `Rect` type, mm/fraction/point conversion, bbox union, containment, the ONE y-flip |
| `web/src/layout/snap.ts` | Snap candidate generation and resolution; returns snapped delta + guide lines |
| `web/src/layout/align.ts` | Align (6), distribute (2), match-size (2), across 3 reference modes |
| `web/src/layout/groups.ts` | Row/column group solving from a gutter; ungroup |
| `web/src/layout/overlap.ts` | Ink-box collision detection between panels |
| `web/src/layout/history.ts` | Command stack: apply/undo/redo, coalescing for drags |
| `web/src/state/editorStore.tsx` | `useReducer` store + context: panels, selection, guides, history, dirty flag |
| `web/src/canvas/Canvas.tsx` | (modify) selection, drag, resize handles, guide overlay |
| `web/src/canvas/Handles.tsx` | 8 resize handles for the selection |
| `web/src/canvas/Guides.tsx` | Ruler guides + live magenta smart-guide lines |
| `web/src/panels/Properties.tsx` | Numeric mm x/y/w/h entry, aspect lock, group gutter |
| `web/src/panels/Toolbar.tsx` | Align/distribute buttons, align-by toggle, undo/redo, save |
| `web/src/api.ts` | (modify) `saveFigure()` |
| `figbuilder/server.py` | (modify) `PUT /api/figure` to persist; serve built UI at `/`; helpful message when unbuilt |

---

## Task 1: Rect model and unit conversion

**Files:**
- Create: `web/src/layout/rect.ts`, `web/src/layout/rect.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `type Rect = { x: number; y: number; w: number; h: number }` (figure fractions, y-up)
  - `MM_PER_INCH`, `PT_PER_MM`
  - `fracToMm(r, figW, figH): Rect`, `mmToFrac(r, figW, figH): Rect`
  - `rectToScreen(r, figWpt, figHpt): {x,y,w,h}` — the ONLY y-flip
  - `unionRect(rects): Rect | null`
  - `edges(r): {left,right,top,bottom,cx,cy}` (figure space, y-up)
  - `rectsIntersect(a, b, eps?): boolean`
  - `normalizeRect(r): Rect` — negative w/h flipped to positive

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/rect.test.ts
import { describe, expect, it } from 'vitest';
import {
  edges, fracToMm, mmToFrac, normalizeRect, rectToScreen, rectsIntersect, unionRect,
  type Rect,
} from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('unit conversion', () => {
  it('round-trips fraction -> mm -> fraction', () => {
    const r = R(0.1, 0.2, 0.5, 0.25);
    const back = mmToFrac(fracToMm(r, 183, 140), 183, 140);
    expect(back.x).toBeCloseTo(r.x, 12);
    expect(back.h).toBeCloseTo(r.h, 12);
  });

  it('converts a rect to millimetres against the figure size', () => {
    const mm = fracToMm(R(0.5, 0.5, 0.25, 0.5), 183, 140);
    expect(mm.x).toBeCloseTo(91.5);
    expect(mm.h).toBeCloseTo(70);
  });
});

describe('rectToScreen', () => {
  it('flips y exactly once: a bottom-anchored rect lands at the BOTTOM of the canvas', () => {
    // figure space y=0 is the BOTTOM; SVG y=0 is the TOP.
    const s = rectToScreen(R(0, 0, 1, 0.25), 400, 200);
    expect(s.x).toBeCloseTo(0);
    expect(s.y).toBeCloseTo(150);      // 200 - (0 + 0.25)*200
    expect(s.h).toBeCloseTo(50);
  });

  it('a top-anchored rect lands at y=0', () => {
    const s = rectToScreen(R(0, 0.75, 1, 0.25), 400, 200);
    expect(s.y).toBeCloseTo(0);
  });
});

describe('edges', () => {
  it('reports left/right/bottom/top and centres in figure space', () => {
    const e = edges(R(0.2, 0.3, 0.4, 0.2));
    expect(e.left).toBeCloseTo(0.2);
    expect(e.right).toBeCloseTo(0.6);
    expect(e.bottom).toBeCloseTo(0.3);
    expect(e.top).toBeCloseTo(0.5);
    expect(e.cx).toBeCloseTo(0.4);
    expect(e.cy).toBeCloseTo(0.4);
  });
});

describe('unionRect', () => {
  it('bounds every input rect', () => {
    const u = unionRect([R(0.1, 0.1, 0.2, 0.2), R(0.5, 0.4, 0.2, 0.3)])!;
    expect(u.x).toBeCloseTo(0.1);
    expect(u.y).toBeCloseTo(0.1);
    expect(u.w).toBeCloseTo(0.6);
    expect(u.h).toBeCloseTo(0.6);
  });
  it('returns null for an empty list', () => {
    expect(unionRect([])).toBeNull();
  });
});

describe('rectsIntersect', () => {
  it('detects overlap and separation', () => {
    expect(rectsIntersect(R(0, 0, 0.5, 0.5), R(0.4, 0.4, 0.5, 0.5))).toBe(true);
    expect(rectsIntersect(R(0, 0, 0.4, 0.4), R(0.5, 0.5, 0.4, 0.4))).toBe(false);
  });
  it('treats mere touching as NOT overlapping', () => {
    expect(rectsIntersect(R(0, 0, 0.5, 1), R(0.5, 0, 0.5, 1))).toBe(false);
  });
});

describe('normalizeRect', () => {
  it('flips a negative-width rect produced by dragging leftwards', () => {
    const n = normalizeRect(R(0.6, 0.5, -0.2, -0.1));
    expect(n.x).toBeCloseTo(0.4);
    expect(n.y).toBeCloseTo(0.4);
    expect(n.w).toBeCloseTo(0.2);
    expect(n.h).toBeCloseTo(0.1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/rect.test.ts`
Expected: FAIL — `Failed to resolve import "./rect"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/rect.ts
/**
 * Rect model for the layout editor.
 *
 * A Rect is [x, y, w, h] in ROOT-FIGURE FRACTIONS with y growing UP from the
 * bottom-left, matching matplotlib's `fig.add_axes`. SVG has y growing DOWN,
 * so `rectToScreen` is the ONE place that flip happens. Do not flip elsewhere.
 */
export type Rect = { x: number; y: number; w: number; h: number };

export const MM_PER_INCH = 25.4;
export const PT_PER_MM = 72 / MM_PER_INCH;

export function fracToMm(r: Rect, figWmm: number, figHmm: number): Rect {
  return { x: r.x * figWmm, y: r.y * figHmm, w: r.w * figWmm, h: r.h * figHmm };
}

export function mmToFrac(r: Rect, figWmm: number, figHmm: number): Rect {
  return { x: r.x / figWmm, y: r.y / figHmm, w: r.w / figWmm, h: r.h / figHmm };
}

/** Figure-space rect -> SVG screen box. The only y-flip in the codebase. */
export function rectToScreen(
  r: Rect, figWpt: number, figHpt: number,
): { x: number; y: number; w: number; h: number } {
  return {
    x: r.x * figWpt,
    y: (1 - (r.y + r.h)) * figHpt,
    w: r.w * figWpt,
    h: r.h * figHpt,
  };
}

export function edges(r: Rect) {
  return {
    left: r.x,
    right: r.x + r.w,
    bottom: r.y,
    top: r.y + r.h,
    cx: r.x + r.w / 2,
    cy: r.y + r.h / 2,
  };
}

export function unionRect(rects: Rect[]): Rect | null {
  if (rects.length === 0) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const r of rects) {
    x0 = Math.min(x0, r.x);
    y0 = Math.min(y0, r.y);
    x1 = Math.max(x1, r.x + r.w);
    y1 = Math.max(y1, r.y + r.h);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Strict overlap: shared area only. Touching edges do NOT count. */
export function rectsIntersect(a: Rect, b: Rect, eps = 1e-9): boolean {
  return (
    a.x + a.w > b.x + eps &&
    b.x + b.w > a.x + eps &&
    a.y + a.h > b.y + eps &&
    b.y + b.h > a.y + eps
  );
}

/** A drag can produce negative w/h; canonicalise to positive extents. */
export function normalizeRect(r: Rect): Rect {
  return {
    x: r.w < 0 ? r.x + r.w : r.x,
    y: r.h < 0 ? r.y + r.h : r.y,
    w: Math.abs(r.w),
    h: Math.abs(r.h),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/rect.test.ts`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/rect.ts web/src/layout/rect.test.ts
git commit -m "feat(web): rect model, unit conversion, and the single y-flip"
```

---

## Task 2: Snapping with smart guides

**Files:**
- Create: `web/src/layout/snap.ts`, `web/src/layout/snap.test.ts`

**Interfaces:**
- Consumes: `Rect`, `edges`, `PT_PER_MM` (Task 1)
- Produces:
  - `type BoxMode = 'axes' | 'ink'`
  - `type SnapTarget = { axis: 'x' | 'y'; value: number; kind: 'margin' | 'edge' | 'center' | 'grid' | 'guide' }`
  - `buildTargets(opts): SnapTarget[]`
  - `snapRect(moving, targets, tolFrac): { rect: Rect; guides: SnapTarget[] }`
  - `screenTolToFrac(tolPx, figExtentPt, zoom): number`

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/snap.test.ts
import { describe, expect, it } from 'vitest';
import { buildTargets, screenTolToFrac, snapRect } from './snap';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('buildTargets', () => {
  it('emits figure margins on both axes', () => {
    const t = buildTargets({ others: [], gridMm: 0, figWmm: 100, figHmm: 100, guides: [] });
    const xs = t.filter((s) => s.axis === 'x' && s.kind === 'margin').map((s) => s.value);
    expect(xs).toEqual(expect.arrayContaining([0, 1]));
  });

  it('emits neighbour edges AND centres', () => {
    const t = buildTargets({
      others: [R(0.2, 0.1, 0.4, 0.3)], gridMm: 0, figWmm: 100, figHmm: 100, guides: [],
    });
    const xs = t.filter((s) => s.axis === 'x');
    expect(xs.some((s) => s.kind === 'edge' && Math.abs(s.value - 0.2) < 1e-9)).toBe(true);
    expect(xs.some((s) => s.kind === 'edge' && Math.abs(s.value - 0.6) < 1e-9)).toBe(true);
    expect(xs.some((s) => s.kind === 'center' && Math.abs(s.value - 0.4) < 1e-9)).toBe(true);
  });

  it('emits a grid when gridMm > 0 and none when it is 0', () => {
    const on = buildTargets({ others: [], gridMm: 10, figWmm: 100, figHmm: 100, guides: [] });
    expect(on.some((s) => s.kind === 'grid')).toBe(true);
    const off = buildTargets({ others: [], gridMm: 0, figWmm: 100, figHmm: 100, guides: [] });
    expect(off.some((s) => s.kind === 'grid')).toBe(false);
  });
});

describe('snapRect', () => {
  const targets = buildTargets({
    others: [R(0.2, 0.1, 0.4, 0.3)], gridMm: 0, figWmm: 100, figHmm: 100, guides: [],
  });

  it('snaps a near-miss left edge onto a neighbour edge and reports the guide', () => {
    const { rect, guides } = snapRect(R(0.205, 0.5, 0.2, 0.2), targets, 0.01);
    expect(rect.x).toBeCloseTo(0.2, 9);
    expect(rect.w).toBeCloseTo(0.2, 9);   // snapping MOVES, never resizes
    expect(guides.some((g) => g.axis === 'x' && Math.abs(g.value - 0.2) < 1e-9)).toBe(true);
  });

  it('leaves a rect alone when nothing is within tolerance', () => {
    const { rect, guides } = snapRect(R(0.44, 0.55, 0.2, 0.2), targets, 0.001);
    expect(rect.x).toBeCloseTo(0.44, 9);
    expect(guides).toHaveLength(0);
  });

  it('prefers the CLOSEST candidate when several are in range', () => {
    const t = [
      { axis: 'x' as const, value: 0.30, kind: 'edge' as const },
      { axis: 'x' as const, value: 0.34, kind: 'edge' as const },
    ];
    const { rect } = snapRect(R(0.335, 0, 0.1, 0.1), t, 0.05);
    expect(rect.x).toBeCloseTo(0.34, 9);
  });

  it('can snap the RIGHT edge, shifting x by the difference', () => {
    const t = [{ axis: 'x' as const, value: 0.6, kind: 'edge' as const }];
    const { rect } = snapRect(R(0.395, 0, 0.2, 0.1), t, 0.02);  // right edge 0.595
    expect(rect.x).toBeCloseTo(0.4, 9);
  });

  it('snaps x and y independently in one call', () => {
    const t = [
      { axis: 'x' as const, value: 0.5, kind: 'edge' as const },
      { axis: 'y' as const, value: 0.25, kind: 'edge' as const },
    ];
    const { rect } = snapRect(R(0.497, 0.252, 0.1, 0.1), t, 0.01);
    expect(rect.x).toBeCloseTo(0.5, 9);
    expect(rect.y).toBeCloseTo(0.25, 9);
  });
});

describe('screenTolToFrac', () => {
  it('shrinks the fractional tolerance as zoom increases, so snapping feels constant on screen', () => {
    const at1 = screenTolToFrac(6, 400, 1);
    const at4 = screenTolToFrac(6, 400, 4);
    expect(at4).toBeCloseTo(at1 / 4, 12);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/snap.test.ts`
Expected: FAIL — `Failed to resolve import "./snap"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/snap.ts
/**
 * Snapping for panel drags.
 *
 * Snapping MOVES a rect; it never resizes it. Tolerance is supplied in
 * FIGURE FRACTIONS but is derived from a screen-pixel budget (see
 * `screenTolToFrac`) so the feel stays constant as the user zooms.
 */
import { edges, type Rect } from './rect';

export type BoxMode = 'axes' | 'ink';

export type SnapKind = 'margin' | 'edge' | 'center' | 'grid' | 'guide';
export type SnapTarget = { axis: 'x' | 'y'; value: number; kind: SnapKind };

export type BuildTargetsOpts = {
  /** Rects of the panels NOT being dragged, in the active box mode. */
  others: Rect[];
  /** Grid spacing in mm; 0 disables the grid. */
  gridMm: number;
  figWmm: number;
  figHmm: number;
  /** User-placed ruler guides, in figure fractions. */
  guides: SnapTarget[];
};

export function buildTargets(o: BuildTargetsOpts): SnapTarget[] {
  const out: SnapTarget[] = [
    { axis: 'x', value: 0, kind: 'margin' },
    { axis: 'x', value: 1, kind: 'margin' },
    { axis: 'y', value: 0, kind: 'margin' },
    { axis: 'y', value: 1, kind: 'margin' },
  ];

  for (const r of o.others) {
    const e = edges(r);
    out.push({ axis: 'x', value: e.left, kind: 'edge' });
    out.push({ axis: 'x', value: e.right, kind: 'edge' });
    out.push({ axis: 'x', value: e.cx, kind: 'center' });
    out.push({ axis: 'y', value: e.bottom, kind: 'edge' });
    out.push({ axis: 'y', value: e.top, kind: 'edge' });
    out.push({ axis: 'y', value: e.cy, kind: 'center' });
  }

  if (o.gridMm > 0) {
    for (let mm = 0; mm <= o.figWmm; mm += o.gridMm) {
      out.push({ axis: 'x', value: mm / o.figWmm, kind: 'grid' });
    }
    for (let mm = 0; mm <= o.figHmm; mm += o.gridMm) {
      out.push({ axis: 'y', value: mm / o.figHmm, kind: 'grid' });
    }
  }

  out.push(...o.guides);
  return out;
}

/** Convert a screen-pixel tolerance into figure fractions at the current zoom. */
export function screenTolToFrac(tolPx: number, figExtentPt: number, zoom: number): number {
  return tolPx / (figExtentPt * zoom);
}

type Best = { delta: number; target: SnapTarget } | null;

function bestFor(candidates: number[], targets: SnapTarget[], axis: 'x' | 'y', tol: number): Best {
  let best: Best = null;
  for (const t of targets) {
    if (t.axis !== axis) continue;
    for (const c of candidates) {
      const d = t.value - c;
      if (Math.abs(d) <= tol && (best === null || Math.abs(d) < Math.abs(best.delta))) {
        best = { delta: d, target: t };
      }
    }
  }
  return best;
}

/**
 * Snap `moving` to the nearest target on each axis independently.
 * Returns the shifted rect and the targets that produced a snap, so the
 * caller can draw a guide line for each.
 */
export function snapRect(
  moving: Rect, targets: SnapTarget[], tolFrac: number,
): { rect: Rect; guides: SnapTarget[] } {
  const e = edges(moving);
  const gx = bestFor([e.left, e.right, e.cx], targets, 'x', tolFrac);
  const gy = bestFor([e.bottom, e.top, e.cy], targets, 'y', tolFrac);

  const guides: SnapTarget[] = [];
  if (gx) guides.push(gx.target);
  if (gy) guides.push(gy.target);

  return {
    rect: {
      x: moving.x + (gx ? gx.delta : 0),
      y: moving.y + (gy ? gy.delta : 0),
      w: moving.w,
      h: moving.h,
    },
    guides,
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/snap.test.ts`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/snap.ts web/src/layout/snap.test.ts
git commit -m "feat(web): snapping to margins, neighbour edges/centres, grid and guides"
```

---

## Task 3: Align, distribute, match-size

**Files:**
- Create: `web/src/layout/align.ts`, `web/src/layout/align.test.ts`

**Interfaces:**
- Consumes: `Rect`, `edges`, `unionRect` (Task 1)
- Produces:
  - `type AlignOp = 'left'|'hcenter'|'right'|'top'|'vcenter'|'bottom'`
  - `type RefMode = 'selection'|'first'|'figure'`
  - `alignRects(rects, op, ref): Rect[]`
  - `distributeRects(rects, axis, mode): Rect[]` with `mode: 'gaps'|'centers'`
  - `matchSize(rects, dim): Rect[]` with `dim: 'w'|'h'`

**A note the implementer must respect:** every one of these takes rects that the CALLER has already resolved to the active box mode (axes or ink). These functions know nothing about panels or ink boxes — they are pure geometry. The caller is responsible for converting an ink-box alignment back into an axes-rect delta; Task 7 does that.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/align.test.ts
import { describe, expect, it } from 'vitest';
import { alignRects, distributeRects, matchSize } from './align';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('alignRects', () => {
  const rects = [R(0.1, 0.1, 0.2, 0.1), R(0.5, 0.4, 0.3, 0.2)];

  it('aligns left to the selection bbox', () => {
    const out = alignRects(rects, 'left', 'selection');
    expect(out[0].x).toBeCloseTo(0.1);
    expect(out[1].x).toBeCloseTo(0.1);
  });

  it('aligns right, preserving each width', () => {
    const out = alignRects(rects, 'right', 'selection');
    expect(out[0].x + out[0].w).toBeCloseTo(0.8);
    expect(out[1].x + out[1].w).toBeCloseTo(0.8);
    expect(out[0].w).toBeCloseTo(0.2);
  });

  it('aligns to the FIRST rect when ref is "first"', () => {
    const out = alignRects(rects, 'left', 'first');
    expect(out[0].x).toBeCloseTo(0.1);
    expect(out[1].x).toBeCloseTo(0.1);
  });

  it('aligns to the FIGURE when ref is "figure"', () => {
    const out = alignRects(rects, 'left', 'figure');
    expect(out[0].x).toBeCloseTo(0);
    expect(out[1].x).toBeCloseTo(0);
  });

  it('centres horizontally about the reference centre', () => {
    const out = alignRects(rects, 'hcenter', 'figure');
    expect(out[0].x + out[0].w / 2).toBeCloseTo(0.5);
    expect(out[1].x + out[1].w / 2).toBeCloseTo(0.5);
  });

  it('aligns TOP using figure-space y-up semantics', () => {
    const out = alignRects(rects, 'top', 'selection');   // max top = 0.6
    expect(out[0].y + out[0].h).toBeCloseTo(0.6);
    expect(out[1].y + out[1].h).toBeCloseTo(0.6);
  });

  it('never changes any rect size', () => {
    for (const op of ['left','hcenter','right','top','vcenter','bottom'] as const) {
      const out = alignRects(rects, op, 'selection');
      out.forEach((r, i) => {
        expect(r.w).toBeCloseTo(rects[i].w);
        expect(r.h).toBeCloseTo(rects[i].h);
      });
    }
  });
});

describe('distributeRects', () => {
  it('equalises GAPS horizontally, pinning the outermost rects', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.15, 0, 0.2, 0.1), R(0.8, 0, 0.2, 0.1)];
    const out = distributeRects(rects, 'x', 'gaps');
    expect(out[0].x).toBeCloseTo(0);          // first pinned
    expect(out[2].x).toBeCloseTo(0.8);        // last pinned
    const g1 = out[1].x - (out[0].x + out[0].w);
    const g2 = out[2].x - (out[1].x + out[1].w);
    expect(g1).toBeCloseTo(g2, 9);
  });

  it('equalises CENTRES horizontally', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.15, 0, 0.3, 0.1), R(0.8, 0, 0.1, 0.1)];
    const out = distributeRects(rects, 'x', 'centers');
    const c = out.map((r) => r.x + r.w / 2);
    expect(c[1] - c[0]).toBeCloseTo(c[2] - c[1], 9);
  });

  it('distributes by position, not by input order', () => {
    const rects = [R(0.8, 0, 0.1, 0.1), R(0, 0, 0.1, 0.1), R(0.4, 0, 0.1, 0.1)];
    const out = distributeRects(rects, 'x', 'gaps');
    expect(out[1].x).toBeCloseTo(0);      // the leftmost input stays leftmost
    expect(out[0].x).toBeCloseTo(0.8);    // the rightmost stays rightmost
  });

  it('returns fewer than 3 rects unchanged', () => {
    const rects = [R(0, 0, 0.1, 0.1), R(0.5, 0, 0.1, 0.1)];
    expect(distributeRects(rects, 'x', 'gaps')).toEqual(rects);
  });
});

describe('matchSize', () => {
  it('matches widths to the FIRST rect without moving anything', () => {
    const rects = [R(0.1, 0.1, 0.25, 0.1), R(0.5, 0.4, 0.4, 0.2)];
    const out = matchSize(rects, 'w');
    expect(out[1].w).toBeCloseTo(0.25);
    expect(out[1].x).toBeCloseTo(0.5);
    expect(out[1].h).toBeCloseTo(0.2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/align.test.ts`
Expected: FAIL — `Failed to resolve import "./align"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/align.ts
/**
 * Align / distribute / match-size, as pure geometry.
 *
 * Callers pass rects ALREADY resolved to the active box mode (axes or ink);
 * these functions know nothing about panels. Alignment never changes a rect's
 * size; match-size never changes its position. All coordinates are figure
 * fractions with y growing UP, so "top" is max(y + h).
 */
import { edges, unionRect, type Rect } from './rect';

export type AlignOp = 'left' | 'hcenter' | 'right' | 'top' | 'vcenter' | 'bottom';
export type RefMode = 'selection' | 'first' | 'figure';

const FIGURE: Rect = { x: 0, y: 0, w: 1, h: 1 };

function reference(rects: Rect[], ref: RefMode): Rect {
  if (ref === 'figure') return FIGURE;
  if (ref === 'first') return rects[0];
  return unionRect(rects) ?? FIGURE;
}

export function alignRects(rects: Rect[], op: AlignOp, ref: RefMode): Rect[] {
  if (rects.length === 0) return rects;
  const R = edges(reference(rects, ref));
  return rects.map((r) => {
    const e = edges(r);
    switch (op) {
      case 'left':    return { ...r, x: R.left };
      case 'right':   return { ...r, x: R.right - r.w };
      case 'hcenter': return { ...r, x: R.cx - r.w / 2 };
      case 'bottom':  return { ...r, y: R.bottom };
      case 'top':     return { ...r, y: R.top - r.h };
      case 'vcenter': return { ...r, y: R.cy - r.h / 2 };
      default:        return { ...r, x: e.left };
    }
  });
}

export function distributeRects(
  rects: Rect[], axis: 'x' | 'y', mode: 'gaps' | 'centers',
): Rect[] {
  if (rects.length < 3) return rects;
  const size = axis === 'x' ? ('w' as const) : ('h' as const);

  // Work in POSITION order, then scatter back to the caller's ordering.
  const order = rects.map((r, i) => i).sort((a, b) => rects[a][axis] - rects[b][axis]);
  const sorted = order.map((i) => rects[i]);
  const out = sorted.map((r) => ({ ...r }));
  const first = sorted[0];
  const last = sorted[sorted.length - 1];

  if (mode === 'gaps') {
    const span = (last[axis] + last[size]) - first[axis];
    const used = sorted.reduce((s, r) => s + r[size], 0);
    const gap = (span - used) / (sorted.length - 1);
    let cursor = first[axis];
    for (let i = 0; i < out.length; i += 1) {
      out[i][axis] = cursor;
      cursor += out[i][size] + gap;
    }
  } else {
    const c0 = first[axis] + first[size] / 2;
    const c1 = last[axis] + last[size] / 2;
    const step = (c1 - c0) / (sorted.length - 1);
    for (let i = 0; i < out.length; i += 1) {
      out[i][axis] = c0 + step * i - out[i][size] / 2;
    }
  }

  const result = rects.map((r) => ({ ...r }));
  order.forEach((origIdx, k) => { result[origIdx] = out[k]; });
  return result;
}

export function matchSize(rects: Rect[], dim: 'w' | 'h'): Rect[] {
  if (rects.length === 0) return rects;
  const target = rects[0][dim];
  return rects.map((r) => ({ ...r, [dim]: target }));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/align.test.ts`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/align.ts web/src/layout/align.test.ts
git commit -m "feat(web): align, distribute and match-size geometry"
```

---

## Task 4: Group solver

**Files:**
- Create: `web/src/layout/groups.ts`, `web/src/layout/groups.test.ts`

**Interfaces:**
- Consumes: `Rect` (Task 1)
- Produces:
  - `type Group = { id: string; axis: 'x'|'y'; gutterMm: number; equal: boolean; rect: Rect }`
  - `solveGroup(group, children, figWmm, figHmm): Rect[]`
  - `groupBounds(children): Rect`
  - `gutterFromChildren(children, axis, figWmm, figHmm): number`

**Why this exists:** `figure.json` stores every panel's SOLVED rect, so rendering never needs the solver — it is needed only once panels can be dragged. The six-frame video strip becomes ONE gutter number instead of six hand-tuned rects.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/groups.test.ts
import { describe, expect, it } from 'vitest';
import { groupBounds, gutterFromChildren, solveGroup, type Group } from './groups';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });
const G = (over: Partial<Group> = {}): Group => ({
  id: 'g', axis: 'x', gutterMm: 2, equal: true, rect: R(0.1, 0.5, 0.8, 0.2), ...over,
});

describe('solveGroup', () => {
  it('lays equal-width children across the group rect with the given gutter', () => {
    const kids = [R(0,0,0.1,0.2), R(0,0,0.3,0.2), R(0,0,0.2,0.2), R(0,0,0.05,0.2)];
    const out = solveGroup(G(), kids, 100, 100);          // gutter 2mm = 0.02 frac
    expect(out).toHaveLength(4);
    const w = out[0].w;
    out.forEach((r) => expect(r.w).toBeCloseTo(w, 9));    // equal:true
    expect(out[0].x).toBeCloseTo(0.1, 9);                  // starts at group left
    expect(out[3].x + out[3].w).toBeCloseTo(0.9, 9);       // ends at group right
    expect(out[1].x - (out[0].x + out[0].w)).toBeCloseTo(0.02, 9);
  });

  it('preserves relative widths when equal is false', () => {
    const kids = [R(0,0,0.1,0.2), R(0,0,0.3,0.2)];
    const out = solveGroup(G({ equal: false, gutterMm: 0 }), kids, 100, 100);
    expect(out[1].w / out[0].w).toBeCloseTo(3, 6);
    expect(out[0].x + out[0].w + out[1].w).toBeCloseTo(0.9, 6);
  });

  it('stacks on the y axis TOP-DOWN so child 0 is visually first', () => {
    const kids = [R(0,0,0.2,0.1), R(0,0,0.2,0.1)];
    const out = solveGroup(G({ axis: 'y', gutterMm: 0 }), kids, 100, 100);
    // group rect y=0.5 h=0.2 -> child 0 occupies the UPPER half
    expect(out[0].y + out[0].h).toBeCloseTo(0.7, 9);
    expect(out[1].y).toBeCloseTo(0.5, 9);
  });

  it('gives every child the group\'s cross-axis extent', () => {
    const kids = [R(0,0,0.1,0.05), R(0,0,0.1,0.9)];
    const out = solveGroup(G(), kids, 100, 100);
    out.forEach((r) => {
      expect(r.y).toBeCloseTo(0.5, 9);
      expect(r.h).toBeCloseTo(0.2, 9);
    });
  });

  it('returns a single child filling the group rect', () => {
    const out = solveGroup(G(), [R(0,0,0.1,0.1)], 100, 100);
    expect(out[0]).toEqual(G().rect);
  });

  it('never produces a negative width when the gutter exceeds the group', () => {
    const kids = [R(0,0,0.1,0.1), R(0,0,0.1,0.1), R(0,0,0.1,0.1)];
    const out = solveGroup(G({ gutterMm: 90 }), kids, 100, 100);
    out.forEach((r) => expect(r.w).toBeGreaterThanOrEqual(0));
  });
});

describe('groupBounds', () => {
  it('is the union of its children', () => {
    const b = groupBounds([R(0.1,0.2,0.2,0.1), R(0.5,0.2,0.2,0.1)]);
    expect(b.x).toBeCloseTo(0.1);
    expect(b.w).toBeCloseTo(0.6);
  });
});

describe('gutterFromChildren', () => {
  it('recovers the mean gap in mm, so ungroup->regroup is stable', () => {
    const kids = [R(0.1,0,0.2,0.1), R(0.35,0,0.2,0.1), R(0.60,0,0.2,0.1)];
    expect(gutterFromChildren(kids, 'x', 100, 100)).toBeCloseTo(5, 6);
  });
  it('is 0 for a single child', () => {
    expect(gutterFromChildren([R(0,0,0.2,0.1)], 'x', 100, 100)).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/groups.test.ts`
Expected: FAIL — `Failed to resolve import "./groups"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/groups.ts
/**
 * Row/column groups: declarative whitespace where it is wanted.
 *
 * A group owns a rect and a single gutter; it solves its children's rects so
 * the six-frame video strip is ONE number instead of six. Everything outside
 * a group stays freely placed, and ungrouping leaves children exactly where
 * the solver last put them.
 *
 * Y-axis groups stack TOP-DOWN (child 0 highest) because that is reading
 * order, even though figure space has y growing up.
 */
import { unionRect, type Rect } from './rect';

export type Group = {
  id: string;
  axis: 'x' | 'y';
  gutterMm: number;
  equal: boolean;
  rect: Rect;
};

export function solveGroup(
  g: Group, children: Rect[], figWmm: number, figHmm: number,
): Rect[] {
  const n = children.length;
  if (n === 0) return [];
  if (n === 1) return [{ ...g.rect }];

  const along = g.axis === 'x' ? ('w' as const) : ('h' as const);
  const extentMm = g.axis === 'x' ? figWmm : figHmm;
  const gutter = Math.max(0, g.gutterMm / extentMm);

  const span = g.axis === 'x' ? g.rect.w : g.rect.h;
  const totalGutter = gutter * (n - 1);
  const usable = Math.max(0, span - totalGutter);

  let sizes: number[];
  if (g.equal) {
    sizes = new Array(n).fill(usable / n);
  } else {
    const sum = children.reduce((s, c) => s + c[along], 0);
    sizes = sum > 0
      ? children.map((c) => (c[along] / sum) * usable)
      : new Array(n).fill(usable / n);
  }

  const out: Rect[] = [];
  if (g.axis === 'x') {
    let cursor = g.rect.x;
    for (let i = 0; i < n; i += 1) {
      out.push({ x: cursor, y: g.rect.y, w: sizes[i], h: g.rect.h });
      cursor += sizes[i] + gutter;
    }
  } else {
    // Top-down: start at the group's TOP edge and walk downwards.
    let top = g.rect.y + g.rect.h;
    for (let i = 0; i < n; i += 1) {
      out.push({ x: g.rect.x, y: top - sizes[i], w: g.rect.w, h: sizes[i] });
      top -= sizes[i] + gutter;
    }
  }
  return out;
}

export function groupBounds(children: Rect[]): Rect {
  return unionRect(children) ?? { x: 0, y: 0, w: 0, h: 0 };
}

/** Mean gap between consecutive children, in mm. Used when forming a group. */
export function gutterFromChildren(
  children: Rect[], axis: 'x' | 'y', figWmm: number, figHmm: number,
): number {
  if (children.length < 2) return 0;
  const along = axis === 'x' ? ('w' as const) : ('h' as const);
  const sorted = [...children].sort((a, b) => a[axis] - b[axis]);
  let total = 0;
  for (let i = 1; i < sorted.length; i += 1) {
    total += sorted[i][axis] - (sorted[i - 1][axis] + sorted[i - 1][along]);
  }
  const meanFrac = total / (sorted.length - 1);
  return meanFrac * (axis === 'x' ? figWmm : figHmm);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/groups.test.ts`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/groups.ts web/src/layout/groups.test.ts
git commit -m "feat(web): row/column group solver with a single gutter"
```

---

## Task 5: Panel-vs-panel overlap detection

**Files:**
- Create: `web/src/layout/overlap.ts`, `web/src/layout/overlap.test.ts`

**Interfaces:**
- Consumes: `Rect`, `rectsIntersect` (Task 1)
- Produces:
  - `type Collision = { a: string; b: string; area: number }`
  - `findCollisions(entries): Collision[]` where `entries: {id: string; ink: Rect}[]`
  - `outOfBounds(entries): string[]`

**Why this task exists, in one sentence:** the M0/M1 exporter's `overflows` flag tests ONLY canvas bounds, so during the first end-to-end render a panel's axis label was drawn INSIDE its neighbour and nothing reported it — the per-tile `ink_box` needed to catch it was already being returned and simply unused.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/overlap.test.ts
import { describe, expect, it } from 'vitest';
import { findCollisions, outOfBounds } from './overlap';
import type { Rect } from './rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

describe('findCollisions', () => {
  it('reports a pair whose INK boxes overlap even though their axes rects do not', () => {
    // This is the real M0 failure: the scut panel's xlabel sat inside the violin panel.
    const hits = findCollisions([
      { id: 'scut', ink: R(0.07, 0.30, 0.90, 0.14) },
      { id: 'zheight', ink: R(0.42, 0.40, 0.22, 0.24) },
    ]);
    expect(hits).toHaveLength(1);
    expect([hits[0].a, hits[0].b].sort()).toEqual(['scut', 'zheight']);
    expect(hits[0].area).toBeGreaterThan(0);
  });

  it('is silent when panels merely touch', () => {
    expect(findCollisions([
      { id: 'a', ink: R(0, 0, 0.5, 1) },
      { id: 'b', ink: R(0.5, 0, 0.5, 1) },
    ])).toHaveLength(0);
  });

  it('reports each pair once, not twice', () => {
    const hits = findCollisions([
      { id: 'a', ink: R(0, 0, 0.6, 0.6) },
      { id: 'b', ink: R(0.4, 0.4, 0.6, 0.6) },
      { id: 'c', ink: R(0.45, 0.45, 0.1, 0.1) },
    ]);
    const keys = hits.map((h) => [h.a, h.b].sort().join('|')).sort();
    expect(new Set(keys).size).toBe(keys.length);
    expect(keys).toHaveLength(3);
  });

  it('sorts the worst overlap first', () => {
    const hits = findCollisions([
      { id: 'a', ink: R(0, 0, 1, 1) },
      { id: 'b', ink: R(0, 0, 0.9, 0.9) },
      { id: 'c', ink: R(0.95, 0.95, 0.1, 0.1) },
    ]);
    expect(hits[0].area).toBeGreaterThanOrEqual(hits[hits.length - 1].area);
  });

  it('returns nothing for a clean layout', () => {
    expect(findCollisions([
      { id: 'a', ink: R(0.0, 0.0, 0.3, 0.3) },
      { id: 'b', ink: R(0.5, 0.5, 0.3, 0.3) },
    ])).toHaveLength(0);
  });
});

describe('outOfBounds', () => {
  it('flags a panel whose ink leaves the canvas', () => {
    expect(outOfBounds([
      { id: 'ok', ink: R(0.1, 0.1, 0.2, 0.2) },
      { id: 'clipped', ink: R(-0.01, 0.5, 0.2, 0.2) },
    ])).toEqual(['clipped']);
  });
  it('tolerates a hair of floating-point slop', () => {
    expect(outOfBounds([{ id: 'a', ink: R(-1e-12, 0, 1, 1) }])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/overlap.test.ts`
Expected: FAIL — `Failed to resolve import "./overlap"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/overlap.ts
/**
 * Ink-box collision detection.
 *
 * The exporter's `overflows` flag tests only CANVAS bounds, so two panels
 * whose ink boxes collide — a neighbour's axis label drawn inside another
 * panel — were reported as fine. That was found by looking at a render, not
 * by a test. This module closes it using the `ink_box` every tile already
 * returns from `POST /api/panel`.
 */
import { rectsIntersect, type Rect } from './rect';

export type InkEntry = { id: string; ink: Rect };
export type Collision = { a: string; b: string; area: number };

function overlapArea(a: Rect, b: Rect): number {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? w * h : 0;
}

/** Every colliding pair, once each, worst first. */
export function findCollisions(entries: InkEntry[]): Collision[] {
  const out: Collision[] = [];
  for (let i = 0; i < entries.length; i += 1) {
    for (let j = i + 1; j < entries.length; j += 1) {
      const a = entries[i];
      const b = entries[j];
      if (rectsIntersect(a.ink, b.ink)) {
        out.push({ a: a.id, b: b.id, area: overlapArea(a.ink, b.ink) });
      }
    }
  }
  return out.sort((p, q) => q.area - p.area);
}

/** Panels whose ink leaves the canvas — the same condition the exporter warns on. */
export function outOfBounds(entries: InkEntry[], eps = 1e-6): string[] {
  return entries
    .filter(({ ink }) =>
      ink.x < -eps || ink.y < -eps ||
      ink.x + ink.w > 1 + eps || ink.y + ink.h > 1 + eps)
    .map((e) => e.id);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/overlap.test.ts`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/overlap.ts web/src/layout/overlap.test.ts
git commit -m "feat(web): ink-box overlap detection between panels"
```

---

## Task 6: Undo/redo command stack

**Files:**
- Create: `web/src/layout/history.ts`, `web/src/layout/history.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `type Snapshot = Record<string, Rect>` (panel id -> rect)
  - `type History = { past: Snapshot[]; present: Snapshot; future: Snapshot[] }`
  - `initHistory(present): History`
  - `commit(history, next, opts?): History` with `opts?: { coalesceKey?: string }`
  - `undo(history): History`, `redo(history): History`
  - `canUndo(h): boolean`, `canRedo(h): boolean`

**The coalescing rule matters:** a drag fires dozens of pointer-move events. Without coalescing, one drag would need dozens of undos. Consecutive commits sharing a `coalesceKey` collapse into one entry; any commit with a different key (or none) starts a fresh entry.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/layout/history.test.ts
import { describe, expect, it } from 'vitest';
import { canRedo, canUndo, commit, initHistory, redo, undo } from './history';
import type { Rect } from './rect';

const S = (x: number): Record<string, Rect> => ({ p: { x, y: 0, w: 0.1, h: 0.1 } });

describe('history', () => {
  it('undoes and redoes a single change', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    expect(h.present.p.x).toBe(1);
    h = undo(h);
    expect(h.present.p.x).toBe(0);
    h = redo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('coalesces consecutive commits sharing a key into ONE undo step', () => {
    let h = initHistory(S(0));
    for (let i = 1; i <= 20; i += 1) h = commit(h, S(i), { coalesceKey: 'drag-p' });
    expect(h.present.p.x).toBe(20);
    h = undo(h);
    expect(h.present.p.x).toBe(0);     // the whole drag, not one pointer-move
    expect(canUndo(h)).toBe(false);
  });

  it('starts a new entry when the coalesce key changes', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1), { coalesceKey: 'drag-a' });
    h = commit(h, S(2), { coalesceKey: 'drag-b' });
    h = undo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('an uncoalesced commit always starts a new entry', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = commit(h, S(2));
    h = undo(h);
    expect(h.present.p.x).toBe(1);
  });

  it('clears the redo stack once a new change is committed', () => {
    let h = initHistory(S(0));
    h = commit(h, S(1));
    h = undo(h);
    expect(canRedo(h)).toBe(true);
    h = commit(h, S(9));
    expect(canRedo(h)).toBe(false);
    expect(h.present.p.x).toBe(9);
  });

  it('is a no-op at the ends of the stack', () => {
    let h = initHistory(S(0));
    expect(canUndo(h)).toBe(false);
    h = undo(h);
    expect(h.present.p.x).toBe(0);
    h = redo(h);
    expect(h.present.p.x).toBe(0);
  });

  it('does not record a commit identical to the present', () => {
    let h = initHistory(S(0));
    h = commit(h, S(0));
    expect(canUndo(h)).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/layout/history.test.ts`
Expected: FAIL — `Failed to resolve import "./history"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/layout/history.ts
/**
 * Undo/redo over whole-layout snapshots.
 *
 * Snapshots rather than inverse-operations: a layout is a few dozen small
 * rects, so copying it is cheap and every operation becomes undoable for free
 * without each one having to describe its own inverse.
 *
 * Coalescing exists because a drag emits dozens of pointer-move commits;
 * without it one drag would cost dozens of undos.
 */
import type { Rect } from './rect';

export type Snapshot = Record<string, Rect>;

export type History = {
  past: Snapshot[];
  present: Snapshot;
  future: Snapshot[];
  /** Key of the entry currently open for coalescing, if any. */
  openKey?: string;
};

export function initHistory(present: Snapshot): History {
  return { past: [], present, future: [] };
}

function sameLayout(a: Snapshot, b: Snapshot): boolean {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => {
    const p = a[k];
    const q = b[k];
    return q !== undefined && p.x === q.x && p.y === q.y && p.w === q.w && p.h === q.h;
  });
}

export function commit(
  h: History, next: Snapshot, opts?: { coalesceKey?: string },
): History {
  if (sameLayout(h.present, next)) return h;

  const key = opts?.coalesceKey;
  // Continuing an open coalesced entry: replace `present`, do not grow `past`.
  if (key !== undefined && h.openKey === key) {
    return { past: h.past, present: next, future: [], openKey: key };
  }
  return {
    past: [...h.past, h.present],
    present: next,
    future: [],
    openKey: key,
  };
}

export function undo(h: History): History {
  if (h.past.length === 0) return h;
  const prev = h.past[h.past.length - 1];
  return {
    past: h.past.slice(0, -1),
    present: prev,
    future: [h.present, ...h.future],
    openKey: undefined,
  };
}

export function redo(h: History): History {
  if (h.future.length === 0) return h;
  const [next, ...rest] = h.future;
  return {
    past: [...h.past, h.present],
    present: next,
    future: rest,
    openKey: undefined,
  };
}

export const canUndo = (h: History): boolean => h.past.length > 0;
export const canRedo = (h: History): boolean => h.future.length > 0;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/layout/history.test.ts`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/layout/history.ts web/src/layout/history.test.ts
git commit -m "feat(web): undo/redo stack with drag coalescing"
```

---

## Task 7: Editor store

**Files:**
- Create: `web/src/state/editorStore.tsx`, `web/src/state/editorStore.test.ts`

**Interfaces:**
- Consumes: everything from Tasks 1-6
- Produces:
  - `type PanelState = { id: string; type: string; rect: Rect; ink?: Rect; group?: string | null; spec: Record<string, unknown>; data: Record<string, unknown> }`
  - `type EditorState = { figWmm; figHmm; panels: PanelState[]; groups: Group[]; selection: string[]; boxMode: BoxMode; gridMm: number; guides: SnapTarget[]; history: History; dirty: boolean }`
  - `reducer(state, action): EditorState` — the whole editor as a pure function
  - Actions: `{type:'load'}`, `'select'`, `'moveSelection'`, `'setRect'`, `'align'`, `'distribute'`, `'matchSize'`, `'nudge'`, `'setBoxMode'`, `'setInk'`, `'undo'`, `'redo'`, `'saved'`
  - `EditorProvider`, `useEditor()`

**Design rule the implementer must follow:** the reducer is PURE and is tested WITHOUT React. Every action that changes geometry routes through `commit()` so undo is uniform, and writes `dirty: true`. Rects live in exactly one place — `panels[].rect` — and `history.present` mirrors them.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/state/editorStore.test.ts
import { describe, expect, it } from 'vitest';
import { initialState, reducer, type EditorState } from './editorStore';
import type { Rect } from '../layout/rect';

const R = (x: number, y: number, w: number, h: number): Rect => ({ x, y, w, h });

function loaded(): EditorState {
  return reducer(initialState, {
    type: 'load',
    figWmm: 100, figHmm: 100,
    panels: [
      { id: 'a', type: 'line', rect: R(0.1, 0.1, 0.2, 0.2), spec: {}, data: {} },
      { id: 'b', type: 'line', rect: R(0.5, 0.5, 0.2, 0.2), spec: {}, data: {} },
      { id: 'c', type: 'line', rect: R(0.8, 0.1, 0.1, 0.2), spec: {}, data: {} },
    ],
    groups: [],
  });
}

describe('reducer', () => {
  it('loads panels and starts clean with nothing selected', () => {
    const s = loaded();
    expect(s.panels).toHaveLength(3);
    expect(s.selection).toEqual([]);
    expect(s.dirty).toBe(false);
  });

  it('selects, adds to selection, and clears', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    expect(s.selection).toEqual(['a']);
    s = reducer(s, { type: 'select', ids: ['b'], additive: true });
    expect(s.selection.sort()).toEqual(['a', 'b']);
    s = reducer(s, { type: 'select', ids: [] });
    expect(s.selection).toEqual([]);
  });

  it('moves the selection by a delta and marks the document dirty', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.05, dy: -0.02 });
    const a = s.panels.find((p) => p.id === 'a')!;
    expect(a.rect.x).toBeCloseTo(0.15);
    expect(a.rect.y).toBeCloseTo(0.08);
    expect(s.dirty).toBe(true);
  });

  it('moves ONLY the selection', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    expect(s.panels.find((p) => p.id === 'b')!.rect.x).toBeCloseTo(0.5);
  });

  it('undoes a whole coalesced drag in one step', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    for (let i = 0; i < 10; i += 1) {
      s = reducer(s, { type: 'moveSelection', dx: 0.01, dy: 0, coalesceKey: 'drag' });
    }
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
  });

  it('redo restores what undo removed', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.1, dy: 0 });
    s = reducer(s, { type: 'undo' });
    s = reducer(s, { type: 'redo' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.2);
  });

  it('aligns the selection left without touching unselected panels', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b'] });
    s = reducer(s, { type: 'align', op: 'left', ref: 'selection' });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.1);
    expect(s.panels.find((p) => p.id === 'b')!.rect.x).toBeCloseTo(0.1);
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(0.8);
  });

  it('nudges by exact millimetres', () => {
    let s = loaded();                       // 100mm figure -> 1mm == 0.01 frac
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'nudge', dxMm: 1, dyMm: 0 });
    expect(s.panels.find((p) => p.id === 'a')!.rect.x).toBeCloseTo(0.11, 9);
  });

  it('setRect replaces one panel\'s rect exactly (numeric entry)', () => {
    let s = loaded();
    s = reducer(s, { type: 'setRect', id: 'b', rect: R(0.25, 0.25, 0.5, 0.5) });
    expect(s.panels.find((p) => p.id === 'b')!.rect).toEqual(R(0.25, 0.25, 0.5, 0.5));
    expect(s.dirty).toBe(true);
  });

  it('records ink boxes from the server without dirtying the document', () => {
    let s = loaded();
    s = reducer(s, { type: 'setInk', id: 'a', ink: R(0.05, 0.05, 0.3, 0.3) });
    expect(s.panels.find((p) => p.id === 'a')!.ink).toEqual(R(0.05, 0.05, 0.3, 0.3));
    expect(s.dirty).toBe(false);          // a render result is not an edit
  });

  it('clears dirty on saved', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'moveSelection', dx: 0.01, dy: 0 });
    s = reducer(s, { type: 'saved' });
    expect(s.dirty).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/state/editorStore.test.ts`
Expected: FAIL — `Failed to resolve import "./editorStore"`

- [ ] **Step 3: Write minimal implementation**

```tsx
// web/src/state/editorStore.tsx
/**
 * The whole editor as a pure reducer, plus a thin React context over it.
 *
 * Rects live in exactly ONE place (`panels[].rect`); `history.present` mirrors
 * them so undo is uniform across drags, aligns, nudges and numeric entry. The
 * reducer imports no React and is tested without it.
 */
import { createContext, useContext, useMemo, useReducer, type ReactNode } from 'react';
import { alignRects, distributeRects, matchSize, type AlignOp, type RefMode } from '../layout/align';
import { canRedo, canUndo, commit, initHistory, redo, undo, type History, type Snapshot } from '../layout/history';
import type { Group } from '../layout/groups';
import type { BoxMode, SnapTarget } from '../layout/snap';
import type { Rect } from '../layout/rect';

export type PanelState = {
  id: string;
  type: string;
  rect: Rect;
  /** Last ink box reported by the server for this panel, if rendered. */
  ink?: Rect;
  group?: string | null;
  spec: Record<string, unknown>;
  data: Record<string, unknown>;
};

export type EditorState = {
  figWmm: number;
  figHmm: number;
  panels: PanelState[];
  groups: Group[];
  selection: string[];
  boxMode: BoxMode;
  gridMm: number;
  guides: SnapTarget[];
  history: History;
  dirty: boolean;
};

export type Action =
  | { type: 'load'; figWmm: number; figHmm: number; panels: PanelState[]; groups: Group[] }
  | { type: 'select'; ids: string[]; additive?: boolean }
  | { type: 'moveSelection'; dx: number; dy: number; coalesceKey?: string }
  | { type: 'setRect'; id: string; rect: Rect; coalesceKey?: string }
  | { type: 'align'; op: AlignOp; ref: RefMode }
  | { type: 'distribute'; axis: 'x' | 'y'; mode: 'gaps' | 'centers' }
  | { type: 'matchSize'; dim: 'w' | 'h' }
  | { type: 'nudge'; dxMm: number; dyMm: number }
  | { type: 'setBoxMode'; mode: BoxMode }
  | { type: 'setInk'; id: string; ink: Rect }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'saved' };

export const initialState: EditorState = {
  figWmm: 183, figHmm: 140,
  panels: [], groups: [], selection: [],
  boxMode: 'axes', gridMm: 1, guides: [],
  history: initHistory({}), dirty: false,
};

const snapshotOf = (panels: PanelState[]): Snapshot =>
  Object.fromEntries(panels.map((p) => [p.id, p.rect]));

const applySnapshot = (panels: PanelState[], snap: Snapshot): PanelState[] =>
  panels.map((p) => (snap[p.id] ? { ...p, rect: snap[p.id] } : p));

/** Commit a new set of panels through history, marking the document dirty. */
function withGeometry(
  state: EditorState, panels: PanelState[], coalesceKey?: string,
): EditorState {
  const history = commit(state.history, snapshotOf(panels), { coalesceKey });
  if (history === state.history) return state;   // nothing actually moved
  return { ...state, panels, history, dirty: true };
}

/** Map a transform over the selected panels only. */
function mapSelected(
  state: EditorState, fn: (rects: Rect[]) => Rect[],
): PanelState[] {
  const sel = state.panels.filter((p) => state.selection.includes(p.id));
  if (sel.length === 0) return state.panels;
  const out = fn(sel.map((p) => p.rect));
  const byId = new Map(sel.map((p, i) => [p.id, out[i]]));
  return state.panels.map((p) => (byId.has(p.id) ? { ...p, rect: byId.get(p.id)! } : p));
}

export function reducer(state: EditorState, action: Action): EditorState {
  switch (action.type) {
    case 'load': {
      const panels = action.panels;
      return {
        ...state,
        figWmm: action.figWmm,
        figHmm: action.figHmm,
        panels,
        groups: action.groups,
        selection: [],
        history: initHistory(snapshotOf(panels)),
        dirty: false,
      };
    }

    case 'select': {
      const ids = action.additive
        ? Array.from(new Set([...state.selection, ...action.ids]))
        : action.ids;
      return { ...state, selection: ids };
    }

    case 'moveSelection':
      return withGeometry(
        state,
        mapSelected(state, (rs) =>
          rs.map((r) => ({ ...r, x: r.x + action.dx, y: r.y + action.dy }))),
        action.coalesceKey,
      );

    case 'nudge': {
      const dx = action.dxMm / state.figWmm;
      const dy = action.dyMm / state.figHmm;
      return withGeometry(
        state,
        mapSelected(state, (rs) => rs.map((r) => ({ ...r, x: r.x + dx, y: r.y + dy }))),
      );
    }

    case 'setRect':
      return withGeometry(
        state,
        state.panels.map((p) => (p.id === action.id ? { ...p, rect: action.rect } : p)),
        action.coalesceKey,
      );

    case 'align':
      return withGeometry(state, mapSelected(state, (rs) => alignRects(rs, action.op, action.ref)));

    case 'distribute':
      return withGeometry(state, mapSelected(state, (rs) => distributeRects(rs, action.axis, action.mode)));

    case 'matchSize':
      return withGeometry(state, mapSelected(state, (rs) => matchSize(rs, action.dim)));

    case 'setBoxMode':
      return { ...state, boxMode: action.mode };

    // A render RESULT is not an edit: it must not dirty the document or
    // enter the undo stack.
    case 'setInk':
      return {
        ...state,
        panels: state.panels.map((p) => (p.id === action.id ? { ...p, ink: action.ink } : p)),
      };

    case 'undo': {
      const history = undo(state.history);
      if (history === state.history) return state;
      return { ...state, history, panels: applySnapshot(state.panels, history.present), dirty: true };
    }

    case 'redo': {
      const history = redo(state.history);
      if (history === state.history) return state;
      return { ...state, history, panels: applySnapshot(state.panels, history.present), dirty: true };
    }

    case 'saved':
      return { ...state, dirty: false };

    default:
      return state;
  }
}

type Ctx = { state: EditorState; dispatch: React.Dispatch<Action>; canUndo: boolean; canRedo: boolean };
const EditorCtx = createContext<Ctx | null>(null);

export function EditorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = useMemo(
    () => ({ state, dispatch, canUndo: canUndo(state.history), canRedo: canRedo(state.history) }),
    [state],
  );
  return <EditorCtx.Provider value={value}>{children}</EditorCtx.Provider>;
}

export function useEditor(): Ctx {
  const ctx = useContext(EditorCtx);
  if (!ctx) throw new Error('useEditor must be used inside <EditorProvider>');
  return ctx;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/state/editorStore.test.ts`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add web/src/state/editorStore.tsx web/src/state/editorStore.test.ts
git commit -m "feat(web): editor reducer with selection, geometry ops and undo"
```

---

## Task 8: Canvas selection, drag, and live guides

**Files:**
- Modify: `web/src/canvas/Canvas.tsx` (M1 version is 78 lines: wheel zoom + drag pan only)
- Create: `web/src/canvas/Guides.tsx`
- Create: `web/src/canvas/hitTest.ts`, `web/src/canvas/hitTest.test.ts`

**Interfaces:**
- Consumes: `useEditor` (Task 7), `snapRect`/`buildTargets`/`screenTolToFrac` (Task 2), `rectToScreen` (Task 1)
- Produces: `hitTestPanels(panels, pt): string | null`, `marqueeHits(panels, marquee): string[]`, `<Guides>`

**Behaviour to preserve from M1:** wheel zoom and background drag-pan must keep working. Panel drag replaces pan ONLY when the pointer goes down on a panel.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/canvas/hitTest.test.ts
import { describe, expect, it } from 'vitest';
import { hitTestPanels, marqueeHits } from './hitTest';
import type { Rect } from '../layout/rect';

const P = (id: string, x: number, y: number, w: number, h: number) =>
  ({ id, rect: { x, y, w, h } as Rect });

const panels = [
  P('back', 0.0, 0.0, 1.0, 1.0),
  P('mid', 0.2, 0.2, 0.4, 0.4),
  P('front', 0.3, 0.3, 0.1, 0.1),
];

describe('hitTestPanels', () => {
  it('returns the TOPMOST panel under the point (last wins)', () => {
    expect(hitTestPanels(panels, { x: 0.35, y: 0.35 })).toBe('front');
  });
  it('falls through to a lower panel outside the top one', () => {
    expect(hitTestPanels(panels, { x: 0.55, y: 0.55 })).toBe('mid');
  });
  it('returns null on empty space', () => {
    expect(hitTestPanels([P('a', 0, 0, 0.1, 0.1)], { x: 0.9, y: 0.9 })).toBeNull();
  });
});

describe('marqueeHits', () => {
  it('selects panels that INTERSECT the marquee', () => {
    const ids = marqueeHits(panels, { x: 0.25, y: 0.25, w: 0.2, h: 0.2 });
    expect(ids).toEqual(expect.arrayContaining(['mid', 'front']));
  });
  it('selects nothing for a marquee over empty space', () => {
    expect(marqueeHits([P('a', 0, 0, 0.1, 0.1)], { x: 0.5, y: 0.5, w: 0.2, h: 0.2 })).toEqual([]);
  });
  it('handles a marquee dragged up-left (negative w/h)', () => {
    const ids = marqueeHits([P('a', 0.1, 0.1, 0.2, 0.2)], { x: 0.5, y: 0.5, w: -0.35, h: -0.35 });
    expect(ids).toEqual(['a']);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/canvas/hitTest.test.ts`
Expected: FAIL — `Failed to resolve import "./hitTest"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/canvas/hitTest.ts
/** Pointer hit-testing in FIGURE space (y-up). Later panels are on top. */
import { normalizeRect, rectsIntersect, type Rect } from '../layout/rect';

export type Hittable = { id: string; rect: Rect };

export function hitTestPanels(panels: Hittable[], pt: { x: number; y: number }): string | null {
  for (let i = panels.length - 1; i >= 0; i -= 1) {
    const r = panels[i].rect;
    if (pt.x >= r.x && pt.x <= r.x + r.w && pt.y >= r.y && pt.y <= r.y + r.h) {
      return panels[i].id;
    }
  }
  return null;
}

export function marqueeHits(panels: Hittable[], marquee: Rect): string[] {
  const m = normalizeRect(marquee);
  return panels.filter((p) => rectsIntersect(p.rect, m)).map((p) => p.id);
}
```

```tsx
// web/src/canvas/Guides.tsx
/** Magenta smart-guide lines drawn over the canvas during a snapped drag. */
import type { SnapTarget } from '../layout/snap';

export function Guides(
  { guides, figWpt, figHpt }: { guides: SnapTarget[]; figWpt: number; figHpt: number },
) {
  return (
    <g id="smart-guides" pointerEvents="none">
      {guides.map((g, i) =>
        g.axis === 'x' ? (
          <line key={i} x1={g.value * figWpt} y1={0} x2={g.value * figWpt} y2={figHpt}
                stroke="#e11d8f" strokeWidth={0.5} strokeDasharray="3 2" />
        ) : (
          // figure y is up; SVG y is down — flip for display only.
          <line key={i} x1={0} y1={(1 - g.value) * figHpt} x2={figWpt} y2={(1 - g.value) * figHpt}
                stroke="#e11d8f" strokeWidth={0.5} strokeDasharray="3 2" />
        ))}
    </g>
  );
}
```

Then rewrite `Canvas.tsx` to add, ON TOP of M1's existing wheel-zoom and background-pan:
- a `mode` ref that is `'pan' | 'drag' | 'marquee' | null`, decided on `pointerdown` by `hitTestPanels`: a hit starts `'drag'` (selecting that panel first, additive on shift), a miss on the background starts `'marquee'` when shift is held and `'pan'` otherwise;
- during `'drag'`, convert the pointer delta from screen px to figure fractions (divide by `figWpt * zoom` / `figHpt * zoom`, negating dy because figure y is up), build snap targets from the UNSELECTED panels in the active `boxMode`, call `snapRect` on the primary selected panel, apply the SNAPPED delta to the whole selection via `dispatch({type:'moveSelection', dx, dy, coalesceKey: 'drag'})`, and hold the returned guides in local state for `<Guides>`;
- clear the guides and the coalesce key on `pointerup`;
- render a selection outline (2px `#2563eb`) around each selected panel using `rectToScreen`;
- render the marquee rectangle while `mode === 'marquee'`, dispatching `select` with `marqueeHits` on release.

Keep tile fetching exactly as M1 has it, and keep calling `dispatch({type:'setInk', ...})` with each tile's `ink_box`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run` (expect 6 files passing) then `npm run build`
Expected: vitest green; build succeeds

- [ ] **Step 5: Manual check, then commit**

Start both servers, open `http://localhost:5173`, and confirm: clicking a panel outlines it; dragging moves it; a magenta guide appears when an edge lines up with a neighbour; shift-drag on the background marquee-selects; background drag still pans; the wheel still zooms.

```bash
git add web/src/canvas/
git commit -m "feat(web): panel selection, drag with snapping, and smart guides"
```

---

## Task 9: Resize handles and keyboard nudge

**Files:**
- Create: `web/src/canvas/Handles.tsx`, `web/src/canvas/resize.ts`, `web/src/canvas/resize.test.ts`
- Modify: `web/src/canvas/Canvas.tsx`

**Interfaces:**
- Consumes: `Rect`, `normalizeRect` (Task 1); `useEditor` (Task 7)
- Produces: `type HandleId = 'nw'|'n'|'ne'|'e'|'se'|'s'|'sw'|'w'`, `resizeRect(rect, handle, dx, dy, opts?): Rect`, `<Handles>`

**Why resize needs a re-render but move does not:** matplotlib tick and axis-label text does NOT scale with the axes box, so a stretched tile would misrepresent the output. Moving a panel re-renders nothing; resizing re-requests exactly that one tile.

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/canvas/resize.test.ts
import { describe, expect, it } from 'vitest';
import { resizeRect } from './resize';
import type { Rect } from '../layout/rect';

const R: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };

describe('resizeRect', () => {
  it('e drags the right edge, keeping x', () => {
    const out = resizeRect(R, 'e', 0.1, 0);
    expect(out.x).toBeCloseTo(0.2);
    expect(out.w).toBeCloseTo(0.5);
    expect(out.h).toBeCloseTo(0.4);
  });

  it('w drags the left edge, moving x and shrinking w', () => {
    const out = resizeRect(R, 'w', 0.1, 0);
    expect(out.x).toBeCloseTo(0.3);
    expect(out.w).toBeCloseTo(0.3);
  });

  it('n grows upward in FIGURE space (y-up)', () => {
    const out = resizeRect(R, 'n', 0, 0.1);
    expect(out.y).toBeCloseTo(0.2);
    expect(out.h).toBeCloseTo(0.5);
  });

  it('s drags the bottom edge, moving y', () => {
    const out = resizeRect(R, 's', 0, 0.1);
    expect(out.y).toBeCloseTo(0.3);
    expect(out.h).toBeCloseTo(0.3);
  });

  it('ne moves both the right and top edges', () => {
    const out = resizeRect(R, 'ne', 0.1, 0.1);
    expect(out.w).toBeCloseTo(0.5);
    expect(out.h).toBeCloseTo(0.5);
    expect(out.x).toBeCloseTo(0.2);
    expect(out.y).toBeCloseTo(0.2);
  });

  it('normalizes a rect dragged inside-out rather than emitting negative w', () => {
    const out = resizeRect(R, 'e', -0.6, 0);
    expect(out.w).toBeGreaterThanOrEqual(0);
    expect(out.x).toBeLessThanOrEqual(0.2);
  });

  it('honours a minimum size so a panel cannot collapse to nothing', () => {
    const out = resizeRect(R, 'e', -0.4, 0, { minW: 0.05, minH: 0.05 });
    expect(out.w).toBeCloseTo(0.05);
  });

  it('preserves aspect when asked', () => {
    const out = resizeRect(R, 'se', 0.2, 0, { aspect: true });
    expect(out.w / out.h).toBeCloseTo(R.w / R.h, 6);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/canvas/resize.test.ts`
Expected: FAIL — `Failed to resolve import "./resize"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/canvas/resize.ts
/**
 * Rect resizing from an edge or corner handle.
 *
 * All coordinates are figure fractions with y growing UP, so 'n' increases
 * height without moving y, and 's' moves y while shrinking height.
 */
import { normalizeRect, type Rect } from '../layout/rect';

export type HandleId = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

export type ResizeOpts = { minW?: number; minH?: number; aspect?: boolean };

export function resizeRect(
  r: Rect, handle: HandleId, dx: number, dy: number, opts: ResizeOpts = {},
): Rect {
  const minW = opts.minW ?? 0;
  const minH = opts.minH ?? 0;

  let { x, y, w, h } = r;
  if (handle.includes('e')) w += dx;
  if (handle.includes('w')) { x += dx; w -= dx; }
  if (handle.includes('n')) h += dy;
  if (handle.includes('s')) { y += dy; h -= dy; }

  let out = normalizeRect({ x, y, w, h });

  if (opts.aspect && r.h !== 0) {
    const ratio = r.w / r.h;
    // Drive height from width so a horizontal drag feels authoritative.
    const newH = out.w / ratio;
    if (handle.includes('s')) out = { ...out, y: out.y + (out.h - newH) };
    out = { ...out, h: newH };
  }

  if (out.w < minW) {
    if (handle.includes('w')) out = { ...out, x: out.x + (out.w - minW) };
    out = { ...out, w: minW };
  }
  if (out.h < minH) {
    if (handle.includes('s')) out = { ...out, y: out.y + (out.h - minH) };
    out = { ...out, h: minH };
  }
  return out;
}
```

```tsx
// web/src/canvas/Handles.tsx
/** Eight resize handles around the selection bbox, in SCREEN coordinates. */
import type { HandleId } from './resize';

const HANDLES: { id: HandleId; fx: number; fy: number }[] = [
  { id: 'nw', fx: 0, fy: 0 }, { id: 'n', fx: 0.5, fy: 0 }, { id: 'ne', fx: 1, fy: 0 },
  { id: 'w', fx: 0, fy: 0.5 }, { id: 'e', fx: 1, fy: 0.5 },
  { id: 'sw', fx: 0, fy: 1 }, { id: 's', fx: 0.5, fy: 1 }, { id: 'se', fx: 1, fy: 1 },
];

const CURSOR: Record<HandleId, string> = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize',
};

export function Handles(
  { box, size = 6, onGrab }:
  { box: { x: number; y: number; w: number; h: number }; size?: number;
    onGrab: (h: HandleId, e: React.PointerEvent) => void },
) {
  return (
    <g id="handles">
      {HANDLES.map(({ id, fx, fy }) => (
        <rect
          key={id}
          x={box.x + fx * box.w - size / 2}
          y={box.y + fy * box.h - size / 2}
          width={size} height={size}
          fill="#ffffff" stroke="#2563eb" strokeWidth={1}
          style={{ cursor: CURSOR[id] }}
          onPointerDown={(e) => { e.stopPropagation(); onGrab(id, e); }}
        />
      ))}
    </g>
  );
}
```

Then in `Canvas.tsx`:
- render `<Handles>` around the selection bbox (screen coords via `rectToScreen` on the union of selected rects) whenever exactly one panel is selected;
- on handle grab, enter `mode='resize'`, remembering the handle and the panel's starting rect; on pointer-move call `resizeRect` with the fraction-space delta and `{minW: 5/figWmm, minH: 5/figHmm, aspect: e.shiftKey}` and dispatch `setRect` with `coalesceKey: 'resize'`;
- **re-request that one tile on pointer-UP, not during the drag** — matplotlib text does not scale with the box, so an interim stretched tile would lie; show the previous tile scaled to the new box while dragging as an explicit preview, then replace it with the true render on release;
- add a keydown handler on the canvas container: arrows dispatch `nudge` with 0.25 mm, 1 mm when shift is held; `ctrl/cmd+z` → `undo`; `ctrl/cmd+shift+z` → `redo`; `escape` → `select []`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run` then `npm run build`
Expected: all suites green; build succeeds

- [ ] **Step 5: Manual check, then commit**

Confirm: handles appear on a selected panel; dragging `e` widens it and the tile re-renders sharp on release (tick labels NOT stretched); shift-drag preserves aspect; arrow keys nudge 0.25 mm and shift+arrow 1 mm; ctrl+z undoes a whole drag in one step.

```bash
git add web/src/canvas/
git commit -m "feat(web): resize handles, aspect lock, and keyboard nudge/undo"
```

---

## Task 10: Properties panel and toolbar

**Files:**
- Create: `web/src/panels/Properties.tsx`, `web/src/panels/Toolbar.tsx`, `web/src/panels/format.ts`, `web/src/panels/format.test.ts`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: `useEditor` (Task 7), `fracToMm`/`mmToFrac` (Task 1), `findCollisions`/`outOfBounds` (Task 5)
- Produces: `parseMm(text, fallback): number`, `formatMm(v): string`, `<Properties>`, `<Toolbar>`

- [ ] **Step 1: Write the failing test**

```typescript
// web/src/panels/format.test.ts
import { describe, expect, it } from 'vitest';
import { formatMm, parseMm } from './format';

describe('parseMm', () => {
  it('parses a plain number', () => expect(parseMm('12.5', 0)).toBeCloseTo(12.5));
  it('tolerates a unit suffix and whitespace', () => {
    expect(parseMm(' 12.5 mm ', 0)).toBeCloseTo(12.5);
  });
  it('returns the fallback for junk, so a half-typed value never wipes a rect', () => {
    expect(parseMm('', 7)).toBe(7);
    expect(parseMm('-', 7)).toBe(7);
    expect(parseMm('abc', 7)).toBe(7);
  });
  it('accepts negatives', () => expect(parseMm('-3.25', 0)).toBeCloseTo(-3.25));
});

describe('formatMm', () => {
  it('shows two decimals', () => expect(formatMm(12.3456)).toBe('12.35'));
  it('trims a trailing .00', () => expect(formatMm(12)).toBe('12'));
  it('renders negative zero as 0', () => expect(formatMm(-0)).toBe('0'));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/panels/format.test.ts`
Expected: FAIL — `Failed to resolve import "./format"`

- [ ] **Step 3: Write minimal implementation**

```typescript
// web/src/panels/format.ts
/** Millimetre text <-> number for the numeric-entry fields. */
export function parseMm(text: string, fallback: number): number {
  const cleaned = text.trim().replace(/mm$/i, '').trim();
  const v = Number(cleaned);
  return cleaned === '' || Number.isNaN(v) ? fallback : v;
}

export function formatMm(v: number): string {
  const s = (Object.is(v, -0) ? 0 : v).toFixed(2);
  return s.replace(/\.00$/, '');
}
```

```tsx
// web/src/panels/Properties.tsx
/**
 * Numeric x/y/w/h entry in MILLIMETRES for the selected panel.
 *
 * mm, not fractions: a paper figure is specified in mm, and 183 mm
 * double-column is a number the researcher thinks in.
 */
import { fracToMm, mmToFrac } from '../layout/rect';
import { useEditor } from '../state/editorStore';
import { formatMm, parseMm } from './format';

export function Properties() {
  const { state, dispatch } = useEditor();
  const sel = state.panels.filter((p) => state.selection.includes(p.id));
  if (sel.length !== 1) {
    return <aside style={{ padding: 12, width: 220 }}>
      <p>{sel.length === 0 ? 'No panel selected' : `${sel.length} panels selected`}</p>
    </aside>;
  }
  const p = sel[0];
  const mm = fracToMm(p.rect, state.figWmm, state.figHmm);

  const set = (k: 'x' | 'y' | 'w' | 'h') => (text: string) => {
    const next = { ...mm, [k]: parseMm(text, mm[k]) };
    dispatch({ type: 'setRect', id: p.id, rect: mmToFrac(next, state.figWmm, state.figHmm) });
  };

  return (
    <aside style={{ padding: 12, width: 220, fontFamily: 'system-ui', fontSize: 13 }}>
      <h3 style={{ margin: '0 0 8px' }}>{p.id}</h3>
      <div style={{ color: '#666', marginBottom: 8 }}>{p.type}</div>
      {(['x', 'y', 'w', 'h'] as const).map((k) => (
        <label key={k} style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
          <span style={{ width: 16 }}>{k}</span>
          <input
            style={{ width: 90 }}
            defaultValue={formatMm(mm[k])}
            key={`${p.id}-${k}-${formatMm(mm[k])}`}
            onBlur={(e) => set(k)(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
          />
          <span style={{ color: '#888' }}>mm</span>
        </label>
      ))}
    </aside>
  );
}
```

```tsx
// web/src/panels/Toolbar.tsx
/** Align / distribute / box-mode / undo / save, plus a live layout-warning count. */
import { findCollisions, outOfBounds } from '../layout/overlap';
import { useEditor } from '../state/editorStore';
import type { AlignOp } from '../layout/align';

const ALIGNS: { op: AlignOp; label: string }[] = [
  { op: 'left', label: '⇤' }, { op: 'hcenter', label: '↔' }, { op: 'right', label: '⇥' },
  { op: 'bottom', label: '⇩' }, { op: 'vcenter', label: '↕' }, { op: 'top', label: '⇧' },
];

export function Toolbar({ onSave }: { onSave: () => void }) {
  const { state, dispatch, canUndo, canRedo } = useEditor();
  const inked = state.panels.filter((p) => p.ink).map((p) => ({ id: p.id, ink: p.ink! }));
  const collisions = findCollisions(inked);
  const clipped = outOfBounds(inked);
  const n = state.selection.length;

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 8,
                  borderBottom: '1px solid #ddd', fontFamily: 'system-ui', fontSize: 13 }}>
      {ALIGNS.map(({ op, label }) => (
        <button key={op} disabled={n < 2} title={`Align ${op}`}
                onClick={() => dispatch({ type: 'align', op, ref: 'selection' })}>{label}</button>
      ))}
      <button disabled={n < 3} title="Distribute horizontally (equal gaps)"
              onClick={() => dispatch({ type: 'distribute', axis: 'x', mode: 'gaps' })}>⇔</button>
      <button disabled={n < 3} title="Distribute vertically (equal gaps)"
              onClick={() => dispatch({ type: 'distribute', axis: 'y', mode: 'gaps' })}>⇕</button>
      <button disabled={n < 2} title="Match width"
              onClick={() => dispatch({ type: 'matchSize', dim: 'w' })}>=w</button>
      <button disabled={n < 2} title="Match height"
              onClick={() => dispatch({ type: 'matchSize', dim: 'h' })}>=h</button>

      <span style={{ width: 12 }} />
      <label title="Align by the axes box or by the ink box (which includes tick labels)">
        <input type="checkbox" checked={state.boxMode === 'ink'}
               onChange={(e) => dispatch({ type: 'setBoxMode', mode: e.target.checked ? 'ink' : 'axes' })} />
        {' '}ink box
      </label>

      <span style={{ width: 12 }} />
      <button disabled={!canUndo} onClick={() => dispatch({ type: 'undo' })}>undo</button>
      <button disabled={!canRedo} onClick={() => dispatch({ type: 'redo' })}>redo</button>

      <span style={{ flex: 1 }} />
      {(collisions.length > 0 || clipped.length > 0) && (
        <span style={{ color: '#b45309' }}
              title={[...collisions.map((c) => `${c.a} overlaps ${c.b}`),
                      ...clipped.map((id) => `${id} leaves the canvas`)].join('\n')}>
          ⚠ {collisions.length + clipped.length}
        </span>
      )}
      <button onClick={onSave} disabled={!state.dirty}>
        {state.dirty ? 'Save' : 'Saved'}
      </button>
    </div>
  );
}
```

Then wire `App.tsx` as `<EditorProvider>` wrapping `<Toolbar>`, `<Canvas>` and `<Properties>` in a flex row, and have it `getFigure()` on mount and `dispatch({type:'load', ...})`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run` then `npm run build`
Expected: all suites green; build succeeds

- [ ] **Step 5: Manual check, then commit**

Confirm: selecting two panels enables the align buttons and clicking ⇤ left-aligns them; three panels enable distribute; typing `20` into `x` and pressing Enter moves the panel to exactly 20 mm; the ⚠ badge appears when panels overlap and its tooltip names the pair.

```bash
git add web/src/panels/ web/src/App.tsx
git commit -m "feat(web): properties panel, align/distribute toolbar, overlap warnings"
```

---

## Task 11: Save, and serve the UI from one URL

**Files:**
- Modify: `figbuilder/server.py`, `web/src/api.ts`, `web/src/App.tsx`
- Test: `tests/test_figbuilder_server.py` (extend)

**Interfaces:**
- Consumes: `load_figure`/`save_figure` (M0), `useEditor` (Task 7)
- Produces: `PUT /api/figure`, `GET /` serving the built UI, `saveFigure(doc)` in `api.ts`

**Why the `/` route matters:** the researcher opened `http://localhost:8765/` and got `{"detail":"Not Found"}`, because the API server mounts no static files and the UI lives on Vite's port. Two ports where the obvious one 404s is a real usability defect, and it cost a round trip to diagnose.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_figbuilder_server.py
def test_put_figure_persists_panel_rects(client, tmp_path):
    """The browser owns the document; PUT is how it writes back."""
    doc = client.get("/api/figure").json()
    doc["panels"][0]["rect"] = [0.25, 0.25, 0.5, 0.5]
    r = client.put("/api/figure", json=doc)
    assert r.status_code == 200
    assert client.get("/api/figure").json()["panels"][0]["rect"] == [0.25, 0.25, 0.5, 0.5]


def test_put_figure_rejects_a_malformed_document(client):
    r = client.put("/api/figure", json={"panels": [{"id": "a"}]})
    assert r.status_code == 400
    assert "rect" in r.json()["detail"].lower() or "type" in r.json()["detail"].lower()


def test_put_figure_does_not_corrupt_the_file_when_invalid(client):
    before = client.get("/api/figure").json()
    client.put("/api/figure", json={"nonsense": True})
    assert client.get("/api/figure").json() == before


def test_root_explains_where_the_ui_is_when_it_is_not_built(client):
    """A bare 404 at / cost a real debugging round trip. Say something useful."""
    r = client.get("/")
    assert r.status_code in (200, 503)
    body = r.text.lower()
    assert "5173" in body or "npm run dev" in body or "<!doctype html" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_figbuilder_server.py -v -k "put_figure or root_explains"`
Expected: FAIL — 405 Method Not Allowed on PUT; 404 at `/`

- [ ] **Step 3: Write minimal implementation**

In `figbuilder/server.py`, inside `create_app`, add:

```python
    @app.put("/api/figure")
    def put_figure(body: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the browser's document.

        Validated BEFORE writing: a malformed post must not corrupt the file
        the researcher's figure is regenerated from.
        """
        import json as _json
        import tempfile

        from figbuilder.figure import load_figure as _load

        tmp = Path(tempfile.mkstemp(suffix=".json", dir=str(figure_path.parent))[1])
        try:
            tmp.write_text(_json.dumps(body, indent=2) + "\n")
            _load(tmp)                      # raises on a malformed document
        except Exception as e:              # noqa: BLE001 - surfaced to the client
            tmp.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e)) from None
        os.replace(tmp, figure_path)        # atomic; never a torn file
        return {"ok": True, "path": str(figure_path)}
```

and, after all `/api` routes are registered (so it cannot shadow them):

```python
    _DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
    if (_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def _ui_hint() -> str:
            # Do NOT return a bare 404 here. It cost a real debugging round trip.
            return (
                "<h1>figbuilder API</h1>"
                "<p>This port serves <code>/api/*</code> only.</p>"
                "<p>The editor UI is not built. Either run "
                "<code>cd web && npm run dev</code> and open "
                "<a href='http://localhost:5173'>http://localhost:5173</a>, "
                "or run <code>cd web && npm run build</code> to have this "
                "server host it here.</p>"
            )
```

Add `import os`, `from fastapi.responses import HTMLResponse` and `from fastapi.staticfiles import StaticFiles` to the imports.

In `web/src/api.ts`:

```typescript
export async function saveFigure(doc: unknown): Promise<void> {
  const r = await fetch(`${BASE}/api/figure`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(doc),
  });
  if (!r.ok) throw new Error(`save failed: ${r.status} ${await r.text()}`);
}
```

In `App.tsx`, implement `onSave`: rebuild the document by taking the ORIGINAL loaded JSON and replacing each panel's `rect` with the editor's current rect (leaving `data`, `spec`, `groups` and `annotations` untouched so nothing the editor does not manage is lost), `await saveFigure(doc)`, then `dispatch({type:'saved'})`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_figbuilder_*.py -q` (expect 164 + 4 new) and `cd web && npm run build && npx vitest run`
Expected: all green

- [ ] **Step 5: Manual check, then commit**

Confirm: drag a panel, click Save, then `python -m figbuilder export <fig4.json> -o /tmp/check.svg` and see the moved panel in the export. Then `cd web && npm run build` and confirm `http://localhost:8765/` now serves the UI instead of a 404.

```bash
git add figbuilder/server.py web/src/api.ts web/src/App.tsx tests/test_figbuilder_server.py
git commit -m "feat: PUT /api/figure to persist layout, and serve the UI at /"
```

---

## Task 12: Group create / ungroup / gutter editing

**Files:**
- Modify: `web/src/state/editorStore.tsx`, `web/src/panels/Properties.tsx`, `web/src/panels/Toolbar.tsx`
- Test: `web/src/state/editorStore.test.ts` (extend)

**Interfaces:**
- Consumes: `solveGroup`, `groupBounds`, `gutterFromChildren` (Task 4)
- Produces: actions `{type:'groupSelection', axis}`, `{type:'ungroup', id}`, `{type:'setGutter', id, gutterMm}`, `{type:'setGroupEqual', id, equal}`

**Why this closes the loop:** Task 4 built the solver and nothing uses it. Without this task, the six-frame video strip stays six hand-placed rects, which is exactly the tedium the design set out to remove.

- [ ] **Step 1: Write the failing test**

```typescript
// add to web/src/state/editorStore.test.ts
describe('groups', () => {
  it('groups the selection, recovering the gutter from the existing spacing', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(1);
    const g = s.groups[0];
    expect(g.axis).toBe('x');
    expect(g.gutterMm).toBeGreaterThan(0);
    expect(s.panels.filter((p) => p.group === g.id).map((p) => p.id).sort()).toEqual(['a', 'c']);
  });

  it('re-solves children when the gutter changes, and the change is undoable', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const before = s.panels.find((p) => p.id === 'c')!.rect.x;
    s = reducer(s, { type: 'setGutter', id: s.groups[0].id, gutterMm: 20 });
    const after = s.panels.find((p) => p.id === 'c')!.rect.x;
    expect(after).not.toBeCloseTo(before);
    s = reducer(s, { type: 'undo' });
    expect(s.panels.find((p) => p.id === 'c')!.rect.x).toBeCloseTo(before, 9);
  });

  it('equal:true gives every child the same width', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'b', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    s = reducer(s, { type: 'setGroupEqual', id: s.groups[0].id, equal: true });
    const ws = s.panels.map((p) => p.rect.w);
    expect(ws[1]).toBeCloseTo(ws[0], 9);
    expect(ws[2]).toBeCloseTo(ws[0], 9);
  });

  it('ungroup leaves children exactly where the solver last put them', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a', 'c'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    const solved = s.panels.map((p) => ({ ...p.rect }));
    s = reducer(s, { type: 'ungroup', id: s.groups[0].id });
    expect(s.groups).toHaveLength(0);
    s.panels.forEach((p, i) => {
      expect(p.rect.x).toBeCloseTo(solved[i].x, 9);
      expect(p.rect.w).toBeCloseTo(solved[i].w, 9);
    });
    expect(s.panels.every((p) => !p.group)).toBe(true);
  });

  it('refuses to group fewer than two panels', () => {
    let s = loaded();
    s = reducer(s, { type: 'select', ids: ['a'] });
    s = reducer(s, { type: 'groupSelection', axis: 'x' });
    expect(s.groups).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/state/editorStore.test.ts`
Expected: FAIL — the new actions fall through to `default` and change nothing

- [ ] **Step 3: Write minimal implementation**

Extend the `Action` union in `editorStore.tsx`:

```typescript
  | { type: 'groupSelection'; axis: 'x' | 'y' }
  | { type: 'ungroup'; id: string }
  | { type: 'setGutter'; id: string; gutterMm: number }
  | { type: 'setGroupEqual'; id: string; equal: boolean }
```

Add a helper above `reducer`, and the four cases inside it:

```typescript
/** Re-solve one group's children and return the full panel list. */
function resolveGroup(state: EditorState, groupId: string, groups: Group[]): PanelState[] {
  const g = groups.find((x) => x.id === groupId);
  if (!g) return state.panels;
  const kids = state.panels.filter((p) => p.group === groupId);
  if (kids.length === 0) return state.panels;
  // Solve in the children's own left-to-right / top-to-bottom order.
  const axisKey = g.axis;
  const ordered = [...kids].sort((a, b) =>
    axisKey === 'x' ? a.rect.x - b.rect.x : b.rect.y - a.rect.y);
  const solved = solveGroup(g, ordered.map((p) => p.rect), state.figWmm, state.figHmm);
  const byId = new Map(ordered.map((p, i) => [p.id, solved[i]]));
  return state.panels.map((p) => (byId.has(p.id) ? { ...p, rect: byId.get(p.id)! } : p));
}
```

```typescript
    case 'groupSelection': {
      if (state.selection.length < 2) return state;
      const kids = state.panels.filter((p) => state.selection.includes(p.id));
      const id = `grp_${Date.now().toString(36)}`;
      const group: Group = {
        id,
        axis: action.axis,
        gutterMm: gutterFromChildren(
          kids.map((p) => p.rect), action.axis, state.figWmm, state.figHmm),
        equal: false,
        rect: groupBounds(kids.map((p) => p.rect)),
      };
      const groups = [...state.groups, group];
      const tagged = state.panels.map((p) =>
        state.selection.includes(p.id) ? { ...p, group: id } : p);
      const panels = resolveGroup({ ...state, panels: tagged }, id, groups);
      return { ...withGeometry({ ...state, groups }, panels), groups };
    }

    case 'ungroup': {
      // Children keep their solved rects; only the tag and the group go away.
      const panels = state.panels.map((p) => (p.group === action.id ? { ...p, group: null } : p));
      return { ...state, panels, groups: state.groups.filter((g) => g.id !== action.id), dirty: true };
    }

    case 'setGutter': {
      const groups = state.groups.map((g) =>
        (g.id === action.id ? { ...g, gutterMm: action.gutterMm } : g));
      const panels = resolveGroup({ ...state, groups }, action.id, groups);
      return { ...withGeometry({ ...state, groups }, panels), groups };
    }

    case 'setGroupEqual': {
      const groups = state.groups.map((g) =>
        (g.id === action.id ? { ...g, equal: action.equal } : g));
      const panels = resolveGroup({ ...state, groups }, action.id, groups);
      return { ...withGeometry({ ...state, groups }, panels), groups };
    }
```

Import `solveGroup`, `groupBounds`, `gutterFromChildren` and the `Group` type at the top.

In `Toolbar.tsx` add two buttons: "group ⇥⇤" (`disabled={n < 2}`, dispatches `groupSelection` with `axis:'x'`) and "ungroup" (enabled when every selected panel shares one non-null `group`, dispatching `ungroup` with that id).

In `Properties.tsx`, when the selected panel belongs to a group, render the group's gutter as an mm field (dispatching `setGutter`) and an "equal sizes" checkbox (dispatching `setGroupEqual`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run` then `npm run build`
Expected: all suites green (16 store tests); build succeeds

- [ ] **Step 5: Manual check, then commit**

Confirm on the real `fig4.json`: select the four `video_*` panels, click group, then change the gutter from 2 mm to 6 mm and watch all four re-space together; undo returns them in one step; ungroup leaves them exactly where they were.

```bash
git add web/src/state/editorStore.tsx web/src/state/editorStore.test.ts web/src/panels/
git commit -m "feat(web): create/ungroup panel groups and edit their gutter"
```

---

## Deferred from M2 (deliberate, with reasons)

| Item | Why deferred |
|---|---|
| **Draggable ruler guides** | `buildTargets` already accepts a `guides: SnapTarget[]` array and `<Guides>` already renders lines, so the snapping half is done. What is missing is only the ruler UI to create and drag them. Panels snap to each other, the grid and the margins without it, which covers the common case. |
| **Annotation editing** (panel letters, arrows, brackets) | M3. The emitters exist on both sides and are conformance-tested; only the authoring UI is missing. Letters are still seeded programmatically by `seed_fig4_layout`, so the export is complete without it. |
| **Panel add/delete and data rebinding** | M4. `/api/bundle` and `/api/panel-types` already expose what that UI will need. |
| **Tile cache keyed on panel code version** | Known residual from the M0/M1 review: editing `figbuilder/panels/*.py` and re-rendering THROUGH THE SERVER can serve a stale tile. The CLI `export` path defaults to `--cache-dir=None` and is unaffected, so the figure-regeneration path is safe. It will matter once panel code is edited during a live session; fix by folding a hash of the panel module's source into `tile_cache_key`. |

## Testing

- Layout math (`web/src/layout/*`): pure vitest, no React, no DOM. This is where correctness lives.
- Store (`web/src/state/editorStore.test.ts`): the reducer is pure and is tested without rendering.
- Canvas/handles/panels: covered indirectly by their pure helpers (`hitTest`, `resize`, `format`) plus the manual checks each task names. No DOM-testing library is added — that would be a new dev dependency.
- Python: `tests/test_figbuilder_server.py` gains the `PUT /api/figure` and `/` cases.
- Every task ends with BOTH suites green: `python -m pytest tests/test_figbuilder_*.py -q` and `cd web && npx vitest run`.

## Risks

| Risk | Mitigation |
|---|---|
| The y-flip gets applied twice and everything is upside-down | Exactly one flip, in `rectToScreen`, with a test asserting a bottom-anchored rect lands at the BOTTOM of the canvas. No other module may flip. |
| Resize shows stretched tick labels | Resize re-requests the tile on pointer-UP; the scaled tile during the drag is an explicit preview, not the result. |
| A drag costs dozens of undos | `commit()` coalesces on a key; a test drives 20 moves and asserts ONE undo restores the start. |
| Ink-box alignment misbehaves on polar panels | Measured: a polar ink box can be NARROWER than its rect. Nothing may assume ink ⊇ axes; `findCollisions` and the align path use whatever rects the caller resolved. |
| A bad PUT corrupts the figure the paper is regenerated from | The server validates into a temp file via `load_figure` BEFORE `os.replace`; a test asserts an invalid PUT leaves the file untouched. |
