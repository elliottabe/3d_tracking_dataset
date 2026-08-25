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
