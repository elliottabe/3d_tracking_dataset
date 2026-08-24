/**
 * Annotation layer: objects -> SVG elements.
 *
 * This is the TypeScript half of a deliberately duplicated emitter; the
 * Python half (`figbuilder/annot.py`) drives headless export. A conformance
 * test (tests/test_figbuilder_annot_conformance.py) asserts the two agree.
 * Duplication is preferred over making headless export depend on a browser.
 *
 * Positions are in millimetres. Figure space has y growing UPWARD from the
 * bottom-left (matching matplotlib); SVG has y growing downward, so every y
 * is flipped here exactly once, inside `Frame`.
 *
 * This module is a direct port of `figbuilder/annot.py` — mirror it
 * element-for-element and attribute-for-attribute, including quirks of its
 * lxml-based serialization (each annotation element is serialized on its
 * own, so it carries its own `ns0:`/`ns1:` namespace prefixes and `xmlns:`
 * declarations rather than sharing a document-level namespace map). Do not
 * "clean up" that shape — the conformance test locks it in byte-for-byte
 * (after whitespace/decimal normalization).
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const XLINK_NS = 'http://www.w3.org/1999/xlink';
const ARROW_MARKER_ID = 'fb_arrowhead';

export const PT_PER_MM = 72 / 25.4;

/**
 * Mirrors figbuilder.annot._num EXACTLY. Six decimals, trailing zeros trimmed.
 * Do not substitute template interpolation of a raw number — that prints
 * JS full precision and diverges from Python. See Ruling 3 in the SDD ledger.
 */
export function num(v: number): string {
  const out = v.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  // Mirrors the Python guard: "-0" and "" both collapse to "0".
  return ['', '-', '0', '-0'].includes(out) ? '0' : out;
}

export const ANNOTATION_KINDS = [
  'text', 'line', 'arrow', 'leader', 'rect', 'ellipse', 'bracket',
  'scalebar', 'image',
] as const;
export type Kind = (typeof ANNOTATION_KINDS)[number];

export interface PanelRef { id: string; rect: [number, number, number, number] }
export interface FigureSpec {
  width_mm: number; height_mm: number; panels: PanelRef[];
}
export interface Annotation {
  id?: string; kind: Kind; parent?: string;
  pos_mm?: [number, number]; to_mm?: [number, number];
  size_mm?: [number, number]; length_mm?: number; tick_mm?: number;
  text?: string; label?: string; href?: string;
  style?: Record<string, string | number>;
}

/** Mirrors figbuilder.annot._Frame exactly. */
class Frame {
  private hMm: number;
  private oxMm: number;
  private oyMm: number;

  constructor(hMm: number, oxMm = 0, oyMm = 0) {
    this.hMm = hMm;
    this.oxMm = oxMm;
    this.oyMm = oyMm;
  }
  static of(fig: FigureSpec, parent?: string | null): Frame {
    if (!parent) return new Frame(fig.height_mm);
    const p = fig.panels.find((q) => q.id === parent);
    if (!p) throw new Error(`unknown parent panel ${parent}`);
    return new Frame(fig.height_mm,
      p.rect[0] * fig.width_mm, p.rect[1] * fig.height_mm);
  }
  x(mm: number) { return (this.oxMm + mm) * PT_PER_MM; }
  y(mm: number) { return (this.hMm - (this.oyMm + mm)) * PT_PER_MM; }
  d(mm: number) { return mm * PT_PER_MM; }
}

/** Mirrors figbuilder.annot._style — same key order, same output. */
function styleStr(d: Record<string, string | number>): string {
  const out: string[] = [];
  if ('color' in d) out.push(`fill:${d.color}`);
  if ('stroke' in d) out.push(`stroke:${d.stroke}`);
  if ('lw_pt' in d) out.push(`stroke-width:${num(Number(d.lw_pt))}`);
  if ('font_size_pt' in d) out.push(`font-size:${num(Number(d.font_size_pt))}px`);
  if (d.weight === 'bold' || d.weight === 700) out.push('font-weight:700');
  if ('font' in d) out.push(`font-family:${d.font}`);
  if ('opacity' in d) out.push(`opacity:${num(Number(d.opacity))}`);
  return out.join(';');
}

function pt2(a: [number, number] | undefined, def: [number, number] = [0, 0]) {
  return a ?? def;
}

// --- A tiny element model mirroring lxml.etree.Element, serialized the same
// way `etree.tostring(e, encoding="unicode")` renders a *detached* element
// (Task 8's Python side calls tostring() per-element, never on a shared
// document root, so every top-level annotation element re-declares its own
// namespace prefixes). ---

interface XAttr { name: string; value: string; xlink?: boolean }
interface XElem { tag: string; attrs: XAttr[]; children?: XElem[]; text?: string }

function escapeText(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAttr(s: string): string {
  return escapeText(s).replace(/"/g, '&quot;');
}

function usesXlink(el: XElem): boolean {
  if (el.attrs.some((a) => a.xlink)) return true;
  return (el.children ?? []).some(usesXlink);
}

function serialize(el: XElem, isRoot: boolean): string {
  const parts = [`ns0:${el.tag}`];
  if (isRoot) {
    parts.push(`xmlns:ns0="${SVG_NS}"`);
    if (usesXlink(el)) parts.push(`xmlns:ns1="${XLINK_NS}"`);
  }
  for (const a of el.attrs) {
    parts.push(`${a.xlink ? 'ns1:' : ''}${a.name}="${escapeAttr(a.value)}"`);
  }
  const open = `<${parts.join(' ')}`;
  const hasContent = (el.children && el.children.length > 0) || el.text !== undefined;
  if (!hasContent) return `${open}/>`;
  const inner = el.text !== undefined
    ? escapeText(el.text)
    : (el.children ?? []).map((c) => serialize(c, false)).join('');
  return `${open}>${inner}</ns0:${el.tag}>`;
}

export function emitAnnotations(fig: FigureSpec, anns: Annotation[]): string {
  const out: XElem[] = [];
  let needsMarker = false;

  for (const a of anns) {
    if (!(ANNOTATION_KINDS as readonly string[]).includes(a.kind)) {
      throw new Error(
        `unknown annotation kind ${JSON.stringify(a.kind)}; expected one of ` +
        `${[...ANNOTATION_KINDS].sort().join(', ')}`);
    }
    const f = Frame.of(fig, a.parent);
    const st: Record<string, string | number> = { ...(a.style ?? {}) };
    const [px, py] = pt2(a.pos_mm);
    let el: XElem;

    switch (a.kind) {
      case 'text': {
        if (!('font' in st)) st.font = 'Arial, Helvetica, sans-serif';
        el = {
          tag: 'text',
          attrs: [
            { name: 'x', value: num(f.x(px)) },
            { name: 'y', value: num(f.y(py)) },
            { name: 'style', value: styleStr(st) },
          ],
          text: String(a.text ?? ''),
        };
        break;
      }

      case 'line':
      case 'leader': {
        const [tx, ty] = pt2(a.to_mm);
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        el = {
          tag: 'line',
          attrs: [
            { name: 'x1', value: num(f.x(px)) },
            { name: 'y1', value: num(f.y(py)) },
            { name: 'x2', value: num(f.x(tx)) },
            { name: 'y2', value: num(f.y(ty)) },
            { name: 'style', value: styleStr(st) },
          ],
        };
        break;
      }

      case 'arrow': {
        const [tx, ty] = pt2(a.to_mm);
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        needsMarker = true;
        el = {
          tag: 'path',
          attrs: [
            {
              name: 'd',
              value: `M ${num(f.x(px))},${num(f.y(py))} L ${num(f.x(tx))},${num(f.y(ty))}`,
            },
            {
              name: 'style',
              value: `${styleStr(st)};fill:none;marker-end:url(#${ARROW_MARKER_ID})`,
            },
          ],
        };
        break;
      }

      case 'rect': {
        const [w, h] = pt2(a.size_mm, [1, 1]);
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        const style = 'color' in st ? styleStr(st) : `${styleStr(st)};fill:none`;
        el = {
          tag: 'rect',
          attrs: [
            { name: 'x', value: num(f.x(px)) },
            { name: 'y', value: num(f.y(py + h)) },
            { name: 'width', value: num(f.d(w)) },
            { name: 'height', value: num(f.d(h)) },
            { name: 'style', value: style },
          ],
        };
        break;
      }

      case 'ellipse': {
        const [w, h] = pt2(a.size_mm, [1, 1]);
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        const style = 'color' in st ? styleStr(st) : `${styleStr(st)};fill:none`;
        el = {
          tag: 'ellipse',
          attrs: [
            { name: 'cx', value: num(f.x(px + w / 2)) },
            { name: 'cy', value: num(f.y(py + h / 2)) },
            { name: 'rx', value: num(f.d(w / 2)) },
            { name: 'ry', value: num(f.d(h / 2)) },
            { name: 'style', value: style },
          ],
        };
        break;
      }

      case 'bracket': {
        const [tx, ty] = pt2(a.to_mm);
        const tick = f.d(a.tick_mm ?? 1.0);
        const x1 = f.x(px);
        const y1 = f.y(py);
        const x2 = f.x(tx);
        const y2 = f.y(ty);
        if (!('stroke' in st)) st.stroke = '#000000';
        if (!('lw_pt' in st)) st.lw_pt = 0.8;
        el = {
          tag: 'path',
          attrs: [
            {
              name: 'd',
              value: `M ${num(x1)},${num(y1 + tick)} L ${num(x1)},${num(y1)} ` +
                `L ${num(x2)},${num(y2)} L ${num(x2)},${num(y2 + tick)}`,
            },
            { name: 'style', value: `${styleStr(st)};fill:none` },
          ],
        };
        break;
      }

      case 'scalebar': {
        const length = a.length_mm ?? 5.0;
        const lineEl: XElem = {
          tag: 'line',
          attrs: [
            { name: 'x1', value: num(f.x(px)) },
            { name: 'y1', value: num(f.y(py)) },
            { name: 'x2', value: num(f.x(px + length)) },
            { name: 'y2', value: num(f.y(py)) },
            {
              name: 'style',
              value: styleStr({
                stroke: (st.stroke as string) ?? '#000000',
                lw_pt: (st.lw_pt as number) ?? 1.2,
              }),
            },
          ],
        };
        const textEl: XElem = {
          tag: 'text',
          attrs: [
            { name: 'x', value: num(f.x(px + length / 2)) },
            { name: 'y', value: num(f.y(py) + f.d(2.0)) },
            {
              name: 'style',
              value: styleStr({
                font_size_pt: (st.font_size_pt as number) ?? 6,
                font: 'Arial, Helvetica, sans-serif',
              }) + ';text-anchor:middle',
            },
          ],
          text: String(a.label ?? ''),
        };
        el = { tag: 'g', attrs: [], children: [lineEl, textEl] };
        break;
      }

      case 'image': {
        const [w, h] = pt2(a.size_mm, [10, 10]);
        el = {
          tag: 'image',
          attrs: [
            { name: 'x', value: num(f.x(px)) },
            { name: 'y', value: num(f.y(py + h)) },
            { name: 'width', value: num(f.d(w)) },
            { name: 'height', value: num(f.d(h)) },
            { name: 'href', value: a.href ?? '', xlink: true },
          ],
        };
        break;
      }

      default:
        throw new Error(`kind ${a.kind} not yet ported`);
    }

    if (a.id) el.attrs.push({ name: 'id', value: a.id });
    out.push(el);
  }

  if (needsMarker) {
    out.unshift({
      tag: 'defs',
      attrs: [],
      children: [{
        tag: 'marker',
        attrs: [
          { name: 'id', value: ARROW_MARKER_ID },
          { name: 'viewBox', value: '0 0 10 10' },
          { name: 'refX', value: '9' },
          { name: 'refY', value: '5' },
          { name: 'markerWidth', value: '5' },
          { name: 'markerHeight', value: '5' },
          { name: 'orient', value: 'auto-start-reverse' },
        ],
        children: [{ tag: 'path', attrs: [{ name: 'd', value: 'M 0,0 L 10,5 L 0,10 z' }] }],
      }],
    });
  }

  return out.map((e) => serialize(e, true)).join('');
}
