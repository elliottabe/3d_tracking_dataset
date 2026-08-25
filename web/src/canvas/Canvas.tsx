import { useEffect, useRef, useState } from 'react';
import { getFigure, renderPanel, type FigureDoc, type PanelSpecDTO, type TileDTO } from '../api';
import { namespaceIdsAndExtractInner } from './namespaceIds';
import { hitTestPanels, marqueeHits, screenToFrac, type Hittable } from './hitTest';
import { Guides } from './Guides';
import { Handles } from './Handles';
import { resizeRect, MIN_SIZE_MM, type HandleId } from './resize';
import { useEditor } from '../state/editorStore';
import { normalizeRect, rectToScreen, PT_PER_MM, type Rect } from '../layout/rect';
import { buildTargets, screenTolToFrac, snapRect, type SnapTarget } from '../layout/snap';
import { groupsFromDoc, panelsFromDoc } from '../layout/load';

/** Screen-pixel radius within which a dragged edge/centre snaps to a target. */
const SNAP_TOL_PX = 6;

/**
 * Screen-pixel movement budget below which a background pointer-down/up is
 * treated as a CLICK (clears selection) rather than a PAN (leaves it alone).
 * Real pointers jitter even when the user means to click, so this must be a
 * few px, not zero.
 */
const CLICK_TOL_PX = 4;

type PointerMode = 'pan' | 'drag' | 'marquee' | 'resize' | null;

type DragState = {
  startFrac: { x: number; y: number };
  /** The box (rect or ink, per active box mode) the drag started from. */
  startBox: Rect;
  /** Total snapped delta already dispatched this gesture, so each pointer
   *  move only sends the INCREMENT — moveSelection deltas accumulate. */
  appliedDelta: { dx: number; dy: number };
  coalesceKey: string;
};

type ResizeState = {
  id: string;
  handle: HandleId;
  startFrac: { x: number; y: number };
  /** The panel's rect at gesture start — resize always acts on the real
   *  axes rect, never the ink box (unlike drag, which can snap to ink). */
  startBox: Rect;
  coalesceKey: string;
};

const tupleToRect = ([x, y, w, h]: [number, number, number, number]): Rect => ({ x, y, w, h });
const rectToTuple = (r: Rect): [number, number, number, number] => [r.x, r.y, r.w, r.h];

/** Nudge step per arrow-key press, in mm; shift multiplies it up. */
const NUDGE_MM = 0.25;
const NUDGE_SHIFT_MM = 1;

export function Canvas() {
  const { state, dispatch } = useEditor();
  const [doc, setDoc] = useState<FigureDoc | null>(null);
  const [tiles, setTiles] = useState<Record<string, TileDTO>>({});
  /** Figure-space rect each cached tile's SVG was actually rendered for —
   *  the anchor a live rect is compared against to derive the on-screen
   *  translate/scale preview. Starts as the server rect and is replaced
   *  whenever a fresh render lands (initial load, or a resize release). */
  const [renderedRect, setRenderedRect] = useState<Record<string, Rect>>({});
  const [view, setView] = useState({ scale: 2, tx: 0, ty: 0 });
  const [panning, setPanning] = useState(false);
  const [marquee, setMarquee] = useState<Rect | null>(null);
  const [guides, setGuides] = useState<SnapTarget[]>([]);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const modeRef = useRef<PointerMode>(null);
  const panRef = useRef<{ x: number; y: number } | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const resizeRef = useRef<ResizeState | null>(null);
  const marqueeStartRef = useRef<{ x: number; y: number } | null>(null);
  /** Whether NO selection-affecting modifier was held when the current pan
   *  gesture began — a pan that started this way and moved negligibly is a
   *  plain click on empty space, which should clear the selection. */
  const panNoModifierRef = useRef(false);
  /** Cumulative screen-px distance travelled since pointer-down, for the
   *  same click-vs-pan decision (path length, not net displacement, so a
   *  jittery round-trip still counts as movement). */
  const panMoveDistRef = useRef(0);

  useEffect(() => {
    void getFigure().then((d) => {
      setDoc(d);
      dispatch({
        type: 'load',
        figWmm: d.figure.width_mm,
        figHmm: d.figure.height_mm,
        panels: panelsFromDoc(d),
        groups: groupsFromDoc(d),
      });
    });
  }, [dispatch]);

  useEffect(() => {
    if (!doc) return;
    let alive = true;
    void (async () => {
      for (const p of doc.panels) {
        const t = await renderPanel(p);
        if (!alive) return;
        setTiles((prev) => ({ ...prev, [p.id]: t }));
        setRenderedRect((prev) => ({ ...prev, [p.id]: tupleToRect(p.rect) }));
        dispatch({ type: 'setInk', id: p.id, ink: tupleToRect(t.ink_box) });
      }
    })();
    return () => { alive = false; };
  }, [doc, dispatch]);

  if (!doc) return <p>loading…</p>;
  const figWpt = doc.figure.width_mm * PT_PER_MM;
  const figHpt = doc.figure.height_mm * PT_PER_MM;

  const onPointerDown = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return;
    const svgRect = svg.getBoundingClientRect();
    const pt = screenToFrac(e.clientX, e.clientY, svgRect, figWpt, figHpt, view.scale);
    const hittables: Hittable[] = state.panels.map((p) => ({ id: p.id, rect: p.rect }));
    const hitId = hitTestPanels(hittables, pt);

    if (hitId) {
      const panel = state.panels.find((p) => p.id === hitId)!;
      const box = state.boxMode === 'ink' ? (panel.ink ?? panel.rect) : panel.rect;
      modeRef.current = 'drag';
      dragRef.current = {
        startFrac: pt,
        startBox: box,
        appliedDelta: { dx: 0, dy: 0 },
        // Fresh per gesture: a whole drag must coalesce into ONE undo
        // entry, but two separate drags must stay two entries.
        coalesceKey: `drag-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      };
      dispatch({ type: 'select', ids: [hitId], additive: e.shiftKey });
      return;
    }

    // Shift here is a PAN-ESCAPE gesture modifier: plain background drag
    // pans the canvas, so shift is what tells us the user wants to marquee
    // instead. It is NOT the additive-selection modifier (that's meta/ctrl,
    // checked at release below) — do not "fix" this by making shift additive.
    if (e.shiftKey) {
      modeRef.current = 'marquee';
      marqueeStartRef.current = pt;
      setMarquee({ x: pt.x, y: pt.y, w: 0, h: 0 });
      return;
    }

    modeRef.current = 'pan';
    panRef.current = { x: e.clientX, y: e.clientY };
    panNoModifierRef.current = !e.shiftKey && !e.metaKey && !e.ctrlKey;
    panMoveDistRef.current = 0;
    setPanning(true);
  };

  /** Grabbing a resize handle starts a gesture on the SELECTED panel's rect
   *  (never the ink box — resize edits the real axes box). */
  const onGrabHandle = (handle: HandleId, e: React.PointerEvent) => {
    const id = state.selection[0];
    const panel = state.panels.find((p) => p.id === id);
    const svg = svgRef.current;
    if (!panel || !svg) return;
    const svgRect = svg.getBoundingClientRect();
    const pt = screenToFrac(e.clientX, e.clientY, svgRect, figWpt, figHpt, view.scale);
    modeRef.current = 'resize';
    resizeRef.current = {
      id: panel.id,
      handle,
      startFrac: pt,
      startBox: panel.rect,
      // Fresh per gesture, same convention as drag's coalesceKey: a whole
      // resize must land as ONE undo entry, but two resizes must not merge.
      coalesceKey: `resize-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };
  };

  /** Re-request the true tile for one panel after a resize release — a
   *  CSS/SVG-stretched preview during the drag is fine (geometry only), but
   *  matplotlib tick/label text does not scale with the axes box, so the
   *  settled size must come back from the server to be trustworthy. */
  const refreshPanel = (id: string, rect: Rect) => {
    if (!doc) return;
    const origPanel = doc.panels.find((p) => p.id === id);
    if (!origPanel) return;
    const panelDTO: PanelSpecDTO = { ...origPanel, rect: rectToTuple(rect) };
    void renderPanel(panelDTO).then((t) => {
      setTiles((prev) => ({ ...prev, [id]: t }));
      setRenderedRect((prev) => ({ ...prev, [id]: rect }));
      dispatch({ type: 'setInk', id, ink: tupleToRect(t.ink_box) });
    });
  };

  const endGesture = (e: React.PointerEvent) => {
    if (modeRef.current === 'marquee' && marquee) {
      const hittables: Hittable[] = state.panels.map((p) => ({ id: p.id, rect: p.rect }));
      // Marquee normally REPLACES the selection. Only when meta/ctrl is ALSO
      // held at release does it extend the existing selection instead.
      dispatch({
        type: 'select',
        ids: marqueeHits(hittables, marquee),
        additive: e.metaKey || e.ctrlKey,
      });
    } else if (modeRef.current === 'pan' && panNoModifierRef.current
      && panMoveDistRef.current <= CLICK_TOL_PX) {
      // Pointer went down and up on empty space, unmodified, with
      // negligible movement: a click, not a pan — clear the selection.
      dispatch({ type: 'select', ids: [] });
    } else if (modeRef.current === 'resize' && resizeRef.current) {
      const { id } = resizeRef.current;
      const panel = state.panels.find((p) => p.id === id);
      if (panel) refreshPanel(id, panel.rect);
    }
    modeRef.current = null;
    panRef.current = null;
    dragRef.current = null;
    resizeRef.current = null;
    marqueeStartRef.current = null;
    panNoModifierRef.current = false;
    panMoveDistRef.current = 0;
    setPanning(false);
    setMarquee(null);
    setGuides([]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (modeRef.current === 'pan') {
      if (!panRef.current) return;
      const dx = e.clientX - panRef.current.x;
      const dy = e.clientY - panRef.current.y;
      panRef.current = { x: e.clientX, y: e.clientY };
      panMoveDistRef.current += Math.hypot(dx, dy);
      setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      return;
    }

    if (modeRef.current === 'marquee') {
      const start = marqueeStartRef.current;
      const svg = svgRef.current;
      if (!start || !svg) return;
      const svgRect = svg.getBoundingClientRect();
      const pt = screenToFrac(e.clientX, e.clientY, svgRect, figWpt, figHpt, view.scale);
      setMarquee({ x: start.x, y: start.y, w: pt.x - start.x, h: pt.y - start.y });
      return;
    }

    if (modeRef.current === 'drag') {
      const d = dragRef.current;
      const svg = svgRef.current;
      if (!d || !svg) return;
      const svgRect = svg.getBoundingClientRect();
      const pt = screenToFrac(e.clientX, e.clientY, svgRect, figWpt, figHpt, view.scale);

      const rawDx = pt.x - d.startFrac.x;
      const rawDy = pt.y - d.startFrac.y;
      const candidate: Rect = { ...d.startBox, x: d.startBox.x + rawDx, y: d.startBox.y + rawDy };

      const selected = new Set(state.selection);
      const others = state.panels
        .filter((p) => !selected.has(p.id))
        .map((p) => (state.boxMode === 'ink' ? (p.ink ?? p.rect) : p.rect));
      const targets = buildTargets({
        others, gridMm: state.gridMm, figWmm: state.figWmm, figHmm: state.figHmm, guides: state.guides,
      });
      // Tolerance is derived from the figure WIDTH for both axes (a
      // deliberate prior ruling) — the effective on-screen radius along y
      // ends up slightly larger whenever height != width. Accepted feel.
      const tolFrac = screenTolToFrac(SNAP_TOL_PX, figWpt, view.scale);
      const { rect: snapped, guides: hitGuides } = snapRect(candidate, targets, tolFrac);

      const totalDx = snapped.x - d.startBox.x;
      const totalDy = snapped.y - d.startBox.y;
      const stepDx = totalDx - d.appliedDelta.dx;
      const stepDy = totalDy - d.appliedDelta.dy;
      d.appliedDelta = { dx: totalDx, dy: totalDy };

      if (stepDx !== 0 || stepDy !== 0) {
        dispatch({ type: 'moveSelection', dx: stepDx, dy: stepDy, coalesceKey: d.coalesceKey });
      }
      setGuides(hitGuides);
      return;
    }

    if (modeRef.current === 'resize') {
      const r = resizeRef.current;
      const svg = svgRef.current;
      if (!r || !svg) return;
      const svgRect = svg.getBoundingClientRect();
      const pt = screenToFrac(e.clientX, e.clientY, svgRect, figWpt, figHpt, view.scale);
      const dx = pt.x - r.startFrac.x;
      const dy = pt.y - r.startFrac.y;
      const next = resizeRect(r.startBox, r.handle, dx, dy, {
        minW: MIN_SIZE_MM / state.figWmm,
        minH: MIN_SIZE_MM / state.figHmm,
        aspect: e.shiftKey,
      });
      // setRect only updates panel.rect (geometry + undo history); it does
      // NOT re-render — the tile stays the stale SVG, stretched to the new
      // box on screen (see the panel-tile transform below). The true render
      // is re-requested on release, in endGesture.
      dispatch({ type: 'setRect', id: r.id, rect: next, coalesceKey: r.coalesceKey });
    }
  };

  /** Arrow keys nudge the selection in mm (bigger step with shift);
   *  ctrl/cmd+z undoes, ctrl/cmd+shift+z redoes; escape clears selection. */
  const onKeyDown = (e: React.KeyboardEvent) => {
    // Guard against hijacking text entry. Today Toolbar/Properties are
    // SIBLINGS of Canvas in App.tsx, so their inputs never bubble a
    // keydown here — this guard is currently redundant. It stays in place
    // so that if an in-canvas text field (e.g. an inline panel-rename box)
    // is ever added, arrow keys and other shortcuts here don't silently
    // steal its caret/typing.
    const target = e.target as HTMLElement | null;
    const tag = target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) {
      return;
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      dispatch({ type: e.shiftKey ? 'redo' : 'undo' });
      return;
    }
    if (e.key === 'Escape') {
      dispatch({ type: 'select', ids: [] });
      return;
    }
    const step = e.shiftKey ? NUDGE_SHIFT_MM : NUDGE_MM;
    switch (e.key) {
      case 'ArrowLeft': e.preventDefault(); dispatch({ type: 'nudge', dxMm: -step, dyMm: 0 }); break;
      case 'ArrowRight': e.preventDefault(); dispatch({ type: 'nudge', dxMm: step, dyMm: 0 }); break;
      // Figure fractions are y-UP, so "up" on screen is +y.
      case 'ArrowUp': e.preventDefault(); dispatch({ type: 'nudge', dxMm: 0, dyMm: step }); break;
      case 'ArrowDown': e.preventDefault(); dispatch({ type: 'nudge', dxMm: 0, dyMm: -step }); break;
      default: break;
    }
  };

  const selectedPanel = state.selection.length === 1
    ? (state.panels.find((p) => p.id === state.selection[0]) ?? null)
    : null;

  return (
    <div
      tabIndex={0}
      style={{
        // Fills the remaining viewport height beneath the Toolbar: the app
        // shell (App.tsx) is a flex column with this component's flex
        // container ancestor set to `flex: 1, minHeight: 0`, so `100%`
        // here resolves to "whatever is left", not a full 100vh on top of
        // the toolbar's own height. Do not hardcode a viewport unit here.
        overflow: 'hidden', width: '100%', height: '100%',
        background: '#f4f4f5', cursor: panning ? 'grabbing' : 'grab',
        outline: 'none',
      }}
      onWheel={(e) => {
        e.preventDefault();
        setView((v) => ({
          ...v,
          scale: Math.min(12, Math.max(0.25, v.scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1))),
        }));
      }}
      onPointerDown={onPointerDown}
      onPointerUp={endGesture}
      onPointerLeave={endGesture}
      onPointerMove={onPointerMove}
      onKeyDown={onKeyDown}
    >
      <svg
        ref={svgRef}
        width={figWpt * view.scale} height={figHpt * view.scale}
        viewBox={`0 0 ${figWpt} ${figHpt}`}
        style={{
          transform: `translate(${view.tx}px, ${view.ty}px)`,
          background: 'white', boxShadow: '0 1px 8px rgba(0,0,0,.15)',
        }}
      >
        {doc.panels.map((p) => {
          const t = tiles[p.id];
          if (!t) return null;
          // `orig` is the rect this cached tile's SVG was actually rendered
          // for (server anchor); `live` is the current committed rect. They
          // differ during a move (translate only) or a resize preview
          // (translate + SCALE — the tile hasn't been re-rendered yet, so a
          // size change must be faked visually until release swaps it out).
          const orig = renderedRect[p.id] ?? tupleToRect(p.rect);
          const live = state.panels.find((x) => x.id === p.id)?.rect ?? orig;
          const origScreen = rectToScreen(orig, figWpt, figHpt);
          const liveScreen = rectToScreen(live, figWpt, figHpt);
          const sx = origScreen.w === 0 ? 1 : liveScreen.w / origScreen.w;
          const sy = origScreen.h === 0 ? 1 : liveScreen.h / origScreen.h;
          const transform =
            `translate(${liveScreen.x} ${liveScreen.y}) scale(${sx} ${sy}) `
            + `translate(${-origScreen.x} ${-origScreen.y})`;
          return (
            <g key={p.id} transform={transform}>
              <g
                id={`panel_${p.id}`}
                dangerouslySetInnerHTML={{ __html: namespaceIdsAndExtractInner(t.svg, p.id) }}
              />
            </g>
          );
        })}

        {state.selection.map((id) => {
          const p = state.panels.find((x) => x.id === id);
          if (!p) return null;
          const s = rectToScreen(p.rect, figWpt, figHpt);
          return (
            <rect
              key={id} x={s.x} y={s.y} width={s.w} height={s.h}
              fill="none" stroke="#2563eb" strokeWidth={2}
              vectorEffect="non-scaling-stroke" pointerEvents="none"
            />
          );
        })}

        {selectedPanel && (
          <Handles box={rectToScreen(selectedPanel.rect, figWpt, figHpt)} onGrab={onGrabHandle} />
        )}

        <Guides guides={guides} figWpt={figWpt} figHpt={figHpt} />

        {marquee && (() => {
          const m = normalizeRect(marquee);
          const s = rectToScreen(m, figWpt, figHpt);
          return (
            <rect
              x={s.x} y={s.y} width={s.w} height={s.h}
              fill="rgba(37,99,235,0.08)" stroke="#2563eb" strokeWidth={1}
              strokeDasharray="4 2" vectorEffect="non-scaling-stroke" pointerEvents="none"
            />
          );
        })()}
      </svg>
    </div>
  );
}
