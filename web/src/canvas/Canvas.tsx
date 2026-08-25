import { useEffect, useRef, useState } from 'react';
import { getFigure, renderPanel, type FigureDoc, type TileDTO } from '../api';
import { namespaceIdsAndExtractInner } from './namespaceIds';
import { hitTestPanels, marqueeHits, screenToFrac, type Hittable } from './hitTest';
import { Guides } from './Guides';
import { useEditor } from '../state/editorStore';
import { normalizeRect, rectToScreen, PT_PER_MM, type Rect } from '../layout/rect';
import { buildTargets, screenTolToFrac, snapRect, type SnapTarget } from '../layout/snap';
import type { Group } from '../layout/groups';

/** Screen-pixel radius within which a dragged edge/centre snaps to a target. */
const SNAP_TOL_PX = 6;

type PointerMode = 'pan' | 'drag' | 'marquee' | null;

type DragState = {
  startFrac: { x: number; y: number };
  /** The box (rect or ink, per active box mode) the drag started from. */
  startBox: Rect;
  /** Total snapped delta already dispatched this gesture, so each pointer
   *  move only sends the INCREMENT — moveSelection deltas accumulate. */
  appliedDelta: { dx: number; dy: number };
  coalesceKey: string;
};

const tupleToRect = ([x, y, w, h]: [number, number, number, number]): Rect => ({ x, y, w, h });

export function Canvas() {
  const { state, dispatch } = useEditor();
  const [doc, setDoc] = useState<FigureDoc | null>(null);
  const [tiles, setTiles] = useState<Record<string, TileDTO>>({});
  const [view, setView] = useState({ scale: 2, tx: 0, ty: 0 });
  const [panning, setPanning] = useState(false);
  const [marquee, setMarquee] = useState<Rect | null>(null);
  const [guides, setGuides] = useState<SnapTarget[]>([]);

  const svgRef = useRef<SVGSVGElement | null>(null);
  const modeRef = useRef<PointerMode>(null);
  const panRef = useRef<{ x: number; y: number } | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const marqueeStartRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    void getFigure().then((d) => {
      setDoc(d);
      dispatch({
        type: 'load',
        figWmm: d.figure.width_mm,
        figHmm: d.figure.height_mm,
        panels: d.panels.map((p) => ({
          id: p.id, type: p.type, rect: tupleToRect(p.rect),
          group: p.group ?? null, spec: p.spec, data: p.data,
        })),
        groups: d.groups as unknown as Group[],
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

    if (e.shiftKey) {
      modeRef.current = 'marquee';
      marqueeStartRef.current = pt;
      setMarquee({ x: pt.x, y: pt.y, w: 0, h: 0 });
      return;
    }

    modeRef.current = 'pan';
    panRef.current = { x: e.clientX, y: e.clientY };
    setPanning(true);
  };

  const endGesture = () => {
    if (modeRef.current === 'marquee' && marquee) {
      const hittables: Hittable[] = state.panels.map((p) => ({ id: p.id, rect: p.rect }));
      dispatch({ type: 'select', ids: marqueeHits(hittables, marquee) });
    }
    modeRef.current = null;
    panRef.current = null;
    dragRef.current = null;
    marqueeStartRef.current = null;
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
    }
  };

  return (
    <div
      style={{
        overflow: 'hidden', width: '100%', height: '100vh',
        background: '#f4f4f5', cursor: panning ? 'grabbing' : 'grab',
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
          const orig = tupleToRect(p.rect);
          const live = state.panels.find((x) => x.id === p.id)?.rect ?? orig;
          const origScreen = rectToScreen(orig, figWpt, figHpt);
          const liveScreen = rectToScreen(live, figWpt, figHpt);
          const dx = liveScreen.x - origScreen.x;
          const dy = liveScreen.y - origScreen.y;
          return (
            <g key={p.id} transform={`translate(${dx} ${dy})`}>
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
