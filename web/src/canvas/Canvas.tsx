import { useEffect, useRef, useState } from 'react';
import { getFigure, renderPanel, type FigureDoc, type TileDTO } from '../api';
import { namespaceIdsAndExtractInner } from './namespaceIds';

const PT_PER_MM = 72 / 25.4;

export function Canvas() {
  const [doc, setDoc] = useState<FigureDoc | null>(null);
  const [tiles, setTiles] = useState<Record<string, TileDTO>>({});
  const [view, setView] = useState({ scale: 2, tx: 0, ty: 0 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => { void getFigure().then(setDoc); }, []);

  useEffect(() => {
    if (!doc) return;
    let alive = true;
    void (async () => {
      for (const p of doc.panels) {
        const t = await renderPanel(p);
        if (!alive) return;
        setTiles((prev) => ({ ...prev, [p.id]: t }));
      }
    })();
    return () => { alive = false; };
  }, [doc]);

  if (!doc) return <p>loading…</p>;
  const wPt = doc.figure.width_mm * PT_PER_MM;
  const hPt = doc.figure.height_mm * PT_PER_MM;

  return (
    <div
      style={{
        overflow: 'hidden', width: '100%', height: '100vh',
        background: '#f4f4f5', cursor: dragging ? 'grabbing' : 'grab',
      }}
      onWheel={(e) => {
        e.preventDefault();
        setView((v) => ({
          ...v,
          scale: Math.min(12, Math.max(0.25, v.scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1))),
        }));
      }}
      onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY }; setDragging(true); }}
      onPointerUp={() => { drag.current = null; setDragging(false); }}
      onPointerLeave={() => { drag.current = null; setDragging(false); }}
      onPointerMove={(e) => {
        if (!drag.current) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        drag.current = { x: e.clientX, y: e.clientY };
        setView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }));
      }}
    >
      <svg
        width={wPt * view.scale} height={hPt * view.scale}
        viewBox={`0 0 ${wPt} ${hPt}`}
        style={{
          transform: `translate(${view.tx}px, ${view.ty}px)`,
          background: 'white', boxShadow: '0 1px 8px rgba(0,0,0,.15)',
        }}
      >
        {doc.panels.map((p) => {
          const t = tiles[p.id];
          if (!t) return null;
          return (
            <g
              key={p.id} id={`panel_${p.id}`}
              dangerouslySetInnerHTML={{ __html: namespaceIdsAndExtractInner(t.svg, p.id) }}
            />
          );
        })}
      </svg>
    </div>
  );
}
