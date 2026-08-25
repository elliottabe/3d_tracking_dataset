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
