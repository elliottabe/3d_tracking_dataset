// Typed `fetch` wrappers around the figbuilder FastAPI backend
// (figbuilder/server.py). The server is stateless with respect to layout:
// the browser owns the design and posts whatever it wants rendered.
const BASE = import.meta.env.VITE_API ?? 'http://127.0.0.1:8765';

export interface PanelSpecDTO {
  id: string; type: string; rect: [number, number, number, number];
  data: Record<string, { dataset: string; slice?: string }>;
  spec: Record<string, unknown>; group?: string | null; z?: number;
}
/** On-disk group shape (`figbuilder/figure.py` `GroupSpec`/`_unparse`):
 *  snake_case, rect as a 4-tuple. The editor's `Group` (`layout/groups.ts`)
 *  is camelCase with an `{x,y,w,h}` rect — `layout/load.ts`/`layout/save.ts`
 *  are the ONLY places that convert between the two. Giving this its own
 *  DTO type (rather than `unknown[]`) is what stops a raw
 *  `as unknown as Group[]` cast from silently type-checking again (F1).
 */
export interface GroupDTO {
  id: string; axis: 'x' | 'y'; gutter_mm: number; equal: boolean;
  rect: [number, number, number, number];
}
export interface FigureDoc {
  figure: { width_mm: number; height_mm: number; dpi: number };
  panels: PanelSpecDTO[]; groups: GroupDTO[]; annotations: unknown[];
}
export interface TileDTO {
  svg: string; ink_box: [number, number, number, number];
  cache_hit: boolean; overflows: boolean;
}

const get = async <T>(path: string) => {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return (await r.json()) as T;
};

export const getFigure = () => get<FigureDoc>('/api/figure');
export const getBundle = () => get<unknown>('/api/bundle');
export const getPanelTypes = () => get<unknown[]>('/api/panel-types');

export async function renderPanel(panel: PanelSpecDTO): Promise<TileDTO> {
  const r = await fetch(`${BASE}/api/panel`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ panel }),
  });
  if (!r.ok) throw new Error(`/api/panel: ${r.status}`);
  return (await r.json()) as TileDTO;
}

export async function saveFigure(doc: unknown): Promise<void> {
  const r = await fetch(`${BASE}/api/figure`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(doc),
  });
  if (!r.ok) throw new Error(`save failed: ${r.status} ${await r.text()}`);
}
