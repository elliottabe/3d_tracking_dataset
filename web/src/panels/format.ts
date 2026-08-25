/**
 * Millimetre text <-> number for the numeric-entry fields (Properties panel).
 *
 * The boundary rule: the user types/reads MM here; the reducer only ever
 * stores figure FRACTIONS. Convert with fracToMm/mmToFrac right at this
 * boundary, never inside the reducer.
 */

/**
 * Parse a user-typed mm string. Tolerates a trailing "mm" unit and stray
 * whitespace. Returns `fallback` — never NaN — for anything that isn't a
 * finite number: empty string, whitespace-only, a bare "-" or ".", "abc",
 * or a malformed multi-dot number like "1.2.3". A NaN reaching a rect
 * silently corrupts the layout and renders as a vanished panel, so this
 * function must never produce one.
 */
export function parseMm(text: string, fallback: number): number {
  const cleaned = text.trim().replace(/mm$/i, '').trim();
  if (cleaned === '') return fallback;
  const v = Number(cleaned);
  return Number.isFinite(v) ? v : fallback;
}

/** Format a mm value to at most 2 decimals, trimming a trailing ".00". */
export function formatMm(v: number): string {
  const s = (Object.is(v, -0) ? 0 : v).toFixed(2);
  return s.replace(/\.00$/, '');
}
