/**
 * Millimetre text <-> number for the numeric-entry fields (Properties panel).
 *
 * The boundary rule: the user types/reads MM here; the reducer only ever
 * stores figure FRACTIONS. Convert with fracToMm/mmToFrac right at this
 * boundary, never inside the reducer.
 */

/**
 * Clean a raw input string down to something `Number()` can parse, tolerating
 * a trailing "mm" unit, stray whitespace, and a European-locale decimal comma
 * ("1,5" -> "1.5"). The comma swap only fires when there is exactly one comma
 * and no dot already present — "1.2,3" or "1,2,3" are still ambiguous/
 * malformed and fall through unparsed, same as before this was added.
 * Returns '' for anything that should fall back (matching the empty-string
 * case), so callers never have to special-case it separately.
 */
function normalizeMmText(text: string): string {
  const cleaned = text.trim().replace(/mm$/i, '').trim();
  if (cleaned === '') return '';
  return cleaned.includes(',') && !cleaned.includes('.') ? cleaned.replace(',', '.') : cleaned;
}

/**
 * Parse a user-typed mm string. Tolerates a trailing "mm" unit, stray
 * whitespace, and a decimal comma. Returns `fallback` — never NaN — for
 * anything that isn't a finite number: empty string, whitespace-only, a bare
 * "-" or ".", "abc", or a malformed multi-dot number like "1.2.3". A NaN
 * reaching a rect silently corrupts the layout and renders as a vanished
 * panel, so this function must never produce one.
 */
export function parseMm(text: string, fallback: number): number {
  const normalized = normalizeMmText(text);
  if (normalized === '') return fallback;
  const v = Number(normalized);
  return Number.isFinite(v) ? v : fallback;
}

/**
 * Whether `text` parses to a real number under the same rules as `parseMm`,
 * as opposed to falling back. `Properties.tsx` uses this (F6) to tell a
 * genuinely-rejected edit — where the displayed text must be reset to match
 * state, or the uncontrolled input keeps showing garbage indefinitely —
 * apart from an edit that happens to reproduce the current value.
 */
export function isValidMmText(text: string): boolean {
  const normalized = normalizeMmText(text);
  return normalized !== '' && Number.isFinite(Number(normalized));
}

/** Format a mm value to at most 2 decimals, trimming a trailing ".00". */
export function formatMm(v: number): string {
  const s = (Object.is(v, -0) ? 0 : v).toFixed(2);
  return s.replace(/\.00$/, '');
}
