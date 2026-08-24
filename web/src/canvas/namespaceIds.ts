/**
 * Mirrors figbuilder/svgutil.py:namespace_ids. That module was rewritten to
 * walk the parsed tree and rewrite only ATTRIBUTE values (`id`, a bare
 * `#id` reference, or a `url(#id)` reference) — never element text/tail —
 * matching complete ids from a collected set. A byte-level string
 * replacement would corrupt `<text>` content that merely contains a
 * substring shaped like a reference (e.g. the label "f(#a)"), so this walks
 * the DOM tree the same way, never `svg.split(...)`.
 *
 * Uses the standard `DOMParser`/`XMLSerializer` globals (available in the
 * browser; a verification script running under Node supplies them via a
 * DOM polyfill such as jsdom).
 */

const SEP = '__';
const URL_REF = /url\(#([^)]+)\)/g;

/**
 * Returns the serialized children of the tile's `<svg>` root (with ids
 * rewritten), ready to drop inside an overlay `<g>` — the tile's own `<svg>`
 * envelope is discarded, only its content is kept.
 */
export function namespaceIdsAndExtractInner(svg: string, prefix: string): string {
  const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
  const root = doc.documentElement;
  if (root.querySelector('parsererror')) return '';

  const all = [root, ...Array.from(root.querySelectorAll('*'))];
  const ids = new Set<string>();
  for (const el of all) {
    const id = el.getAttribute('id');
    if (id) ids.add(id);
  }

  if (ids.size > 0) {
    const prefixed = (old: string) => `${prefix}${SEP}${old}`;
    for (const el of all) {
      const oldId = el.getAttribute('id');
      if (oldId !== null) el.setAttribute('id', prefixed(oldId));

      for (const attr of Array.from(el.attributes)) {
        const value = attr.value;

        // Bare fragment reference, e.g. xlink:href="#m2e9b37ff53".
        if (value.startsWith('#') && ids.has(value.slice(1))) {
          el.setAttribute(attr.name, `#${prefixed(value.slice(1))}`);
          continue;
        }

        // url(#X) occurrences anywhere in the value: covers
        // clip-path="url(#p1)" and embedded refs like a `style` of
        // "...;marker-end:url(#arrow)".
        if (value.includes('url(#')) {
          const newValue = value.replace(URL_REF, (whole, target: string) =>
            ids.has(target) ? `url(#${prefixed(target)})` : whole);
          if (newValue !== value) el.setAttribute(attr.name, newValue);
        }
      }
    }
  }

  const serializer = new XMLSerializer();
  return Array.from(root.childNodes).map((n) => serializer.serializeToString(n)).join('');
}
