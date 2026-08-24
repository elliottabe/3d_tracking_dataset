// Prints the fixture emission for the Python<->TypeScript conformance test
// (tests/test_figbuilder_annot_conformance.py), using the same figure as the
// Python side's `_fig()`.
import { readFileSync } from 'node:fs';
import { emitAnnotations, type Annotation, type FigureSpec } from './emit';

const fig: FigureSpec = {
  width_mm: 100, height_mm: 50,
  panels: [{ id: 'wing', rect: [0.1, 0.5, 0.8, 0.4] }],
};
const anns = JSON.parse(
  readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'),
) as Annotation[];
process.stdout.write(emitAnnotations(fig, anns));
