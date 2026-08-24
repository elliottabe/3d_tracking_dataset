import { describe, expect, it } from 'vitest';
import { emitAnnotations, num, type FigureSpec } from './emit';

const fig: FigureSpec = {
  width_mm: 100, height_mm: 50,
  panels: [{ id: 'wing', rect: [0.1, 0.5, 0.8, 0.4] }],
};

describe('emitAnnotations', () => {
  it('flips y from figure space to SVG space', () => {
    const svg = emitAnnotations(fig, [
      { id: 't', kind: 'text', pos_mm: [0, 0], text: 'x' }]);
    // Must match figbuilder/annot.py byte-for-byte, not JS full precision.
    expect(svg).toContain('y="141.732283"');
  });

  it('offsets a panel-parented annotation from that panel corner', () => {
    const svg = emitAnnotations(fig, [
      { id: 'L', kind: 'text', parent: 'wing', pos_mm: [0, 0], text: 'A' }]);
    expect(svg).toContain('x="28.346457"');
  });

  it('formats numbers the same way the Python emitter does', () => {
    expect(num(0.8)).toBe('0.8');
    expect(num(8)).toBe('8');
    expect(num(141.73228346456693)).toBe('141.732283');
    expect(num(-0)).toBe('0');
  });

  it('rejects an unknown kind by name', () => {
    expect(() => emitAnnotations(fig, [{ id: 'z', kind: 'hologram' } as never]))
      .toThrow(/unknown annotation kind/);
  });

  it('emits the arrowhead defs first, and only when an arrow is present', () => {
    const noArrow = emitAnnotations(fig, [{ id: 't', kind: 'text', pos_mm: [0, 0], text: 'x' }]);
    expect(noArrow).not.toContain('<ns0:defs');

    const withArrow = emitAnnotations(fig, [
      { id: 't', kind: 'text', pos_mm: [0, 0], text: 'x' },
      { id: 'a', kind: 'arrow', pos_mm: [1, 1], to_mm: [9, 5] },
    ]);
    expect(withArrow.indexOf('<ns0:defs')).toBe(0);
  });
});
