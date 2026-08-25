import { describe, expect, it } from 'vitest';
import { formatMm, isValidMmText, parseMm } from './format';

describe('parseMm', () => {
  it('parses a plain number', () => expect(parseMm('12.5', 0)).toBeCloseTo(12.5));
  it('tolerates a unit suffix and whitespace', () => {
    expect(parseMm(' 12.5 mm ', 0)).toBeCloseTo(12.5);
  });
  it('returns the fallback for junk, so a half-typed value never wipes a rect', () => {
    expect(parseMm('', 7)).toBe(7);
    expect(parseMm('-', 7)).toBe(7);
    expect(parseMm('abc', 7)).toBe(7);
  });
  it('accepts negatives', () => expect(parseMm('-3.25', 0)).toBeCloseTo(-3.25));

  // CLAUDE.md / task brief carried-forward requirements: more garbage forms
  // that must never corrupt a rect with NaN.
  it('returns the fallback for whitespace-only input', () => expect(parseMm('   ', 7)).toBe(7));
  it('returns the fallback for a multi-dot number', () => expect(parseMm('1.2.3', 7)).toBe(7));
  it('returns the fallback for a bare dot', () => expect(parseMm('.', 7)).toBe(7));

  // F6: a European-locale user types a decimal comma. Every existing
  // garbage-input case above must still return the fallback unchanged.
  it('accepts a decimal comma', () => expect(parseMm('1,5', 0)).toBeCloseTo(1.5));
  it('accepts a decimal comma with a unit suffix and whitespace', () => {
    expect(parseMm(' 12,5 mm ', 0)).toBeCloseTo(12.5);
  });
  it('accepts a negative decimal-comma value', () => expect(parseMm('-3,25', 0)).toBeCloseTo(-3.25));
  it('still falls back for an ambiguous multi-comma value', () => expect(parseMm('1,2,3', 7)).toBe(7));
  it('still falls back when both a comma and a dot are present', () => expect(parseMm('1,2.3', 7)).toBe(7));
});

describe('isValidMmText (F6)', () => {
  it('is true for a plain number and a decimal-comma number', () => {
    expect(isValidMmText('12.5')).toBe(true);
    expect(isValidMmText('12,5')).toBe(true);
  });
  it('is false for exactly the garbage inputs parseMm falls back on', () => {
    expect(isValidMmText('')).toBe(false);
    expect(isValidMmText('   ')).toBe(false);
    expect(isValidMmText('-')).toBe(false);
    expect(isValidMmText('.')).toBe(false);
    expect(isValidMmText('abc')).toBe(false);
    expect(isValidMmText('1.2.3')).toBe(false);
    expect(isValidMmText('1,2,3')).toBe(false);
  });
});

describe('formatMm', () => {
  it('shows two decimals', () => expect(formatMm(12.3456)).toBe('12.35'));
  it('trims a trailing .00', () => expect(formatMm(12)).toBe('12'));
  it('renders negative zero as 0', () => expect(formatMm(-0)).toBe('0'));
});
