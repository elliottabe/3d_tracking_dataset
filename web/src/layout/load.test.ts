/**
 * F1 regression: `Canvas.tsx` used to load `figure.json` groups via
 * `d.groups as unknown as Group[]` — a raw cast that type-checks a document
 * whose groups are still snake_case with a tuple `rect`. Every existing
 * group test in `state/editorStore.test.ts` builds groups THROUGH the
 * reducer, starting from `groups: []`; none of them load a document that
 * already has groups on disk, which is the only path the real
 * `figures/paper_figures/fig4.json` ever takes. This file closes that gap
 * by reading the real, committed fixture from disk (no jsdom needed — this
 * is pure data conversion, not a rendered component).
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import type { FigureDoc } from '../api';
import { reducer, initialState } from '../state/editorStore';
import { groupFromDto, groupsFromDoc, panelsFromDoc } from './load';

const HERE = dirname(fileURLToPath(import.meta.url));
// web/src/layout/ -> repo root -> figures/paper_figures/fig4.json
const FIXTURE_PATH = resolve(HERE, '../../../figures/paper_figures/fig4.json');

function loadFixtureDoc(): FigureDoc {
  return JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8')) as FigureDoc;
}

describe('groupFromDto / groupsFromDoc (F1)', () => {
  it('converts the real fig4.json groups to fully-typed Group objects', () => {
    const doc = loadFixtureDoc();
    // Sanity: the fixture actually has groups, or this test would pass for
    // the wrong reason (nothing to convert).
    expect(doc.groups.length).toBeGreaterThan(0);
    expect(doc.groups.map((g) => g.id).sort()).toEqual(['render_strip', 'video_strip']);

    const groups = groupsFromDoc(doc);
    expect(groups).toHaveLength(doc.groups.length);
    for (const g of groups) {
      // The F1 bug: a raw cast left `gutterMm` `undefined` (the DTO only
      // has `gutter_mm`) and `rect` still an array (the DTO's tuple).
      expect(typeof g.gutterMm).toBe('number');
      expect(Number.isFinite(g.gutterMm)).toBe(true);
      expect(Array.isArray(g.rect)).toBe(false);
      expect(typeof g.rect).toBe('object');
      for (const k of ['x', 'y', 'w', 'h'] as const) {
        expect(typeof g.rect[k]).toBe('number');
        expect(Number.isFinite(g.rect[k])).toBe(true);
      }
    }
  });

  it('one converted group matches its on-disk values exactly', () => {
    const doc = loadFixtureDoc();
    const raw = doc.groups.find((g) => g.id === 'video_strip')!;
    const g = groupFromDto(raw);
    expect(g.gutterMm).toBe(raw.gutter_mm);
    expect(g.equal).toBe(raw.equal);
    expect(g.rect).toEqual({ x: raw.rect[0], y: raw.rect[1], w: raw.rect[2], h: raw.rect[3] });
  });
});

describe('loading the real fig4.json through the reducer produces no NaN (F1)', () => {
  it('selecting a grouped panel and editing its gutter never NaNs the group or its children', () => {
    const doc = loadFixtureDoc();
    let s = reducer(initialState, {
      type: 'load',
      figWmm: doc.figure.width_mm,
      figHmm: doc.figure.height_mm,
      panels: panelsFromDoc(doc),
      groups: groupsFromDoc(doc),
    });

    expect(s.groups.map((g) => g.id).sort()).toEqual(['render_strip', 'video_strip']);

    const member = s.panels.find((p) => p.group === 'video_strip')!;
    expect(member).toBeTruthy();

    // Clicking one grouped panel must select the whole group (M2-13).
    s = reducer(s, { type: 'select', ids: [member.id] });
    const groupMembers = s.panels.filter((p) => p.group === 'video_strip');
    expect(s.selection.sort()).toEqual(groupMembers.map((p) => p.id).sort());
    expect(groupMembers.length).toBe(4);

    s = reducer(s, { type: 'setGutter', id: 'video_strip', gutterMm: 3 });
    const g = s.groups.find((gg) => gg.id === 'video_strip')!;
    expect(Number.isNaN(g.gutterMm)).toBe(false);
    expect(g.gutterMm).toBe(3);
    expect(Number.isNaN(g.rect.x)).toBe(false);
    expect(Number.isNaN(g.rect.y)).toBe(false);
    for (const p of s.panels.filter((pp) => pp.group === 'video_strip')) {
      expect(Number.isNaN(p.rect.x)).toBe(false);
      expect(Number.isNaN(p.rect.y)).toBe(false);
      expect(Number.isNaN(p.rect.w)).toBe(false);
      expect(Number.isNaN(p.rect.h)).toBe(false);
    }
  });
});
