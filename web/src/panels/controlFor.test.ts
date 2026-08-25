import { describe, expect, it } from 'vitest';
import type { JsonSchemaField } from '../api';
import { controlFor } from './controlFor';

describe('controlFor (Task 13 schema -> control mapping)', () => {
  it('boolean -> checkbox', () => {
    expect(controlFor({ type: 'boolean' })).toEqual({ kind: 'checkbox' });
  });

  it('number -> number input (not integer)', () => {
    expect(controlFor({ type: 'number' })).toEqual({ kind: 'number', integer: false });
  });

  it('integer -> number input, flagged integer', () => {
    expect(controlFor({ type: 'integer' })).toEqual({ kind: 'number', integer: true });
  });

  it('string with enum -> select of the enum', () => {
    const schema: JsonSchemaField = { type: 'string', enum: ['best', 'upper right', 'lower left'] };
    expect(controlFor(schema)).toEqual({ kind: 'select', options: ['best', 'upper right', 'lower left'] });
  });

  it('string with format "color" -> colour input', () => {
    expect(controlFor({ type: 'string', format: 'color' })).toEqual({ kind: 'color' });
  });

  it('plain string -> text input', () => {
    expect(controlFor({ type: 'string' })).toEqual({ kind: 'text' });
  });

  it('array of numbers -> comma-separated numberArray control', () => {
    const schema: JsonSchemaField = { type: 'array', items: { type: 'number' } };
    expect(controlFor(schema)).toEqual({ kind: 'numberArray' });
  });

  it('array of integers also -> numberArray (still a number-typed item)', () => {
    const schema: JsonSchemaField = { type: 'array', items: { type: 'integer' } };
    expect(controlFor(schema)).toEqual({ kind: 'numberArray' });
  });

  it('array of a non-number item type -> unsupported (never guessed at)', () => {
    const schema: JsonSchemaField = { type: 'array', items: { type: 'string' } };
    expect(controlFor(schema)).toEqual({ kind: 'unsupported' });
  });

  it('object with properties -> a nested group, carrying the nested schema for recursion', () => {
    const properties: Record<string, JsonSchemaField> = {
      top: { type: 'boolean', title: 'Top' },
      right: { type: 'boolean', title: 'Right' },
    };
    const schema: JsonSchemaField = { type: 'object', title: 'Spines', properties };
    expect(controlFor(schema)).toEqual({ kind: 'object', properties });
  });

  it('object with no properties -> a nested group with an empty property map', () => {
    expect(controlFor({ type: 'object' })).toEqual({ kind: 'object', properties: {} });
  });

  it('an unrecognised schema type -> unsupported', () => {
    expect(controlFor({ type: 'null' })).toEqual({ kind: 'unsupported' });
  });

  it('matches the real merged legend schema (nested enum + number array), field by field', () => {
    // Mirrors figbuilder/cosmetics.py's SHARED_SCHEMA["legend"].
    const legend: JsonSchemaField = {
      type: 'object',
      title: 'Legend',
      properties: {
        hide: { type: 'boolean', title: 'Hide legend' },
        loc: { type: 'string', enum: ['best', 'upper right', 'lower left'], title: 'Position' },
        bbox_to_anchor: {
          type: 'array', items: { type: 'number' },
          title: 'Anchor (x, y[, w, h]) in axes fractions',
        },
      },
    };
    const outer = controlFor(legend);
    expect(outer.kind).toBe('object');
    if (outer.kind !== 'object') throw new Error('unreachable');
    expect(controlFor(outer.properties.hide)).toEqual({ kind: 'checkbox' });
    expect(controlFor(outer.properties.loc)).toEqual({
      kind: 'select', options: ['best', 'upper right', 'lower left'],
    });
    expect(controlFor(outer.properties.bbox_to_anchor)).toEqual({ kind: 'numberArray' });
  });
});
