/**
 * Pure JSON-Schema -> UI-control mapping for the schema-driven properties
 * form (Task 13). Extracted out of `Properties.tsx` so the mapping is
 * unit-testable without jsdom (this repo has none, and adding a test
 * dependency is forbidden) — `Properties.tsx` switches on `.kind` to render
 * an actual control; this module owns ONLY the mapping, never React, so the
 * architecture rule ("adding a panel type is Python-only — the UI is
 * generated from the schema") stays honest: nothing here hardcodes a field
 * name like "spines" or "legend".
 *
 * Table (`figbuilder/panels/base.py`'s `full_schema` / `figbuilder/cosmetics.py`'s
 * `SHARED_SCHEMA` are what actually produce these shapes):
 *   {"type":"boolean"}                         -> checkbox
 *   {"type":"number"} / {"type":"integer"}     -> number input
 *   {"type":"string","enum":[...]}             -> <select> of the enum
 *   {"type":"string","format":"color"}         -> colour input
 *   {"type":"string"}                          -> text input
 *   {"type":"array","items":{"type":"number"}} -> comma-separated text, parsed to numbers
 *   {"type":"object","properties":{...}}       -> nested group, recursing on the same rules
 */
import type { JsonSchemaField } from '../api';

export type ControlKind =
  | { kind: 'checkbox' }
  | { kind: 'number'; integer: boolean }
  | { kind: 'select'; options: string[] }
  | { kind: 'color' }
  | { kind: 'text' }
  | { kind: 'numberArray' }
  | { kind: 'object'; properties: Record<string, JsonSchemaField> }
  // A schema shape this form doesn't (yet) know how to render — skipped
  // rather than guessed at, so an unrecognised field is simply not offered
  // rather than risking corrupting it.
  | { kind: 'unsupported' };

const isNumericItems = (items: JsonSchemaField | undefined): boolean =>
  items?.type === 'number' || items?.type === 'integer';

export function controlFor(schema: JsonSchemaField): ControlKind {
  switch (schema.type) {
    case 'boolean':
      return { kind: 'checkbox' };
    case 'number':
      return { kind: 'number', integer: false };
    case 'integer':
      return { kind: 'number', integer: true };
    case 'string':
      if (schema.enum) return { kind: 'select', options: schema.enum };
      if (schema.format === 'color') return { kind: 'color' };
      return { kind: 'text' };
    case 'array':
      return isNumericItems(schema.items) ? { kind: 'numberArray' } : { kind: 'unsupported' };
    case 'object':
      return { kind: 'object', properties: schema.properties ?? {} };
    default:
      return { kind: 'unsupported' };
  }
}
