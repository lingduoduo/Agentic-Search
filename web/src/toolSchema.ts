/**
 * Turn a tool's JSON Schema into form fields.
 *
 * A JSON Schema is a form definition, but the invoke dialog treated it as
 * decoration and made the user hand-write the arguments object. These helpers
 * read the schema so the dialog can render labelled inputs instead.
 *
 * Only the shapes a flat form can honestly represent are supported. Anything
 * else (nested objects, tuples, oneOf) reports `supported: false` so the dialog
 * falls back to the raw JSON editor rather than silently dropping arguments.
 */

export type FieldKind = "string" | "text" | "number" | "integer" | "boolean" | "enum" | "stringList";

export interface ToolField {
  name: string;
  kind: FieldKind;
  label: string;
  description: string;
  required: boolean;
  options: string[];
}

export interface ToolForm {
  fields: ToolField[];
  /** False when the schema has shapes a flat form cannot express. */
  supported: boolean;
}

type JsonObject = Record<string, unknown>;

const asObject = (v: unknown): JsonObject | null =>
  typeof v === "object" && v !== null && !Array.isArray(v) ? (v as JsonObject) : null;

/** "max_results" / "maxResults" -> "Max results" */
export function humanizeName(name: string): string {
  const spaced = name
    .replace(/[_-]+/g, " ")
    // camelCase → separate words, lowercased so the label reads as a sentence
    // ("Max results") rather than a title ("Max Results").
    .replace(/([a-z0-9])([A-Z])/g, (_m, a: string, b: string) => `${a} ${b.toLowerCase()}`)
    .trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function fieldFor(name: string, raw: JsonObject, required: boolean): ToolField | null {
  const description = typeof raw.description === "string" ? raw.description : "";
  const base = { name, label: humanizeName(name), description, required, options: [] as string[] };

  const enumValues = Array.isArray(raw.enum)
    ? raw.enum.filter((v): v is string => typeof v === "string")
    : null;
  if (enumValues && enumValues.length > 0) {
    return { ...base, kind: "enum", options: enumValues };
  }

  switch (raw.type) {
    case "string":
      // Long free text (a document, a prompt) deserves a textarea.
      return { ...base, kind: /\b(text|content|body|prompt)\b/i.test(name) ? "text" : "string" };
    case "number":
      return { ...base, kind: "number" };
    case "integer":
      return { ...base, kind: "integer" };
    case "boolean":
      return { ...base, kind: "boolean" };
    case "array": {
      const items = asObject(raw.items);
      // Only arrays of plain strings; anything richer needs the JSON editor.
      return items && items.type === "string" ? { ...base, kind: "stringList" } : null;
    }
    default:
      return null;
  }
}

export function toolFormFromSchema(parameters: unknown): ToolForm {
  const schema = asObject(parameters);
  if (!schema) return { fields: [], supported: false };
  const properties = asObject(schema.properties);
  if (!properties) {
    // No declared parameters is a supported case: a tool taking no arguments.
    return { fields: [], supported: schema.properties === undefined };
  }
  const declaredRequired = schema.required;
  const requiredNames = new Set(
    Array.isArray(declaredRequired)
      ? declaredRequired.filter((v): v is string => typeof v === "string")
      : [],
  );

  const fields: ToolField[] = [];
  for (const [name, rawValue] of Object.entries(properties)) {
    const raw = asObject(rawValue);
    const field = raw ? fieldFor(name, raw, requiredNames.has(name)) : null;
    if (!field) return { fields: [], supported: false };
    fields.push(field);
  }
  return { fields, supported: true };
}

/** The empty value a field starts at, so controls are always controlled. */
export function initialValues(fields: ToolField[]): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const f of fields) {
    values[f.name] = f.kind === "stringList" ? [""] : f.kind === "boolean" ? false : "";
  }
  return values;
}

const isBlank = (v: unknown) => typeof v === "string" && v.trim() === "";

/**
 * Convert form values into the arguments object.
 *
 * Blank optional fields are omitted rather than sent as "" — an empty string is
 * a real value to a tool, and sending one silently changes what it does.
 */
export function buildArguments(
  fields: ToolField[],
  values: Record<string, unknown>,
): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const f of fields) {
    const value = values[f.name];
    if (f.kind === "stringList") {
      const items = (Array.isArray(value) ? value : [])
        .filter((v): v is string => typeof v === "string")
        .map((v) => v.trim())
        .filter((v) => v !== "");
      if (items.length > 0) args[f.name] = items;
      continue;
    }
    if (f.kind === "boolean") {
      if (value === true) args[f.name] = true;
      continue;
    }
    if (isBlank(value) || value === undefined || value === null) continue;
    if (f.kind === "number" || f.kind === "integer") {
      const n = Number(value);
      if (!Number.isNaN(n)) args[f.name] = n;
      continue;
    }
    args[f.name] = value;
  }
  return args;
}

/** Human-readable problems, so the user hears about them before invoking. */
export function validate(
  fields: ToolField[],
  values: Record<string, unknown>,
): Record<string, string> {
  const errors: Record<string, string> = {};
  const args = buildArguments(fields, values);
  for (const f of fields) {
    if (f.required && !(f.name in args)) {
      errors[f.name] = `${f.label} is required.`;
      continue;
    }
    if ((f.kind === "number" || f.kind === "integer") && f.name in args) {
      const n = args[f.name] as number;
      if (f.kind === "integer" && !Number.isInteger(n)) {
        errors[f.name] = `${f.label} must be a whole number.`;
      }
    } else if (
      (f.kind === "number" || f.kind === "integer") &&
      !isBlank(values[f.name]) &&
      values[f.name] !== undefined &&
      Number.isNaN(Number(values[f.name]))
    ) {
      errors[f.name] = `${f.label} must be a number.`;
    }
  }
  return errors;
}
