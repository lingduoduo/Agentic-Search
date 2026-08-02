import { describe, expect, it } from "vitest";
import {
  buildArguments,
  humanizeName,
  initialValues,
  toolFormFromSchema,
  validate,
} from "../toolSchema";

// The real web_search schema — the one the dialog made you hand-write JSON for.
const WEB_SEARCH = {
  type: "object",
  properties: {
    queries: {
      type: "array",
      items: { type: "string" },
      description: "One or more search queries to run in parallel.",
    },
  },
  required: ["queries"],
};

describe("toolFormFromSchema", () => {
  it("reads a real tool schema into a labelled field", () => {
    const { fields, supported } = toolFormFromSchema(WEB_SEARCH);
    expect(supported).toBe(true);
    expect(fields).toEqual([
      {
        name: "queries",
        kind: "stringList",
        label: "Queries",
        description: "One or more search queries to run in parallel.",
        required: true,
        options: [],
      },
    ]);
  });

  it("maps scalar types to their controls", () => {
    const { fields } = toolFormFromSchema({
      type: "object",
      properties: {
        query: { type: "string" },
        top_k: { type: "integer" },
        threshold: { type: "number" },
        rerank: { type: "boolean" },
        mode: { type: "string", enum: ["bm25", "dense"] },
      },
      required: ["query"],
    });
    expect(fields.map((f) => [f.name, f.kind])).toEqual([
      ["query", "string"],
      ["top_k", "integer"],
      ["threshold", "number"],
      ["rerank", "boolean"],
      ["mode", "enum"],
    ]);
    expect(fields.find((f) => f.name === "mode")?.options).toEqual(["bm25", "dense"]);
    expect(fields.find((f) => f.name === "query")?.required).toBe(true);
    expect(fields.find((f) => f.name === "top_k")?.required).toBe(false);
  });

  it("gives long free text a textarea", () => {
    const { fields } = toolFormFromSchema({
      type: "object",
      properties: { text: { type: "string" } },
    });
    expect(fields[0].kind).toBe("text");
  });

  it("treats a tool with no parameters as supported", () => {
    expect(toolFormFromSchema({ type: "object" })).toEqual({ fields: [], supported: true });
  });

  // Falling back is the honest outcome: rendering a partial form would silently
  // drop the arguments it could not represent.
  it("reports unsupported for a nested object", () => {
    expect(
      toolFormFromSchema({
        type: "object",
        properties: { filters: { type: "object", properties: { a: { type: "string" } } } },
      }).supported,
    ).toBe(false);
  });

  it("reports unsupported for an array of objects", () => {
    expect(
      toolFormFromSchema({
        type: "object",
        properties: { items: { type: "array", items: { type: "object" } } },
      }).supported,
    ).toBe(false);
  });
});

describe("humanizeName", () => {
  it("turns identifiers into labels", () => {
    expect(humanizeName("max_results")).toBe("Max results");
    expect(humanizeName("maxResults")).toBe("Max results");
    expect(humanizeName("query")).toBe("Query");
  });
});

describe("buildArguments", () => {
  const { fields } = toolFormFromSchema({
    type: "object",
    properties: {
      query: { type: "string" },
      top_k: { type: "integer" },
      rerank: { type: "boolean" },
      tags: { type: "array", items: { type: "string" } },
    },
  });

  it("omits blank optional fields rather than sending empty strings", () => {
    expect(buildArguments(fields, initialValues(fields))).toEqual({});
  });

  it("coerces numbers and keeps only non-empty list entries", () => {
    const args = buildArguments(fields, {
      query: "faiss",
      top_k: "5",
      rerank: true,
      tags: ["a", "  ", "b"],
    });
    expect(args).toEqual({ query: "faiss", top_k: 5, rerank: true, tags: ["a", "b"] });
  });

  it("trims list entries", () => {
    expect(buildArguments(fields, { ...initialValues(fields), tags: [" a "] })).toEqual({
      tags: ["a"],
    });
  });
});

describe("validate", () => {
  const { fields } = toolFormFromSchema(WEB_SEARCH);

  it("flags a required field the user left empty", () => {
    expect(validate(fields, initialValues(fields))).toEqual({
      queries: "Queries is required.",
    });
  });

  it("passes once the required field has a value", () => {
    expect(validate(fields, { queries: ["faiss"] })).toEqual({});
  });

  it("flags a non-numeric number", () => {
    const numeric = toolFormFromSchema({
      type: "object",
      properties: { top_k: { type: "integer" } },
    }).fields;
    expect(validate(numeric, { top_k: "abc" })).toEqual({
      top_k: "Top k must be a number.",
    });
  });

  it("flags a fractional integer", () => {
    const numeric = toolFormFromSchema({
      type: "object",
      properties: { top_k: { type: "integer" } },
    }).fields;
    expect(validate(numeric, { top_k: "2.5" })).toEqual({
      top_k: "Top k must be a whole number.",
    });
  });
});
