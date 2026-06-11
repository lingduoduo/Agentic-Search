import { useEffect, useRef, useState } from "react";
import {
  deleteOpenAPIProvider,
  invokeTool,
  listTools,
  registerOpenAPITools,
} from "../api";
import type { ToolInvokeResponse, ToolView } from "../types";

// ── Source badge ──────────────────────────────────────────────────────────────

function SourceBadge({ source }: { source: string }) {
  const tone = source === "function" ? "good" : source === "openapi" ? "active" : "neutral";
  const label = source === "function" ? "fn" : source === "openapi" ? "openapi" : source;
  return (
    <span className="tool-badge" data-tone={tone}>
      {label}
    </span>
  );
}

// ── Invoke modal ──────────────────────────────────────────────────────────────

interface InvokeModalProps {
  tool: ToolView;
  onClose: () => void;
}

function InvokeModal({ tool, onClose }: InvokeModalProps) {
  const [argsJson, setArgsJson] = useState("{}");
  const [result, setResult] = useState<ToolInvokeResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  async function handleInvoke(e: React.FormEvent) {
    e.preventDefault();
    setJsonError(null);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(argsJson);
    } catch {
      setJsonError("Invalid JSON.");
      return;
    }
    setRunning(true);
    try {
      const res = await invokeTool(tool.name, { arguments: parsed });
      setResult(res);
    } catch (err) {
      setResult({
        response: "",
        raw: null,
        errors: [err instanceof Error ? err.message : "Invocation failed."],
      });
    } finally {
      setRunning(false);
    }
  }

  // Build a pretty schema hint for the arguments textarea placeholder
  const props = (tool.parameters as { properties?: Record<string, unknown> }).properties ?? {};
  const hint = JSON.stringify(
    Object.fromEntries(Object.keys(props).map((k) => [k, "..."])),
    null,
    2,
  );

  return (
    <div className="tool-modal-overlay" onClick={onClose}>
      <div className="tool-modal" onClick={(e) => e.stopPropagation()}>
        <div className="tool-modal-header">
          <h3>Invoke: {tool.name}</h3>
          <button type="button" className="btn-ghost btn-sm" onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="tool-modal-desc">{tool.description || "No description."}</p>
        <form onSubmit={handleInvoke}>
          <label>
            Arguments (JSON)
            <textarea
              className="tool-args-input"
              value={argsJson}
              onChange={(e) => setArgsJson(e.target.value)}
              placeholder={hint}
              rows={6}
              spellCheck={false}
            />
          </label>
          {jsonError && <p className="form-error">{jsonError}</p>}
          <div className="form-actions">
            <button type="submit" className="btn-primary" disabled={running}>
              {running ? "Running…" : "▶ Invoke"}
            </button>
            <button type="button" className="btn-ghost" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>
        {result && (
          <div className="tool-result">
            {result.errors.length > 0 ? (
              <div className="tool-result-errors">
                {result.errors.map((e, i) => (
                  <p key={i} className="form-error">
                    {e}
                  </p>
                ))}
              </div>
            ) : (
              <pre className="tool-result-pre">{result.response}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── OpenAPI registration form ─────────────────────────────────────────────────

interface RegisterFormProps {
  onRegistered: (names: string[]) => void;
  onCancel: () => void;
}

function RegisterForm({ onRegistered, onCancel }: RegisterFormProps) {
  const [providerName, setProviderName] = useState("");
  const [openApiJson, setOpenApiJson] = useState("");
  const [authHeader, setAuthHeader] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!providerName.trim()) {
      setError("Provider name is required.");
      return;
    }
    if (!openApiJson.trim()) {
      setError("OpenAPI JSON is required.");
      return;
    }
    setSaving(true);
    const headers: Record<string, string> = {};
    if (authHeader.trim()) {
      headers["Authorization"] = authHeader.trim();
    }
    try {
      const res = await registerOpenAPITools({
        name: providerName.trim(),
        openapi_json: openApiJson.trim(),
        headers,
      });
      onRegistered(res.tool_names);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="tool-register-form" onSubmit={handleSubmit}>
      <h3>Register OpenAPI tools</h3>
      <label>
        Provider name
        <input
          value={providerName}
          onChange={(e) => setProviderName(e.target.value)}
          placeholder="My API"
          autoFocus
        />
      </label>
      <label>
        OpenAPI JSON schema
        <textarea
          value={openApiJson}
          onChange={(e) => setOpenApiJson(e.target.value)}
          placeholder={'{"openapi":"3.0.0","info":{"title":"..."},"paths":{...}}'}
          rows={8}
          spellCheck={false}
        />
      </label>
      <label>
        Authorization header (optional)
        <input
          value={authHeader}
          onChange={(e) => setAuthHeader(e.target.value)}
          placeholder="Bearer sk-..."
        />
      </label>
      {error && <p className="form-error">{error}</p>}
      <div className="form-actions">
        <button type="submit" className="btn-primary" disabled={saving}>
          {saving ? "Registering…" : "Register"}
        </button>
        <button type="button" className="btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

// ── Tool row ──────────────────────────────────────────────────────────────────

interface ToolRowProps {
  tool: ToolView;
  onInvoke: (t: ToolView) => void;
  onDeleteProvider: (providerId: string) => void;
}

function ToolRow({ tool, onInvoke, onDeleteProvider }: ToolRowProps) {
  const props = (tool.parameters as { properties?: Record<string, unknown> }).properties ?? {};
  const paramList = Object.keys(props).join(", ") || "—";

  return (
    <tr className="tool-row">
      <td className="tool-name">
        <span>{tool.name}</span>
        <SourceBadge source={tool.source} />
      </td>
      <td className="tool-desc">{tool.description || <em>No description</em>}</td>
      <td className="tool-params">{paramList}</td>
      <td className="tool-actions">
        <button
          type="button"
          className="btn-run btn-sm"
          onClick={() => onInvoke(tool)}
          title="Test-invoke this tool"
        >
          ▶ Test
        </button>
        {tool.provider_id && (
          <button
            type="button"
            className="btn-ghost btn-sm btn-danger"
            onClick={() => onDeleteProvider(tool.provider_id!)}
            title="Remove all tools from this provider"
          >
            Delete provider
          </button>
        )}
      </td>
    </tr>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function ToolPanel() {
  const [tools, setTools] = useState<ToolView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showRegister, setShowRegister] = useState(false);
  const [invoking, setInvoking] = useState<ToolView | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async (signal?: AbortSignal) => {
    try {
      const data = await listTools({ signal });
      setTools(data);
      setError(null);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof Error ? e.message : "Failed to load tools.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    pollRef.current = setInterval(() => load(), 10000);
    return () => {
      controller.abort();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleRegistered = async (names: string[]) => {
    setShowRegister(false);
    await load();
    setError(
      names.length
        ? null
        : "No tools were registered from the provided schema.",
    );
  };

  const handleDeleteProvider = async (providerId: string) => {
    if (!confirm("Delete all tools from this provider?")) return;
    try {
      await deleteOpenAPIProvider(providerId);
      setTools((prev) => prev.filter((t) => t.provider_id !== providerId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    }
  };

  // Group function tools vs openapi tools for display
  const fnTools = tools.filter((t) => t.source === "function");
  const apiTools = tools.filter((t) => t.source !== "function");

  return (
    <div className="tool-panel">
      {invoking && (
        <InvokeModal tool={invoking} onClose={() => setInvoking(null)} />
      )}

      <div className="tool-panel-header">
        <h2>Tools</h2>
        <button
          type="button"
          className="btn-primary btn-sm"
          onClick={() => setShowRegister((v) => !v)}
        >
          {showRegister ? "Cancel" : "+ Register OpenAPI"}
        </button>
      </div>

      {showRegister && (
        <RegisterForm
          onRegistered={handleRegistered}
          onCancel={() => setShowRegister(false)}
        />
      )}

      {error && <div className="error-banner">{error}</div>}

      {loading && !tools.length ? (
        <p className="tool-empty">Loading tools…</p>
      ) : tools.length === 0 ? (
        <p className="tool-empty">
          No tools registered. Built-in function tools load automatically; add
          OpenAPI schemas to register external API tools.
        </p>
      ) : (
        <>
          {fnTools.length > 0 && (
            <section>
              <h3 className="tool-section-label">Built-in function tools</h3>
              <table className="tool-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Description</th>
                    <th>Parameters</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {fnTools.map((t) => (
                    <ToolRow
                      key={t.name}
                      tool={t}
                      onInvoke={setInvoking}
                      onDeleteProvider={handleDeleteProvider}
                    />
                  ))}
                </tbody>
              </table>
            </section>
          )}
          {apiTools.length > 0 && (
            <section>
              <h3 className="tool-section-label">OpenAPI tools</h3>
              <table className="tool-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Description</th>
                    <th>Parameters</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {apiTools.map((t) => (
                    <ToolRow
                      key={t.name}
                      tool={t}
                      onInvoke={setInvoking}
                      onDeleteProvider={handleDeleteProvider}
                    />
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </div>
  );
}
