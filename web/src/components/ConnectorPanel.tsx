import { useEffect, useRef, useState } from "react";
import {
  createConnector,
  deleteConnector,
  listConnectors,
  runConnector,
  updateConnector,
} from "../api";
import type { ConnectorCreateRequest, ConnectorView, IndexAttemptStatus } from "../types";

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: IndexAttemptStatus | null }) {
  const label =
    status === "success"
      ? "Synced"
      : status === "in_progress"
        ? "Running"
        : status === "failed"
          ? "Failed"
          : status === "not_started"
            ? "Queued"
            : "Never run";
  const tone =
    status === "success"
      ? "good"
      : status === "failed"
        ? "bad"
        : status === "in_progress"
          ? "active"
          : "neutral";
  return (
    <span className="connector-badge" data-tone={tone}>
      {label}
    </span>
  );
}

// ── Source icon label ─────────────────────────────────────────────────────────

const SOURCE_LABELS: Record<string, string> = {
  file: "Local File",
  web: "Web Crawl",
  rss: "RSS Feed",
  search: "Search Results",
  github: "GitHub",
  slack: "Slack",
  confluence: "Confluence",
  notion: "Notion",
  google_drive: "Google Drive",
  ingestion_api: "Ingestion API",
};

function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source;
}

// ── Add connector form ────────────────────────────────────────────────────────

type SourceType = "file" | "web" | "rss" | "search";

const SOURCE_OPTIONS: { value: SourceType; label: string }[] = [
  { value: "file", label: "Local File" },
  { value: "web", label: "Web Crawl" },
  { value: "rss", label: "RSS Feed" },
  { value: "search", label: "Search Results" },
];

function configHint(source: SourceType): string {
  switch (source) {
    case "file":
      return 'Paths (one per line), e.g.\n/data/docs\n/home/user/notes';
    case "web":
      return 'Seed URLs (one per line), e.g.\nhttps://docs.example.com';
    case "rss":
      return 'Feed URL, e.g.\nhttps://example.com/feed.xml';
    case "search":
      return 'Search queries (one per line), e.g.\nAI research\nopen source tools';
  }
}

function buildConfig(source: SourceType, raw: string): Record<string, unknown> {
  const lines = raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  switch (source) {
    case "file":
      return { paths: lines };
    case "web":
      return { urls: lines, max_depth: 2 };
    case "rss":
      return { feed_url: lines[0] ?? "" };
    case "search":
      return { queries: lines, provider: "retrieval" };
  }
}

interface AddFormProps {
  onCreated: (c: ConnectorView) => void;
  onCancel: () => void;
}

function AddConnectorForm({ onCreated, onCancel }: AddFormProps) {
  const [name, setName] = useState("");
  const [source, setSource] = useState<SourceType>("file");
  const [configRaw, setConfigRaw] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!name.trim()) {
      setError("Name is required.");
      return;
    }
    setSaving(true);
    try {
      const req: ConnectorCreateRequest = {
        name: name.trim(),
        source,
        config: buildConfig(source, configRaw),
      };
      const connector = await createConnector(req);
      onCreated(connector);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create connector.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="add-connector-form" onSubmit={handleSubmit}>
      <h3>Add connector</h3>
      <label>
        Name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My docs"
          autoFocus
        />
      </label>
      <label>
        Source type
        <select value={source} onChange={(e) => setSource(e.target.value as SourceType)}>
          {SOURCE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Configuration
        <textarea
          value={configRaw}
          onChange={(e) => setConfigRaw(e.target.value)}
          placeholder={configHint(source)}
          rows={4}
        />
      </label>
      {error && <p className="form-error">{error}</p>}
      <div className="form-actions">
        <button type="submit" className="btn-primary" disabled={saving}>
          {saving ? "Creating…" : "Create"}
        </button>
        <button type="button" className="btn-ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

// ── Connector row ─────────────────────────────────────────────────────────────

interface ConnectorRowProps {
  connector: ConnectorView;
  onRun: (id: string) => void;
  onToggle: (id: string, enabled: boolean) => void;
  onDelete: (id: string) => void;
  running: boolean;
}

function ConnectorRow({ connector, onRun, onToggle, onDelete, running }: ConnectorRowProps) {
  const lastSync = connector.last_attempt?.updated_at?.slice(0, 16).replace("T", " ") ?? "—";
  const status = connector.last_attempt?.status ?? null;

  return (
    <tr className={`connector-row${connector.enabled ? "" : " disabled"}`}>
      <td className="connector-name">
        <span>{connector.name}</span>
        <span className="connector-source">{sourceLabel(connector.source)}</span>
      </td>
      <td>
        <StatusBadge status={status} />
      </td>
      <td className="connector-meta">
        {connector.last_attempt?.total_documents ?? 0} docs
      </td>
      <td className="connector-meta">{lastSync}</td>
      <td className="connector-actions">
        <button
          type="button"
          className="btn-run"
          disabled={running || !connector.enabled}
          onClick={() => onRun(connector.id)}
          title="Sync now"
        >
          {running ? "…" : "▶ Sync"}
        </button>
        <button
          type="button"
          className="btn-ghost btn-sm"
          onClick={() => onToggle(connector.id, !connector.enabled)}
          title={connector.enabled ? "Disable" : "Enable"}
        >
          {connector.enabled ? "Disable" : "Enable"}
        </button>
        <button
          type="button"
          className="btn-ghost btn-sm btn-danger"
          onClick={() => onDelete(connector.id)}
          title="Delete connector"
        >
          Delete
        </button>
      </td>
    </tr>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function ConnectorPanel() {
  const [connectors, setConnectors] = useState<ConnectorView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async (signal?: AbortSignal) => {
    try {
      const data = await listConnectors({}, { signal });
      setConnectors(data);
      setError(null);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof Error ? e.message : "Failed to load connectors.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    pollRef.current = setInterval(() => load(), 5000);
    return () => {
      controller.abort();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleRun = async (id: string) => {
    setRunningIds((prev) => new Set(prev).add(id));
    try {
      await runConnector(id);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sync failed.");
    } finally {
      setRunningIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const handleToggle = async (id: string, enabled: boolean) => {
    try {
      const updated = await updateConnector(id, { enabled });
      setConnectors((prev) => prev.map((c) => (c.id === id ? updated : c)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Update failed.");
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this connector and all its documents?")) return;
    try {
      await deleteConnector(id);
      setConnectors((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed.");
    }
  };

  const handleCreated = (connector: ConnectorView) => {
    setConnectors((prev) => [connector, ...prev]);
    setShowAdd(false);
  };

  return (
    <div className="connector-panel">
      <div className="connector-panel-header">
        <h2>Connectors</h2>
        <button
          type="button"
          className="btn-primary btn-sm"
          onClick={() => setShowAdd((v) => !v)}
        >
          {showAdd ? "Cancel" : "+ Add connector"}
        </button>
      </div>

      {showAdd && (
        <AddConnectorForm onCreated={handleCreated} onCancel={() => setShowAdd(false)} />
      )}

      {error && <div className="error-banner">{error}</div>}

      {loading && !connectors.length ? (
        <p className="connector-empty">Loading connectors…</p>
      ) : connectors.length === 0 ? (
        <p className="connector-empty">
          No connectors configured. Add one to start ingesting content.
        </p>
      ) : (
        <table className="connector-table">
          <thead>
            <tr>
              <th>Name / Source</th>
              <th>Status</th>
              <th>Documents</th>
              <th>Last sync</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {connectors.map((c) => (
              <ConnectorRow
                key={c.id}
                connector={c}
                onRun={handleRun}
                onToggle={handleToggle}
                onDelete={handleDelete}
                running={runningIds.has(c.id)}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
