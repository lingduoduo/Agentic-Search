import { useEffect, useState } from "react";
import { getAuditSummary, getQueryHistory } from "../api";
import type { AuditSummary, QueryHistoryItem, QueryHistoryPage } from "../types";

const PAGE_SIZE = 20;

type FeedbackFilter = "all" | "like" | "dislike";

function DislikeRateBadge({ rate }: { rate: number }) {
  const pct = (rate * 100).toFixed(1);
  const tone = rate >= 0.3 ? "bad" : rate >= 0.1 ? "watch" : "good";
  return (
    <span
      className="dislike-badge"
      data-tone={tone}
      title={`${pct}% of sessions with feedback received a dislike`}
    >
      {pct}% dislike
    </span>
  );
}

function AuditBar({ summary }: { summary: AuditSummary }) {
  return (
    <div className="audit-bar">
      <div className="audit-stat">
        <span className="audit-label">Sessions</span>
        <span className="audit-value">{summary.total_sessions}</span>
      </div>
      <div className="audit-stat">
        <span className="audit-label">Messages</span>
        <span className="audit-value">{summary.total_messages}</span>
      </div>
      <div className="audit-stat">
        <span className="audit-label">Feedback coverage</span>
        <span className="audit-value">{summary.sessions_with_feedback}</span>
      </div>
      <div className="audit-stat">
        <span className="audit-label">Dislike rate</span>
        <span className="audit-value">
          <DislikeRateBadge rate={summary.dislike_rate} />
        </span>
      </div>
    </div>
  );
}

function SessionRow({ item }: { item: QueryHistoryItem }) {
  const feedbackLabel =
    item.feedback_type === "like"
      ? "👍"
      : item.feedback_type === "dislike"
        ? "👎"
        : item.feedback_type === "mixed"
          ? "±"
          : "—";
  return (
    <tr className="session-row">
      <td className="session-time">{item.time_created.slice(0, 16).replace("T", " ")}</td>
      <td className="session-user">{item.user_id ?? "—"}</td>
      <td className="session-query">{item.first_user_message}</td>
      <td className="session-llm">{item.llm_name ?? "—"}</td>
      <td className="session-turns">{item.conversation_length}</td>
      <td className="session-feedback">{feedbackLabel}</td>
    </tr>
  );
}

export function QueryHistoryPanel() {
  const [audit, setAudit] = useState<AuditSummary | null>(null);
  const [page, setPage] = useState<QueryHistoryPage | null>(null);
  const [pageNum, setPageNum] = useState(0);
  const [feedbackFilter, setFeedbackFilter] = useState<FeedbackFilter>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAuditSummary().then(setAudit).catch(() => undefined);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getQueryHistory(
      {
        page_num: pageNum,
        page_size: PAGE_SIZE,
        feedback_type: feedbackFilter === "all" ? undefined : feedbackFilter,
      },
      { signal: controller.signal },
    )
      .then(setPage)
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : "Failed to load history");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [feedbackFilter, pageNum]);

  const totalPages = page ? Math.ceil(page.total_items / PAGE_SIZE) : 0;

  return (
    <div className="query-history-panel">
      <div className="qh-header">
        <h2>Query History</h2>
        {audit && <AuditBar summary={audit} />}
      </div>

      {audit && audit.top_disliked_queries.length > 0 && (
        <details className="top-dislikes">
          <summary>Top disliked queries ({audit.top_disliked_queries.length})</summary>
          <ol>
            {audit.top_disliked_queries.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ol>
        </details>
      )}

      <div className="qh-controls">
        <label>
          Feedback&nbsp;
          <select
            value={feedbackFilter}
            onChange={(e) => {
              setFeedbackFilter(e.target.value as FeedbackFilter);
              setPageNum(0);
            }}
          >
            <option value="all">All</option>
            <option value="like">Liked</option>
            <option value="dislike">Disliked</option>
          </select>
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <table className="qh-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>User</th>
            <th>Query</th>
            <th>LLM</th>
            <th>Turns</th>
            <th>Feedback</th>
          </tr>
        </thead>
        <tbody>
          {loading && (
            <tr>
              <td colSpan={6} className="qh-loading">
                Loading…
              </td>
            </tr>
          )}
          {!loading && page?.items.length === 0 && (
            <tr>
              <td colSpan={6} className="qh-empty">
                No sessions found.
              </td>
            </tr>
          )}
          {!loading &&
            page?.items.map((item) => <SessionRow key={item.id} item={item} />)}
        </tbody>
      </table>

      {totalPages > 1 && (
        <div className="qh-pagination">
          <button
            type="button"
            disabled={pageNum === 0}
            onClick={() => setPageNum((p) => p - 1)}
          >
            ‹ Prev
          </button>
          <span>
            {pageNum + 1} / {totalPages}
          </span>
          <button
            type="button"
            disabled={pageNum >= totalPages - 1}
            onClick={() => setPageNum((p) => p + 1)}
          >
            Next ›
          </button>
        </div>
      )}
    </div>
  );
}
