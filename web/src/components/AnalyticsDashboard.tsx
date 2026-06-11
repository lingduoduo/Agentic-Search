import type { BreakdownAnalytics, BreakdownItem } from "../types";

interface AnalyticsDashboardProps {
  byLLM: BreakdownAnalytics | null;
  byPersona: BreakdownAnalytics | null;
  byFlow: BreakdownAnalytics | null;
}

const PALETTE = [
  "#0f766e",
  "#0891b2",
  "#7c3aed",
  "#b45309",
  "#be185d",
  "#15803d",
  "#1d4ed8",
  "#9f1239",
];

function BarChart({ items }: { items: BreakdownItem[] }) {
  const max = Math.max(...items.map((i) => i.session_count), 1);
  const barH = 28;
  const gap = 8;
  const labelW = 120;
  const countW = 36;
  const chartW = 260;
  const totalH = items.length * (barH + gap) - gap;

  return (
    <svg
      width={labelW + chartW + countW + 16}
      height={totalH}
      role="img"
      aria-label="Usage bar chart"
    >
      {items.map((item, idx) => {
        const y = idx * (barH + gap);
        const barW = Math.max(2, (item.session_count / max) * chartW);
        const color = PALETTE[idx % PALETTE.length];
        return (
          <g key={item.label}>
            <text
              x={labelW - 8}
              y={y + barH / 2}
              dominantBaseline="middle"
              textAnchor="end"
              className="analytics-bar-label"
            >
              {item.label.length > 14
                ? `${item.label.slice(0, 13)}…`
                : item.label}
            </text>
            <rect
              x={labelW}
              y={y}
              width={barW}
              height={barH}
              rx={4}
              fill={color}
              opacity={0.85}
            />
            <text
              x={labelW + barW + 6}
              y={y + barH / 2}
              dominantBaseline="middle"
              className="analytics-bar-count"
            >
              {item.session_count}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function BreakdownPanel({
  title,
  data,
}: {
  title: string;
  data: BreakdownAnalytics | null;
}) {
  return (
    <div className="analytics-panel">
      <div className="analytics-panel-header">
        <h3>{title}</h3>
        {data && (
          <span className="analytics-total">
            {data.total_sessions} total
          </span>
        )}
      </div>
      {!data ? (
        <p className="analytics-empty">Loading…</p>
      ) : data.items.length === 0 ? (
        <p className="analytics-empty">No sessions in this window.</p>
      ) : (
        <BarChart items={data.items} />
      )}
    </div>
  );
}

export function AnalyticsDashboard({
  byLLM,
  byPersona,
  byFlow,
}: AnalyticsDashboardProps) {
  return (
    <section className="analytics-dashboard" aria-label="Usage analytics">
      <h2 className="analytics-heading">Usage breakdown</h2>
      <div className="analytics-grid">
        <BreakdownPanel title="By LLM" data={byLLM} />
        <BreakdownPanel title="By agent / persona" data={byPersona} />
        <BreakdownPanel title="By flow type" data={byFlow} />
      </div>
    </section>
  );
}
