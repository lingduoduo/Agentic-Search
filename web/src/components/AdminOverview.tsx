import {
  Activity,
  Boxes,
  Braces,
  DatabaseZap,
  KeyRound,
  LineChart,
  ShieldCheck,
  UsersRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { AdminSurfaceKey, AdminSurfaceSummary } from "../types";

interface AdminOverviewProps {
  summary: AdminSurfaceSummary;
}

const iconByKey: Record<AdminSurfaceKey, LucideIcon> = {
  connectors: Boxes,
  indexing: DatabaseZap,
  access: UsersRound,
  auth: KeyRound,
  models: Braces,
  tools: ShieldCheck,
  analytics: LineChart,
  enterprise: Activity,
};

export function AdminOverview({ summary }: AdminOverviewProps) {
  return (
    <section className="admin-overview" aria-label="Admin and observability">
      <div className="admin-overview-header">
        <div>
          <h2>Admin overview</h2>
          <p>Operational health across retrieval, governance, and enterprise controls.</p>
        </div>
        <div className="admin-health">
          <span>{summary.healthLabel}</span>
          <strong>{summary.healthScore}%</strong>
        </div>
      </div>

      <div className="admin-metrics" aria-label="Operational metrics">
        {summary.metrics.map((metric) => (
          <div className="admin-metric" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <small>{metric.detail}</small>
          </div>
        ))}
      </div>

      <div className="admin-grid">
        {summary.sections.map((section) => {
          const Icon = iconByKey[section.key];
          return (
            <article className="admin-card" key={section.key}>
              <div className="admin-card-title">
                <Icon size={17} />
                <h3>{section.title}</h3>
                <span data-tone={section.tone}>{section.status}</span>
              </div>
              <p>{section.description}</p>
              <ul>
                {section.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          );
        })}
      </div>
    </section>
  );
}
