import { useEffect, useState } from "react";
import type { ToolApprovalView } from "../types";

type Decision = "approve" | "deny";
type DecisionState = "idle" | "submitting" | "decided" | "error";

interface ToolApprovalCardProps {
  approval: ToolApprovalView;
  onDecision: (decision: Decision) => Promise<unknown> | unknown;
}

function secondsUntil(expiresAt: string): number {
  return Math.max(0, Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 1_000));
}

function argumentText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null) return "null";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value) ?? String(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function ToolApprovalCard({ approval, onDecision }: ToolApprovalCardProps) {
  const [state, setState] = useState<DecisionState>("idle");
  const [error, setError] = useState("");
  const [secondsRemaining, setSecondsRemaining] = useState(() =>
    secondsUntil(approval.expires_at),
  );

  useEffect(() => {
    const updateCountdown = () => setSecondsRemaining(secondsUntil(approval.expires_at));
    updateCountdown();
    const timer = window.setInterval(updateCountdown, 1_000);
    return () => window.clearInterval(timer);
  }, [approval.expires_at]);

  const decide = async (decision: Decision) => {
    if (state === "submitting" || state === "decided") return;
    setState("submitting");
    setError("");
    try {
      await onDecision(decision);
      setState("decided");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to submit decision");
      setState("error");
    }
  };

  const locked = state === "submitting" || state === "decided";

  return (
    <section
      className="tool-approval-card"
      aria-label={`Approval required for ${approval.tool_name}`}
    >
      <div className="tool-approval-header">
        <div>
          <p className="tool-approval-eyebrow">Approval required</p>
          <h3>{approval.tool_name}</h3>
        </div>
        <p className="tool-approval-countdown" aria-live="polite">
          Expires in {secondsRemaining} {secondsRemaining === 1 ? "second" : "seconds"}
        </p>
      </div>

      <dl className="tool-approval-arguments">
        {Object.entries(approval.arguments).map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{argumentText(value)}</dd>
          </div>
        ))}
      </dl>

      {error && <p role="alert" className="tool-approval-error">{error}</p>}
      {state === "decided" && <p className="tool-approval-status">Decision submitted</p>}

      <div className="tool-approval-actions">
        <button type="button" onClick={() => void decide("approve")} disabled={locked}>
          Approve
        </button>
        <button type="button" onClick={() => void decide("deny")} disabled={locked}>
          Deny
        </button>
      </div>
    </section>
  );
}
