import { useEffect, useState } from "react";
import { discoverTools, getDebugTools } from "../../api";
import type { CatalogServer, ToolDiscoverResult } from "../../types";
import { ToolCatalog } from "../ToolCatalog";

/**
 * Dev-console container for {@link ToolCatalog}.
 *
 * Reads `/api/debug/tools`, whose catalog is already grouped server-side and
 * carries no agent_callable / user_scoped flags, so no badges render here. The
 * `/tools` page uses the same renderer over `/admin/tools` instead.
 */
export function ToolCatalogPanel() {
  const [catalog, setCatalog] = useState<CatalogServer[] | null>(null);
  const [registeredCount, setRegisteredCount] = useState(0);
  const [discovery, setDiscovery] = useState<ToolDiscoverResult | null>(null);

  useEffect(() => {
    let alive = true;
    getDebugTools().then(
      (r) => {
        if (!alive) return;
        setCatalog(r.catalog);
        setRegisteredCount(r.registered.length);
      },
      () => alive && setCatalog([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <ToolCatalog
      servers={catalog}
      registeredCount={registeredCount}
      discovery={discovery}
      onDiscover={(q) => discoverTools(q).then(setDiscovery, () => setDiscovery(null))}
    />
  );
}
