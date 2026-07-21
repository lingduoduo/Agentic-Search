# Plan: Document registry inspection endpoints in tool-engine.md

1. Accuracy-sync existing tool-engine.md claims against `src/tools/` +
   `src/agents/tool/`. → verify: tool names, TF-IDF router, symbols all match (done — all accurate).
2. Confirm the debug endpoints' handlers + return shapes in
   `src/internal/servers/web/debug_router.py`. → verify: `GET /tools`, `POST /tools/discover`.
3. Add "Inspecting the registry" section documenting both endpoints. → verify:
   names + return shapes match handlers; no other content touched.
4. Commit to the existing PR branch, push. → verify: `git push` clean.
