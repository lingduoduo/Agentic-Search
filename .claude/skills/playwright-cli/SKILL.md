---
name: playwright-cli
description: Automate browser interactions, test web pages and work with Playwright tests. Use when opening URLs, clicking, typing, taking screenshots, or driving any browser interaction from the CLI.
allowed-tools: Bash(playwright-cli:*) Bash(npx:*) Bash(npm:*)
---

# Browser Automation with playwright-cli

## Quick start

```bash
playwright-cli open https://example.com/
playwright-cli snapshot
playwright-cli fill e5 "search query" --submit
playwright-cli screenshot --filename=result.png
playwright-cli close
```

## Open parameters

```bash
playwright-cli open --browser=chrome
playwright-cli open --persistent                          # persistent profile (in-memory by default)
playwright-cli open --profile=/path/to/profile            # custom profile dir
playwright-cli attach --cdp=chrome                        # attach to running Chrome
playwright-cli attach --cdp=http://localhost:9222
```

## Core commands

```bash
playwright-cli goto https://example.com
playwright-cli snapshot                                   # capture page state + refs
playwright-cli click e3
playwright-cli fill e5 "text" --submit                    # fill + press Enter
playwright-cli type "text"
playwright-cli press Enter
playwright-cli hover e4
playwright-cli select e9 "option-value"
playwright-cli check e12
playwright-cli eval "document.title"
playwright-cli eval "el => el.textContent" e5
playwright-cli resize 1920 1080
playwright-cli close
```

## Targeting elements

Use refs from snapshot output (`e3`, `e15`, etc.) — preferred. Also accepts:

```bash
playwright-cli click "#main > button.submit"              # CSS selector
playwright-cli click "getByRole('button', { name: 'Submit' })"
playwright-cli click "getByTestId('submit-button')"
```

## Navigation & keyboard

```bash
playwright-cli go-back
playwright-cli go-forward
playwright-cli reload
playwright-cli press ArrowDown
playwright-cli keydown Shift && playwright-cli keyup Shift
```

## Screenshots & PDF

```bash
playwright-cli screenshot
playwright-cli screenshot --filename=page.png
playwright-cli screenshot e5                              # element screenshot
playwright-cli pdf --filename=page.pdf
```

## Tabs

```bash
playwright-cli tab-new https://example.com/other
playwright-cli tab-list
playwright-cli tab-select 0
playwright-cli tab-close 2
```

## Storage

```bash
playwright-cli state-save auth.json
playwright-cli state-load auth.json
playwright-cli cookie-list
playwright-cli cookie-get session_id
playwright-cli cookie-set session_id abc123 --domain=example.com --httpOnly --secure
playwright-cli localstorage-get theme
playwright-cli localstorage-set theme dark
```

## Network & DevTools

```bash
playwright-cli requests
playwright-cli request 5
playwright-cli route "https://api.example.com/**" --body='{"mock": true}'
playwright-cli console
playwright-cli tracing-start && playwright-cli tracing-stop
playwright-cli video-start video.webm && playwright-cli video-stop
```

## Raw output (for piping)

```bash
playwright-cli --raw eval "document.title"
playwright-cli --raw snapshot > before.yml
playwright-cli --raw eval "JSON.stringify([...document.querySelectorAll('a')].map(a=>a.href))" > links.json
```

## Session management

```bash
playwright-cli -s=mysession open example.com --persistent
playwright-cli -s=mysession click e6
playwright-cli -s=mysession close
playwright-cli list
playwright-cli close-all
playwright-cli kill-all
```

## UI review / design feedback

```bash
playwright-cli open https://example.com
playwright-cli show --annotate        # user draws boxes + adds comments; you receive annotated screenshot
```
