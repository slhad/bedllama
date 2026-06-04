# bedllama — agent & contributor notes

## Platform support

**bedllama targets Linux and macOS. Windows support is partial — WSL2 is recommended.**

Node.js ≥ 22 is required. The following table summarises the Windows situation:

| area | status | notes |
|---|---|---|
| `sleep()` | ✅ fixed | Was `Atomics.wait` (needed `SharedArrayBuffer` flags). Now `setTimeout`-based async. |
| `litellm-proxy` check | ✅ fixed | Now uses `uv tool list` (cross-platform) with binary fallback. |
| `detached` process spawn | ✅ | Node 22 supports `detached: true` on Windows. |
| `taskkill` for stop | ✅ | Already handled for `win32`. |
| `SIGTERM` / `SIGINT` handlers | ⚠️ | Not real OS signals on Windows; Node emulates them. Graceful shutdown may not fire for child processes. |
| LiteLLM proxy mode | ⚠️ | LiteLLM's proxy (uvicorn/asyncio) has known issues on Windows. May work but is untested. |
| Shebang `#!/usr/bin/env node` | ⚠️ | Ignored on Windows. Use `npm link` (creates a `.cmd` wrapper) or run via `node dist/bedllama.js` directly. |

**Recommended on Windows: use WSL2 and treat it as Linux.**

---

## Architecture

```
client (Cursor / Continue / any OpenAI or Ollama client)
  │
  ├─ OpenAI-compatible  →  bedllama front  (:4000/v1/...)
  │                              │
  └─ Ollama-compatible  →  bedllama ollama (:11434/api/...)
                                 │
                         LiteLLM proxy (:4001)   ← local loopback, ~5-20ms overhead
                                 │
                         AWS Bedrock (eu-west-3 by default)
```

### Overhead chain (reference timings, home network, 98K context)

```
proc (bedllama JSON processing)  ~2–20ms
LiteLLM local loopback           ~5–20ms   (included in ttfb)
AWS Bedrock prefill (98K tokens) ~2000–4500ms
AWS Bedrock generation           ~2000–3000ms (stream)
─────────────────────────────────────────────
total                            ~4–8s typical
```

`proc` is the only part bedllama controls. If `proc` spikes above ~100ms on a
corp network, a proxy or DPI appliance is inspecting the payload.

---

## Log field reference

Run `bedllama legend` for the full reference. Quick summary:

| field | meaning |
|---|---|
| `read=` | time to receive request body from client |
| `ttfb=` | time-to-first-byte from LiteLLM (includes Bedrock latency) |
| `stream=` | time to drain response chunks to client |
| `proc=` | pure bedllama processing time (`read + stream`) |
| `total=` | wall-clock for the full request |
| `prompt=` / `completion=` | token counts |

---

## Running & restarting bedllama

If the systemd user service is installed (`~/.config/systemd/user/bedllama.service`),
**always use `systemctl` to start, stop, or restart bedllama** — do not run
`bedllama serve` or `bedllama start` directly. A stray manual process will hold
port 4000 and cause the service to crash with `EADDRINUSE`.

```bash
# Check whether the service is installed
systemctl --user status bedllama

# Preferred: restart via systemctl
systemctl --user restart bedllama

# Start / stop
systemctl --user start bedllama
systemctl --user stop bedllama

# Follow logs
journalctl --user -fu bedllama
```

If port 4000 is already in use by a stray process, kill it first:
```bash
lsof -i :4000          # find the PID
kill <pid>             # kill the stray process
systemctl --user start bedllama
```

Only use `bedllama serve` / `bedllama start` directly when the systemd service
is **not** installed.

---

## Development

```bash
npm install
npm run build       # compile TypeScript → dist/
npm run typecheck   # type-check without emitting
npm link            # install as global `bedllama` binary
```

Runtime state lives in `~/.cache/bedllama/` (overridable via `BEDLLAMA_STATE_DIR`).

## Dependencies

- Node.js ≥ 20
- `litellm[proxy]` installed via `uv tool install 'litellm[proxy]'`
- AWS CLI (`aws`) with a configured profile that has Bedrock access
- Valid AWS credentials (`bedllama start` checks these before launching)
