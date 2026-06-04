# bedllama

`bedllama` is the tracked command for this stack.

It consolidates the current local stack management flow:

- launch LiteLLM with your existing AWS credentials
- generate a dedicated LiteLLM config for this stack
- run one integrated `bedllama` server that exposes both:
  - an OpenAI-compatible endpoint
  - an Ollama-compatible endpoint
- manage the stack as background processes with PID and log files

## Current commands

```bash
bedllama init      # scaffold a config.jsonc in the current directory
bedllama start
bedllama stop
bedllama restart
bedllama status
bedllama logs [litellm|front|ollama|server|all]
bedllama legend    # explain log field acronyms

# systemd
bedllama run       # run the full stack in the foreground (used by systemd ExecStart)
bedllama install   # install ~/.config/systemd/user/bedllama.service and reload daemon
bedllama uninstall # disable and remove the systemd user service

# integrations
bedllama vscode    # update VS Code chatLanguageModels.json with live Bedrock model data
```

## Install for local use

From this repo:

```bash
npm install
npm link
```

Then use:

```bash
bedllama start
```

`config.jsonc` is parsed as JSONC: comments and trailing commas are accepted.

## LiteLLM config generation

`bedllama` writes its own LiteLLM config before startup instead of relying on
or mutating your global `~/.config/litellm/config.yaml`.

Default generated path:

```bash
~/.cache/bedllama/litellm.config.yaml
```

## Runtime state

`bedllama` keeps its runtime files under:

```bash
~/.cache/bedllama
```

This includes:

- generated LiteLLM config
- `litellm.pid` and `bedllama.pid`
- `litellm.log` and `bedllama.log`

## Build and typecheck

```bash
npm run build
npm run typecheck
```

## Environment overrides

- `BEDLLAMA_STATE_DIR`
- `BEDLLAMA_LITELLM_PROCESS`
- `BEDLLAMA_SERVER_PROCESS`
- `BEDLLAMA_FRONT_HOST`
- `BEDLLAMA_OLLAMA_HOST`
- `BEDLLAMA_LITELLM_PORT`
- `BEDLLAMA_FRONT_PORT`
- `BEDLLAMA_OLLAMA_PORT`
- `BEDLLAMA_API_KEY`
- `BEDLLAMA_LOG`
- `BEDLLAMA_AWS_PROFILE`
- `BEDLLAMA_AWS_REGION`
- `BEDLLAMA_LITELLM_BIN`
- `BEDLLAMA_LITELLM_CONFIG`
- `BEDLLAMA_OLLAMA_VERSION`
- `BEDLLAMA_OLLAMA_DEFAULT_MODEL`
- `BEDLLAMA_MODELS`
- `BEDLLAMA_POSTGRES_PORT`
- `BEDLLAMA_POSTGRES_PASSWORD`
- `BEDLLAMA_POSTGRES_CONTAINER`
`BEDLLAMA_MODELS` accepts a comma-separated list.
Available models are discovered dynamically from the Bedrock API — no per-model env vars needed.

Anthropic models that expose extended output limits are also surfaced with
additional shim IDs for larger context windows:

- `claude-sonnet-4-6:latest` for the base 200K entry
- `claude-sonnet-4-6-400k:latest` for the 400K variant
- `claude-sonnet-4-6-1m:latest` for the 1M variant

These map back to the same upstream Bedrock model; bedllama rewrites the shim
name before forwarding to LiteLLM.

## Admin UI and spend tracking

The LiteLLM admin UI gives you a web dashboard to inspect usage, costs per
model/key/user, and manage virtual API keys.

### Prerequisites

- **docker** or **podman** must be on your `PATH` — bedllama starts a
  `postgres:16` container automatically.
- The `prisma` Python package must be installable via `uv` — bedllama handles
  this on first start.

### Enable

Set `adminUi: true` in `config.jsonc` (run `bedllama init` to create one):

```jsonc
{
  "adminUi": true,
  "adminUiUsername": "admin",       // default: admin
  "adminUiPassword": "bedllama-admin" // change this
}
```

### What bedllama does on start

1. Starts a `postgres:16` container named `bedllama-postgres` with a named
   volume (`bedllama-postgres-data`) for persistence.
2. Waits for postgres to be fully ready (`pg_isready`).
3. Runs `prisma db push` to create/migrate the LiteLLM schema (idempotent).
4. Starts LiteLLM with `database_url` and `store_model_in_db: true` injected
   into the generated config.
5. Prints the UI URL on success.

### Access the UI

```
http://127.0.0.1:4001/ui
```

Log in with the `adminUiUsername` / `adminUiPassword` from your config.

### Spend tracking

Spend is tracked automatically for every request once the DB is connected.
LiteLLM logs each call to `LiteLLMSpendLogs` and aggregates by key, user, and
team. The `x-litellm-response-cost` header is forwarded on every response so
clients can see per-request cost without hitting the UI.

### `leanProxy` — latency vs full logging

By default (`leanProxy: false`) every request is fully logged: the complete
prompt text and response are written to the `LiteLLMSpendLogs` table so the
admin UI shows the full request history.

For long-context responses this adds measurable overhead:

| LiteLLM behaviour | default | `leanProxy: true` |
|---|---|---|
| `store_prompts_in_spend_logs` | `true` — full text stored | `false` — text omitted |
| `disable_streaming_logging` | `false` — per-chunk handlers run | `true` — handlers skipped |
| Token counts + cost tracking | ✅ | ✅ |
| Admin UI works | ✅ | ✅ |
| Prompt history in UI | ✅ | ❌ |

`proxy_batch_write_at: 60` (flush DB every 60 s instead of 10 s) is always
applied when `adminUi: true` — no data loss, just less frequent flushing.

To enable lean mode, add `"leanProxy": true` to your `config.jsonc`:

```jsonc
{
  "adminUi": true,
  "leanProxy": true
}
```

Has no effect when `adminUi: false` — that mode is always fully lean
(`disable_spend_logs: true` + `disable_streaming_logging: true`).

### PostgreSQL options

| config.jsonc key   | env var                    | default            |
|--------------------|----------------------------|--------------------|
| `postgresPort`     | `BEDLLAMA_POSTGRES_PORT`   | `5432`             |
| `postgresPassword` | `BEDLLAMA_POSTGRES_PASSWORD` | `bedllama`       |
| —                  | `BEDLLAMA_POSTGRES_CONTAINER` | `bedllama-postgres` |

The postgres password is passed to the container via a temporary env file
(`~/.cache/bedllama/postgres.env`, mode `0600`) so it never appears in
`ps aux`. The generated `litellm.config.yaml` is also written `0600`.

### Stop behaviour

`bedllama stop` stops the LiteLLM and bedllama processes **and** stops the
postgres container. The data volume is preserved — spend history survives
restarts.
