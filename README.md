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
bedllama start
bedllama stop
bedllama restart
bedllama status
bedllama logs [litellm|front|ollama|server|all]
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
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_HAIKU_4_5`
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_SONNET_4_5`
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_SONNET_4_6`
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_OPUS_4_5`
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_OPUS_4_6`
- `BEDLLAMA_BEDROCK_MODEL_CLAUDE_OPUS_4_7`

`BEDLLAMA_MODELS` accepts a comma-separated list.
