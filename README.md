# langchain-topxai

LangChain chat model for [TopxAI](https://ai.topxea.com), an API gateway that serves Claude, GPT, Grok, GLM, Kimi and DeepSeek models through one OpenAI-compatible endpoint at fixed USD prices per route.

## Installation

```bash
pip install -U langchain-topxai
```

Create a key at https://ai.topxea.com/keys and set it:

```bash
export TOPXAI_API_KEY="your-api-key"
```

## Chat models

```python
from langchain_topxai import ChatTopxAI

model = ChatTopxAI(model="kimi-k3", temperature=0)
model.invoke("Sing a ballad of LangChain.")
```

`model` takes any text model id from https://ai.topxea.com/pricing, for example `claude-sonnet-5-5`, `gpt-6.1-sol`, `grok-4.7`, `kimi-k3`, `deepseek-v4-pro` or `deepseek-flash`. `ChatTopxAI` builds on `langchain-openai`'s `BaseChatOpenAI` and always calls the Chat Completions API: streaming, async calls, tool calling, structured output and token usage work as they do there. For TopxAI's Responses API, use `ChatOpenAI` from `langchain-openai` with `base_url="https://ai.topxea.com/v1"` and `use_responses_api=True`.

Models that return their reasoning (for example `deepseek-flash` and `kimi-k3`) put it in `additional_kwargs["reasoning_content"]`, also when streaming:

```python
msg = ChatTopxAI(model="deepseek-flash").invoke("What is 17 * 3?")
msg.additional_kwargs.get("reasoning_content")
```

`with_structured_output` uses JSON Schema by default (`method="json_schema"`, as in `ChatOpenAI`). Keep the default for the Claude models (`claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1`): they do not accept a forced tool call, so `method="function_calling"` fails for them.

## Configuration

| Argument | Environment variable | Default |
| --- | --- | --- |
| `api_key` | `TOPXAI_API_KEY` | none; required. `OPENAI_API_KEY` is never read. |
| `base_url` | `TOPXAI_API_BASE` | `https://ai.topxea.com/v1` |

## Links

- Guide: https://ai.topxea.com/docs/use-topxai-with-langchain
- Models and prices: https://ai.topxea.com/pricing
- API reference: https://ai.topxea.com/docs/api-reference

## Development

```bash
make tests              # unit tests, no network
make integration_tests  # LangChain standard integration tests, needs TOPXAI_API_KEY
make lint type
```

MIT licensed. Maintained by TopXEA, the operator of TopxAI.
