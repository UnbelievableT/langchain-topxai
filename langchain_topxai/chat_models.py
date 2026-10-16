"""TopxAI chat models."""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

import langchain_topxai._content_blocks  # noqa: F401  (registers the topxai translator)
from langchain_topxai._version import __version__

DEFAULT_API_BASE = "https://ai.topxea.com/v1"


def _reasoning_from(container: Any) -> str | None:
    """Return the reasoning text of a message or delta, if the model sent one."""
    if isinstance(container, dict):
        value = container.get("reasoning_content")
        if value is None:
            value = container.get("reasoning")
        return value if isinstance(value, str) else None
    value = getattr(container, "reasoning_content", None)
    if value is None:
        extra = getattr(container, "model_extra", None) or {}
        value = extra.get("reasoning_content", extra.get("reasoning"))
    return value if isinstance(value, str) else None


class ChatTopxAI(BaseChatOpenAI):
    r"""TopxAI chat model integration.

    TopxAI serves Claude, GPT, Grok, GLM, Kimi and DeepSeek models through one
    OpenAI-compatible Chat Completions endpoint, at fixed USD prices per route.
    The model ids and prices are listed at https://ai.topxea.com/pricing.

    Setup:
        Install `langchain-topxai` and set the environment variable
        `TOPXAI_API_KEY` to a key from https://ai.topxea.com/keys.

        ```bash
        pip install -U langchain-topxai
        export TOPXAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Model id, for example `kimi-k3`, `gpt-6.1-sol`, `grok-4.7`,
            `deepseek-flash` or `claude-sonnet-5-5`.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            TopxAI API key. If not passed in, read from env var
            `TOPXAI_API_KEY`.
        base_url:
            API base URL. If not passed in, read from env var
            `TOPXAI_API_BASE`, then defaults to `https://ai.topxea.com/v1`.

    See the full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        ```python
        from langchain_topxai import ChatTopxAI

        model = ChatTopxAI(
            model="kimi-k3",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

    Async:
        ```python
        await model.ainvoke(messages)
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather])
        ai_msg = model_with_tools.invoke("What is the weather like in Paris today?")
        ai_msg.tool_calls
        ```

    Structured output:
        ```python
        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        Whether a model honours JSON Schema output depends on the model. For
        Claude models, structured output is also available through
        `langchain-anthropic`'s `ChatAnthropic` with
        `base_url="https://ai.topxea.com"`.

    Reasoning:
        For models that return their reasoning (for example `deepseek-flash`
        and `kimi-k3`), the text is in
        `ai_msg.additional_kwargs["reasoning_content"]`, also when streaming.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The id of the TopxAI model to call."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("TOPXAI_API_KEY", default=None),
    )
    """TopxAI API key.

    Automatically read from env variable `TOPXAI_API_KEY` if not provided.
    """
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("TOPXAI_API_BASE", default=DEFAULT_API_BASE),
    )
    """TopxAI API base URL.

    Automatically read from env variable `TOPXAI_API_BASE` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-topxai"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "TOPXAI_API_KEY"}

    def _use_responses_api(self, payload: dict) -> bool:  # noqa: ARG002
        """Send every request to Chat Completions.

        `langchain-openai` switches `gpt-6*` models with tools to the Responses
        API, but `BaseChatOpenAI` does not route streaming there, so streamed
        tool calls would break. `validate_environment` rejects explicit
        Responses API settings.
        """
        return False

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "topxai"
        return ls_params

    @model_validator(mode="after")
    def _set_topxai_version(self) -> Self:
        """Set package version in metadata.

        Named uniquely to avoid shadowing `BaseChatOpenAI._set_openai_chat_version`;
        Pydantic replaces same-named validators rather than chaining them.
        """
        self._add_version("langchain-topxai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.use_responses_api or self.output_version == "responses/v1":
            msg = (
                "ChatTopxAI calls the Chat Completions API only. For TopxAI's "
                "Responses API, use langchain-openai's ChatOpenAI with "
                "base_url='https://ai.topxea.com/v1' and use_responses_api=True."
            )
            raise ValueError(msg)
        if not (self.api_key and self.api_key.get_secret_value()):
            # Never let the openai client fall back to OPENAI_API_KEY: that key
            # would be sent to TopxAI or to a custom base URL.
            msg = (
                "TOPXAI_API_KEY must be set, or pass api_key. ChatTopxAI does "
                "not read OPENAI_API_KEY."
            )
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value(),
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "topxai"

        if isinstance(response, dict):
            choices = response.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
        else:
            choices = getattr(response, "choices", None) or []
            message = choices[0].message if choices else None
        if message is not None and rtn.generations:
            reasoning = _reasoning_from(message)
            if reasoning is not None:
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )
        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            message = generation_chunk.message
            if isinstance(message, AIMessageChunk):
                message.response_metadata = {
                    **message.response_metadata,
                    "model_provider": "topxai",
                }
                reasoning = _reasoning_from(choices[0].get("delta") or {})
                if reasoning is not None:
                    message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk
