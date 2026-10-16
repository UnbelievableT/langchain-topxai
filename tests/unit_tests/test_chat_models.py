"""Unit tests for `ChatTopxAI`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel as PydanticModel
from pydantic import SecretStr

from langchain_topxai import ChatTopxAI, __version__
from langchain_topxai.chat_models import DEFAULT_API_BASE

MODEL_NAME = "kimi-k3"


class MockOpenAIResponse(BaseModel):
    choices: list
    error: None = None

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: str | bool = True,
        context: dict[str, Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        choices_list = []
        for choice in self.choices:
            if isinstance(choice.message, ChatCompletionMessage):
                message_dict = choice.message.model_dump()
                if "model_extra" in message_dict:
                    message_dict.update(message_dict["model_extra"])
            else:
                message_dict = {"role": "assistant", "content": choice.message.content}
                if hasattr(choice.message, "reasoning_content"):
                    message_dict["reasoning_content"] = choice.message.reasoning_content
            choices_list.append({"message": message_dict, "finish_reason": "stop"})
        return {"choices": choices_list, "error": self.error}


class TestChatTopxAIUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatTopxAI`."""

    @property
    def chat_model_class(self) -> type[ChatTopxAI]:
        return ChatTopxAI

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "TOPXAI_API_KEY": "api_key",
                "TOPXAI_API_BASE": "https://example.invalid/v1",
            },
            {"model": MODEL_NAME},
            {
                "api_key": "api_key",
                "api_base": "https://example.invalid/v1",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME, "api_key": "api_key"}


def test_default_base_url() -> None:
    model = ChatTopxAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
    assert model.api_base == DEFAULT_API_BASE == "https://ai.topxea.com/v1"
    assert str(model.root_client.base_url) == "https://ai.topxea.com/v1/"


def test_missing_key_on_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPXAI_API_KEY", raising=False)
    monkeypatch.delenv("TOPXAI_API_BASE", raising=False)
    with pytest.raises(ValueError, match="TOPXAI_API_KEY must be set"):
        ChatTopxAI(model=MODEL_NAME)


@pytest.mark.parametrize(
    "base_url", ["https://proxy.example/v1", "https://ai.topxea.com/v1/"]
)
def test_never_falls_back_to_openai_key(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.delenv("TOPXAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-be-used")
    with pytest.raises(ValueError, match="does not read OPENAI_API_KEY"):
        ChatTopxAI(model=MODEL_NAME, base_url=base_url)


def test_custom_base_url_reaches_both_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPXAI_API_BASE", "https://proxy.example/v1")
    model = ChatTopxAI(
        model=MODEL_NAME,
        api_key=SecretStr("api_key"),
        default_headers={"X-Test": "1"},
    )
    assert str(model.root_client.base_url) == "https://proxy.example/v1/"
    assert str(model.root_async_client.base_url) == "https://proxy.example/v1/"
    assert model.root_client.api_key == "api_key"
    assert model.root_client.default_headers["X-Test"] == "1"
    assert model.root_async_client.default_headers["X-Test"] == "1"


def test_responses_api_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="Chat Completions API only"):
        ChatTopxAI(
            model="gpt-6.1-sol", api_key=SecretStr("api_key"), use_responses_api=True
        )
    with pytest.raises(ValueError, match="Chat Completions API only"):
        ChatTopxAI(
            model="gpt-6.1-sol",
            api_key=SecretStr("api_key"),
            output_version="responses/v1",
        )


def test_package_version_in_metadata() -> None:
    model = ChatTopxAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
    assert model.metadata is not None
    assert model.metadata["lc_versions"]["langchain-topxai"] == __version__


def test_ls_params_and_llm_type() -> None:
    model = ChatTopxAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
    assert model._llm_type == "chat-topxai"
    assert model._get_ls_params()["ls_provider"] == "topxai"
    assert model.lc_secrets == {"api_key": "TOPXAI_API_KEY"}


def test_reasoning_content_in_result() -> None:
    model = ChatTopxAI(model="deepseek-flash", api_key=SecretStr("api_key"))
    mock_message = MagicMock()
    mock_message.content = "51"
    mock_message.reasoning_content = "17 times 3 is 51."
    mock_message.role = "assistant"
    response = MockOpenAIResponse(
        choices=[MagicMock(message=mock_message)],
        error=None,
    )
    result = model._create_chat_result(response)
    message = result.generations[0].message
    assert message.content == "51"
    assert message.additional_kwargs.get("reasoning_content") == "17 times 3 is 51."
    assert message.response_metadata["model_provider"] == "topxai"


def test_reasoning_content_from_dict_response() -> None:
    model = ChatTopxAI(model="kimi-k3", api_key=SecretStr("api_key"))
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "51",
                    "reasoning_content": "Multiply.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    result = model._create_chat_result(response)
    assert result.generations[0].message.additional_kwargs["reasoning_content"] == (
        "Multiply."
    )


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_from_real_chat_completion(field: str) -> None:
    model = ChatTopxAI(model="GLM-5.3-Abliterated", api_key=SecretStr("api_key"))
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "GLM-5.3-Abliterated",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "51",
                        field: "Seventeen threes.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    )
    result = model._create_chat_result(completion)
    message = result.generations[0].message
    assert message.content == "51"
    assert message.additional_kwargs["reasoning_content"] == "Seventeen threes."
    assert message.response_metadata["model_provider"] == "topxai"


def test_no_reasoning_key_when_absent() -> None:
    model = ChatTopxAI(model="gpt-6.1-sol", api_key=SecretStr("api_key"))
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
    }
    result = model._create_chat_result(response)
    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_content_in_stream(field: str) -> None:
    model = ChatTopxAI(model="deepseek-flash", api_key=SecretStr("api_key"))
    chunk = {
        "choices": [{"delta": {"content": "", field: "Thinking..."}}],
    }
    generation = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
    assert generation is not None
    assert isinstance(generation.message, AIMessageChunk)
    assert generation.message.additional_kwargs["reasoning_content"] == "Thinking..."
    assert generation.message.response_metadata["model_provider"] == "topxai"


def test_stream_chunk_without_reasoning() -> None:
    model = ChatTopxAI(model="gpt-6.1-sol", api_key=SecretStr("api_key"))
    chunk = {"choices": [{"delta": {"content": "Hi"}}]}
    generation = model._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)
    assert generation is not None
    assert "reasoning_content" not in generation.message.additional_kwargs


def test_tools_on_gpt6_models_stay_on_chat_completions() -> None:
    class Add(PydanticModel):
        """Add two numbers."""

        a: int
        b: int

    model = ChatTopxAI(model="gpt-6.1-sol", api_key=SecretStr("api_key"))
    bound = model.bind_tools([Add])
    assert isinstance(bound, RunnableBinding)
    payload = model._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    assert "messages" in payload
    assert "input" not in payload
    assert payload["model"] == "gpt-6.1-sol"


def test_content_blocks_carry_reasoning_and_tool_calls() -> None:
    message = AIMessage(
        content="Calling the tool.",
        additional_kwargs={"reasoning_content": "I should add."},
        tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "call_1"}],
        response_metadata={"model_provider": "topxai"},
    )
    blocks = message.content_blocks
    assert [block["type"] for block in blocks] == ["reasoning", "text", "tool_call"]
    assert blocks[0] == {"type": "reasoning", "reasoning": "I should add."}
    assert dict(blocks[2])["name"] == "add"


def test_chunk_content_blocks_carry_reasoning() -> None:
    chunk = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "Thinking"},
        response_metadata={"model_provider": "topxai"},
    )
    assert chunk.content_blocks[0] == {"type": "reasoning", "reasoning": "Thinking"}


def test_tool_call_chunk_index_does_not_collide_with_text() -> None:
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "add", "args": '{"a"', "id": "call_1", "index": 0},
        ],
        response_metadata={"model_provider": "topxai"},
    )
    blocks = chunk.content_blocks
    assert [dict(block).get("index") for block in blocks] == ["lc_tc_0"]


def test_structured_output_defaults_to_json_schema() -> None:
    class City(PydanticModel):
        name: str

    model = ChatTopxAI(model="claude-sonnet-5-5", api_key=SecretStr("api_key"))
    structured = model.with_structured_output(City)
    assert isinstance(structured, RunnableSequence)
    binding = structured.first
    assert isinstance(binding, RunnableBinding)
    assert binding.kwargs["response_format"] is City
    assert "tools" not in binding.kwargs
    assert "tool_choice" not in binding.kwargs
