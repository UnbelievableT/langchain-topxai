"""Standard integration tests for `ChatTopxAI` against the live API.

Set `TOPXAI_API_KEY` to run them. They make a few dozen small requests.
"""

from __future__ import annotations

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_topxai import ChatTopxAI


class TestChatTopxAIIntegration(ChatModelIntegrationTests):
    """Standard integration tests for `ChatTopxAI`."""

    @property
    def chat_model_class(self) -> type[ChatTopxAI]:
        return ChatTopxAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-6.1-sol"}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True


class TestChatTopxAIDeepSeekIntegration(ChatModelIntegrationTests):
    """The same suite on the current DeepSeek private deployment."""

    @property
    def chat_model_class(self) -> type[ChatTopxAI]:
        return ChatTopxAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "deepseek-flash-Abliterated"}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @pytest.mark.skip(
        reason="Replacement Flash reasoning response awaits live acceptance"
    )
    def test_reasoning_content(self) -> None:
        """Returned reasoning is preserved in additional_kwargs."""
        model = ChatTopxAI(model="deepseek-flash-Abliterated")
        message = model.invoke("What is 17 * 3? Answer with the number only.")
        assert isinstance(message.additional_kwargs.get("reasoning_content"), str)
        full = None
        for chunk in model.stream("What is 12 * 4? Answer with the number only."):
            full = chunk if full is None else full + chunk
        assert full is not None
        assert isinstance(full.additional_kwargs.get("reasoning_content"), str)
