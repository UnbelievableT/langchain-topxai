"""LangChain integration for TopxAI."""

from langchain_topxai._version import __version__
from langchain_topxai.chat_models import ChatTopxAI

__all__ = [
    "ChatTopxAI",
    "__version__",
]
