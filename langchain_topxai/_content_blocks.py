"""Standard content blocks for messages from TopxAI.

TopxAI answers in the OpenAI Chat Completions format, plus a
`reasoning_content` field for models that return their reasoning. The
translator reuses langchain-core's OpenAI translator and puts a reasoning
block first. It is registered under the `topxai` provider name, which
`ChatTopxAI` writes to `response_metadata["model_provider"]`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators import openai as openai_blocks
from langchain_core.messages.block_translators import register_translator


def _reasoning_blocks(message: AIMessage) -> list[types.ContentBlock]:
    reasoning = message.additional_kwargs.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return [{"type": "reasoning", "reasoning": reasoning}]
    return []


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a TopxAI message."""
    return [*_reasoning_blocks(message), *openai_blocks.translate_content(message)]


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a TopxAI message chunk.

    The OpenAI translator numbers tool-call chunks by their position among the
    tool calls (0, 1, ...), while langchain-core numbers the other blocks of a
    stream in order of appearance. When a model writes text or reasoning before
    calling a tool, both are 0 and the blocks merge. Tool-call chunks get their
    own string index instead, one per tool call.
    """
    blocks: list[types.ContentBlock] = [
        *_reasoning_blocks(message),
        *openai_blocks.translate_content_chunk(message),
    ]
    for block in blocks:
        if block["type"] == "tool_call_chunk" and isinstance(block.get("index"), int):
            block["index"] = f"lc_tc_{block['index']}"
    return blocks


register_translator("topxai", translate_content, translate_content_chunk)
