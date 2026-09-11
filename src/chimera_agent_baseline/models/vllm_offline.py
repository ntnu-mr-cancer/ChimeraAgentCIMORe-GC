"""vLLM offline model backend.

Wraps :class:`vllm.LLM` as a LangChain :class:`BaseChatModel` so it can be
used directly with ``bind_tools()`` and the ReAct graph — no HTTP server
needed.

Tool-call parsing uses vLLM's native ``ToolParserManager``, meaning this
backend does not need to know how individual model tool-call formats are
implemented.

Requires::

    pip install vllm
"""

import json
import logging
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, model_validator

log = logging.getLogger(__name__)


class ChatVLLM(BaseChatModel):
    """LangChain chat model backed by vLLM offline inference.

    Supports ``bind_tools()`` and structured ``tool_calls`` in responses,
    which is required by the LangGraph ReAct loop.

    Tool-call parsing is delegated entirely to vLLM's native
    :class:`ToolParserManager`.

    Usage::

        from vllm import LLM, SamplingParams
        from chimera_agent_baseline.models.vllm_offline import ChatVLLM

        llm = LLM(model="./model", dtype="auto")

        model = ChatVLLM(
            llm=llm,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=4096,
            ),
            tool_parser="lfm2",
        )

        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(messages)
    """

    llm: Any = Field(exclude=True)
    """A :class:`vllm.LLM` instance."""

    sampling_params: Any = Field(exclude=True)
    """A :class:`vllm.SamplingParams` instance."""

    tokenizer: Any = Field(default=None, exclude=True)
    """Tokenizer used for decoding model output."""

    tool_parser: str = Field(default="gemma4")
    """Name of the vLLM tool parser, e.g. ``lfm2``, ``gemma4``, ``hermes``."""

    bound_tools: list[dict] = Field(default_factory=list)
    """OpenAI-format tool schemas, set via :meth:`bind_tools`."""

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _resolve_tokenizer(self) -> "ChatVLLM":
        """Resolve the tokenizer from the vLLM LLM instance."""
        if self.tokenizer is None:
            self.tokenizer = self.llm.get_tokenizer()
        return self

    # -------------------------------------------------------------------------
    # LangChain interface
    # -------------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "vllm-offline"

    def bind_tools(self, tools: list, **kwargs: Any) -> "ChatVLLM":
        """Return a copy of this model with tools bound."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [
            convert_to_openai_tool(tool)
            for tool in tools
        ]

        return self.model_copy(
            update={
                "bound_tools": formatted,
                "tool_parser": self.tool_parser,
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using vLLM offline inference."""

        msg_dicts = _to_openai_messages(messages)
        tools = self.bound_tools or None

        outputs = self.llm.chat(
            messages=msg_dicts,
            sampling_params=self.sampling_params,
            tools=tools,
            use_tqdm=False,
        )

        output = outputs[0].outputs[0]

        # Decode WITH special tokens.
        #
        # This is important for model-specific parsers such as LFM2, whose
        # parser expects tokens like:
        #
        #   <|tool_call_start|>
        #   <|tool_call_end|>
        #
        # The native vLLM parser is responsible for interpreting these.
        full_text = self.tokenizer.decode(
            output.token_ids,
            skip_special_tokens=False,
        )

        parsed = _parse_tool_calls(
            text=full_text,
            parser=self.tool_parser,
            tokenizer=self.tokenizer,
            tools=tools,
        )

        if parsed["tool_calls"]:
            lc_tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["arguments"],
                    "id": str(uuid4()),
                    "type": "tool_call",
                }
                for tc in parsed["tool_calls"]
            ]

            message = AIMessage(
                content=parsed["content"] or "",
                tool_calls=lc_tool_calls,
            )
        else:
            # output.text is vLLM's normal decoded output without the
            # special tokens used internally by some tool parsers.
            message = AIMessage(
                content=parsed["content"] or output.text,
            )

        return ChatResult(
            generations=[
                ChatGeneration(message=message)
            ]
        )
    


# -----------------------------------------------------------------------------
# Message conversion
# -----------------------------------------------------------------------------


def _to_openai_messages(
    messages: list[BaseMessage],
) -> list[dict]:
    """Convert LangChain messages to OpenAI-format dictionaries for vLLM."""

    result = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append(
                {
                    "role": "system",
                    "content": msg.content,
                }
            )

        elif isinstance(msg, HumanMessage):
            result.append(
                {
                    "role": "user",
                    "content": msg.content,
                }
            )

        elif isinstance(msg, AIMessage):
            entry: dict = {
                "role": "assistant",
            }

            if msg.tool_calls:
                entry["content"] = None

                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                entry["content"] = msg.content

            result.append(entry)

        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )

        else:
            result.append(
                {
                    "role": "user",
                    "content": str(msg.content),
                }
            )

    return result


# -----------------------------------------------------------------------------
# Tool-call parsing
# -----------------------------------------------------------------------------


def _parse_tool_calls(
    text: str,
    parser: str,
    tokenizer: Any,
    tools: list[dict] | None = None,
) -> dict:
    """Parse tool calls using vLLM's native ToolParserManager.

    The parser implementation is resolved dynamically through vLLM's
    ToolParserManager. This means this wrapper does not need to know where
    individual parser implementations live.

    Args:
        text:
            Complete model output, including special tokens.

        parser:
            Name registered with vLLM's ToolParserManager, e.g. ``lfm2``,
            ``gemma4``, ``hermes``, ``llama3_json``, etc.

        tokenizer:
            The tokenizer used by the vLLM model.

        tools:
            OpenAI-format tool definitions.

    Returns:
        A dictionary containing:

        ``tool_calls``
            List of parsed tool calls.

        ``content``
            Normal assistant content surrounding the tool calls.
    """

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.tool_parsers import ToolParserManager

    # Resolve the parser through vLLM's central registry.
    try:
        parser_cls = ToolParserManager.get_tool_parser(parser)
    except KeyError as exc:
        available = ToolParserManager.list_registered()

        raise RuntimeError(
            f"vLLM tool parser {parser!r} is not registered. "
            f"Available parsers: {available}"
        ) from exc

    # Instantiate the actual vLLM parser.
    parser_instance = parser_cls(
        tokenizer,
        tools,
    )

    # The parser API expects a ChatCompletionRequest.
    #
    # For offline parsing we only need the fields relevant to tool parsing.
    request = ChatCompletionRequest(
        model="offline",
        messages=[],
        tools=tools or None,
    )

    result = parser_instance.extract_tool_calls(
        text,
        request,
    )

    parsed_calls = []

    for call in result.tool_calls:
        arguments = call.function.arguments

        # vLLM's parsers commonly return function arguments as a JSON string.
        # LangChain AIMessage.tool_calls expects ``args`` as a dictionary when
        # possible.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Keep the original string if the parser returned something
                # that is not valid JSON.
                pass

        parsed_calls.append(
            {
                "name": call.function.name,
                "arguments": arguments,
            }
        )

    return {
        "tool_calls": parsed_calls,
        "content": result.content,
    }