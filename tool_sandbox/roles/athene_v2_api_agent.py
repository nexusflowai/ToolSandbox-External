# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for Athene-V2 models."""

import json
import os
from os import getenv
import uuid
from typing import Any, Literal, Optional, Union, cast

from nexusflowai import NexusflowAI
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent


class AtheneV2ChatAPIAgent(BaseRole):
    """Agent role for Athene-V2-Chat."""

    role_type: RoleType = RoleType.AGENT
    model_name: str

    def __init__(self, model_name: str) -> None:
        super().__init__()

        base_url = os.getenv("ATHENE_BASE_URL")
        api_key = os.getenv("ATHENE_API_KEY")
        assert base_url and api_key, f"Please provide `ATHENE_BASE_URL` and `ATHENE_API_KEY`"
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: list[Message] = self.get_messages(ending_index=ending_index)
        response_messages: list[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[list[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        # Call model
        openai_response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        openai_response_message = openai_response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: Union[list[ChatCompletionToolParam], NotGiven],
    ) -> Any:
        """Run Athene-V2-Chat model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        openai_response = self.client.chat.completions.create(
            messages=openai_messages,
            tools=openai_tools if openai_tools else [],
            parallel_tool_calls=False,
            model=self.model_name,
            max_tokens=4096,
            temperature=0.0,
        )
        return openai_response


class AtheneV2AgentAPIAgent(OpenAIAPIAgent):
    model_name = "Nexusflow/Athene-V2-Agent"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        rapid_api_key = getenv("RAPID_API_KEY")
        assert rapid_api_key, "Please provide a valid `RAPID_API_KEY` that is subscribed to all 5 APIs in `tool_sandbox/tools/rapid_api_search_tools.py`"

        base_url = getenv("ATHENE_BASE_URL")
        api_key = getenv("ATHENE_API_KEY")
        assert base_url and api_key, "Please provide `ATHENE_BASE_URL` and `ATHENE_API_KEY`"
        self.nexusflowai_client: NexusflowAI = NexusflowAI(base_url=base_url, api_key=api_key)

    def model_inference(self, openai_messages, openai_tools):
        request = dict(
            model=self.model_name,
            messages=cast(list[ChatCompletionMessageParam], openai_messages),
            tools=openai_tools,
            temperature=0.0,
            max_tokens=1024,
            store_nexusflowai_extras_in_memory=True,
            tool_choice="required_with_chat",
        )

        with all_logging_disabled():
            response = self.nexusflowai_client.completions.create_with_tools(**request)

            # Currently Nexusflow/Athene-V2-Agent is trained to output nothing when a trajectory finishes
            # We deterministically interpret this and force a chat.
            message = response.choices[0].message
            if not (message.content or message.tool_calls):
                request["tool_choice"] = "none"
                response = self.nexusflowai_client.completions.create_with_tools(**request)

        return response
