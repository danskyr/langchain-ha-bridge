"""Conversation platform for LangChain Remote."""
import asyncio
import logging
import uuid
from typing import List, Dict, Any

from homeassistant.components import conversation as conversation
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationEntity,
    ConversationResult,
    async_get_chat_log,
    AssistantContent,
)
from homeassistant.components.conversation.util import async_get_result_from_chat_log
from homeassistant.components.ollama.entity import _format_tool
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from homeassistant.helpers import device_registry as dr, llm

from .const import DOMAIN
from .client import LangChainClient

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation platform."""
    async_add_entities([RemoteConversationAgent(hass, config_entry)])


class RemoteConversationAgent(AbstractConversationAgent, ConversationEntity):
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super(AbstractConversationAgent, self).__init__()
        super(ConversationEntity, self).__init__()
        self.hass = hass
        self.entry = entry
        # self._name = f"LangChain Conversation Agent ({get_host_from_url(entry.data.get('url'))})"
        self._name = f"LangChain Conversation Agent"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=self._name,
            default_name="LangChain Conversation Agent",
            manufacturer="LangChain",
            model="Custom",
            entry_type=dr.DeviceEntryType.SERVICE,
        )


    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return ["en"]

    # async def async_process(self, user_input: conversation.ConversationInput) -> ConversationResult:
    #     """Process a sentence."""
    #     entry_id = list(self.hass.data[DOMAIN].keys())[0]
    #     url = self.hass.data[DOMAIN][entry_id].get("url", "http://127.0.0.1:8000/process")
    #
    #     # fire off HTTP POST to your LangChain server
    #     resp = requests.post(url, json={"text": text}, timeout=10)
    #     resp.raise_for_status()
    #     response_text = resp.json()["response"]
    #
    #     return ConversationResult(
    #         response=response_text,
    #         conversation_id=conversation_id,
    #     )

    # Note: This `async_process` is copied straight from `ConversationEntity`,
    #       but is marked as abstract on `AbstractConversationAgent` and required to be implemented here...
    async def async_process(self, user_input: conversation.ConversationInput) -> ConversationResult:
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)


    async def _async_handle_message(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle message via WebSocket."""
        config = self.hass.data[DOMAIN][self.entry.entry_id]
        client: LangChainClient = config.get("client")
        timeout = config.get("timeout", 90)

        if not client:
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=self.entity_id, content="LangChain client not configured")
            )
            return async_get_result_from_chat_log(user_input, chat_log)

        # Prepare LLM data if available
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                user_llm_hass_api="assist",
                user_llm_prompt=None,
                user_extra_system_prompt=user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Convert chat_log.content to message format
        messages = self._chat_log_to_messages(chat_log)

        response_text = None
        api_continue_conversation = None

        try:
            async with asyncio.timeout(timeout):
                conv_id = user_input.conversation_id or str(uuid.uuid4())
                response = await client.send_conversation(
                    conversation_id=conv_id,
                    messages=messages,
                    tools=tools
                )

            # Handle response
            if response.get("type") == "tool_call":
                # Execute tools and continue
                response_text, api_continue_conversation = await self._execute_tools_and_continue_ws(
                    client, response, chat_log, messages, tools, timeout
                )
            elif response.get("type") == "response":
                response_text = response.get("response", "")
                api_continue_conversation = response.get("continue_conversation")
            elif response.get("type") == "error":
                response_text = f"Error: {response.get('message', 'Unknown error')}"
            else:
                response_text = "Unexpected response from server"

        except TimeoutError:
            response_text = "Request timed out"
        except ConnectionError as e:
            _LOGGER.warning("Connection error: %s", e)
            response_text = "LangChain server unavailable - please try again"
        except Exception as e:
            _LOGGER.error("Error in message handling: %s", e, exc_info=True)
            response_text = "An error occurred"

        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(agent_id=self.entity_id, content=response_text or "")
        )
        result = async_get_result_from_chat_log(user_input, chat_log)

        if api_continue_conversation is not None:
            result.continue_conversation = api_continue_conversation

        return result

    def _chat_log_to_messages(self, chat_log: conversation.ChatLog) -> List[Dict[str, Any]]:
        """Convert ChatLog content to message format for server."""
        messages = []
        for content in chat_log.content:
            if content.role == "system":
                messages.append({"role": "system", "content": content.content})
            elif content.role == "user":
                messages.append({"role": "user", "content": content.content})
            elif content.role == "assistant":
                msg = {"role": "assistant", "content": content.content}
                if hasattr(content, 'tool_calls') and content.tool_calls:
                    msg["tool_calls"] = [
                        {"id": tc.id, "name": tc.tool_name, "args": tc.tool_args}
                        for tc in content.tool_calls
                    ]
                messages.append(msg)
            elif content.role == "tool_result":
                messages.append({
                    "role": "tool_result",
                    "tool_call_id": content.tool_call_id,
                    "tool_name": content.tool_name,
                    "tool_result": content.tool_result
                })
        return messages

    async def _execute_tools_and_continue_ws(
        self,
        client: LangChainClient,
        tool_response: dict,
        chat_log: conversation.ChatLog,
        messages: List[Dict],
        tools: List[Dict],
        timeout: int
    ) -> tuple[str, bool | None]:
        """Execute tools and send results back to server via WebSocket."""
        tool_calls = tool_response.get("tool_calls", [])
        conv_id = tool_response.get("conversation_id")

        _LOGGER.info("Executing %d tool calls", len(tool_calls))

        if not chat_log.llm_api:
            _LOGGER.error("No LLM API available for tool execution")
            return "Error: No LLM API available for tool execution", None

        # Create AssistantContent with tool calls
        assistant_content = AssistantContent(
            agent_id=self.entity_id,
            content="",
            tool_calls=[
                llm.ToolInput(id=tc["id"], tool_name=tc["name"], tool_args=tc["args"])
                for tc in tool_calls
            ]
        )

        # Execute tools via chat_log
        tool_results_for_msg = []
        async for result in chat_log.async_add_assistant_content(assistant_content):
            _LOGGER.info("Tool %s executed: %s", result.tool_name, result.tool_result)
            tool_results_for_msg.append({
                "role": "tool_result",
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "tool_result": result.tool_result
            })

        # Build updated messages with tool call and results
        updated_messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        ] + tool_results_for_msg

        # Send back to server
        try:
            async with asyncio.timeout(timeout):
                response = await client.send_conversation(
                    conversation_id=conv_id,
                    messages=updated_messages,
                    tools=tools
                )

            if response.get("type") == "response":
                return response.get("response", ""), response.get("continue_conversation")
            elif response.get("type") == "error":
                return f"Error: {response.get('message')}", None
            else:
                return "Unexpected response", None
        except Exception as e:
            _LOGGER.error("Error continuing after tool execution: %s", e)
            return f"Error: {e}", None

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()
