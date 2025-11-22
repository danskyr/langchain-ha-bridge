"""Conversation platform for LangChain Remote."""
import json
import logging
from typing import AsyncGenerator

import aiohttp
from homeassistant.components import conversation as conversation
from homeassistant.components.conversation import AbstractConversationAgent, ConversationResult, ConversationEntity, \
    async_get_chat_log, AssistantContent
from homeassistant.components.conversation.util import async_get_result_from_chat_log
from homeassistant.components.conversation.chat_log import AssistantContentDeltaDict
from homeassistant.components.ollama.entity import _format_tool
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponseErrorCode

from homeassistant.helpers import device_registry as dr, llm

from .const import DOMAIN
from .utils import get_host_from_url

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
        entry_id = list(self.hass.data[DOMAIN].keys())[0]
        configuration = self.hass.data[DOMAIN][entry_id]
        url = configuration.get("url")
        timeout = configuration.get("timeout", 10)
        verify_ssl = configuration.get("verify_ssl", True)
        use_streaming = configuration.get("streaming", False)

        # Prepare LLM data if available
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                user_llm_hass_api="assist",  # Use the assist API to get tools
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

        # Use streaming if enabled
        if use_streaming:
            return await self._async_handle_message_streaming(
                user_input, chat_log, url, timeout, verify_ssl, tools
            )

        response_text = None
        api_continue_conversation = None  # Let API override continue_conversation

        try:
            session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)

            # Initial request to LangChain
            response = await session.post(
                f"{url}/v1/completions",
                json={"prompt": user_input.text, "tools": tools},
                timeout=aiohttp.ClientTimeout(total=timeout)
            )

            if response.status == 200:
                response_body = await response.json()

                # Check if LangChain wants to execute tools
                if response_body.get("type") == "tool_call" and response_body.get("tool_calls"):
                    _LOGGER.info(f"LangChain requested {len(response_body['tool_calls'])} tool calls")

                    # Execute tools and get results
                    response_text, api_continue_conversation = await self._execute_tools_and_continue(
                        session, url, timeout, response_body, chat_log, user_input.text, tools
                    )

                elif response_body.get("type") == "response":
                    # Direct response without tools
                    response_text = response_body.get("response", "No response provided")
                    api_continue_conversation = response_body.get("continue_conversation")

                else:
                    _LOGGER.error("Unknown response type from LangChain service: %s", response_body)
                    response_text = "Invalid response format from LangChain service."
            else:
                response_text = f"Invalid response from LangChain service. Status code: {response.status}"

        except aiohttp.ClientConnectorError:
            response_text = "Connection to LangChain service refused."
        except aiohttp.ClientSSLError:
            response_text = "SSL Error when connecting to LangChain service."
        except aiohttp.ClientTimeout as err:
            response_text = f"LangChain service timed out. Error: {err}"
        except Exception as err:
            _LOGGER.error("Unexpected error in message handling: %s", err)
            response_text = "Unknown error when connecting to LangChain service."

        # Add response to chat_log and return result
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(agent_id=self.entity_id, content=response_text or "")
        )
        result = async_get_result_from_chat_log(user_input, chat_log)

        # Override continue_conversation if API specified it
        if api_continue_conversation is not None:
            result.continue_conversation = api_continue_conversation

        return result

    async def _async_handle_message_streaming(
            self,
            user_input: conversation.ConversationInput,
            chat_log: conversation.ChatLog,
            url: str,
            timeout: int,
            verify_ssl: bool,
            tools: list
    ) -> conversation.ConversationResult:
        """Handle message with streaming support for preliminary responses."""
        api_continue_conversation = None
        tool_calls_pending = None

        async def stream_from_langchain() -> AsyncGenerator[AssistantContentDeltaDict, None]:
            """Stream deltas from LangChain SSE endpoint."""
            nonlocal api_continue_conversation, tool_calls_pending

            session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)

            try:
                async with session.post(
                    f"{url}/v1/completions/stream",
                    json={"prompt": user_input.text, "tools": tools},
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status != 200:
                        _LOGGER.error(f"Streaming endpoint returned {response.status}")
                        yield {"role": "assistant", "content": f"Error: Status {response.status}"}
                        return

                    # Start new assistant message
                    yield {"role": "assistant"}

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or not line.startswith('data: '):
                            continue

                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break

                        try:
                            event = json.loads(data)
                            event_type = event.get('type')

                            if event_type == 'preliminary':
                                # Stream preliminary text
                                content = event.get('content', '')
                                _LOGGER.info(f"Streaming preliminary: {content}")
                                yield {"content": content + " "}

                            elif event_type == 'response':
                                # Stream final response
                                content = event.get('response', '')
                                api_continue_conversation = event.get('continue_conversation')
                                _LOGGER.info(f"Streaming response: {content[:50]}...")
                                yield {"content": content}

                            elif event_type == 'tool_call':
                                # Tool calls need special handling - can't stream these
                                tool_calls_pending = event
                                _LOGGER.info(f"Received tool calls in stream")
                                yield {"content": "Processing..."}

                        except json.JSONDecodeError as e:
                            _LOGGER.error(f"Failed to parse SSE event: {e}")
                            continue

            except aiohttp.ClientError as e:
                _LOGGER.error(f"Streaming error: {e}")
                yield {"role": "assistant", "content": f"Connection error: {e}"}

        try:
            # Stream content through chat_log
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                stream_from_langchain()
            ):
                _LOGGER.debug(f"Delta content processed: {type(content)}")

            # Handle pending tool calls if any
            if tool_calls_pending:
                session = async_get_clientsession(self.hass, verify_ssl=verify_ssl)
                response_text, api_continue_conversation = await self._execute_tools_and_continue(
                    session, url, timeout, tool_calls_pending, chat_log, user_input.text, tools
                )
                # Add tool response to chat_log
                chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(agent_id=self.entity_id, content=response_text or "")
                )

        except Exception as err:
            _LOGGER.error(f"Error in streaming handler: {err}", exc_info=True)
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=self.entity_id, content=f"Error: {err}")
            )

        result = async_get_result_from_chat_log(user_input, chat_log)

        # Override continue_conversation if API specified it
        if api_continue_conversation is not None:
            result.continue_conversation = api_continue_conversation

        return result

    async def _execute_tools_and_continue(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout: int,
        tool_call_response: dict,
        chat_log: conversation.ChatLog,
        original_query: str,
        tools: list
    ) -> tuple[str, bool | None]:
        """Execute tools via Home Assistant and continue conversation with LangChain.

        Returns:
            tuple: (response_text, continue_conversation)
        """

        tool_calls = tool_call_response.get("tool_calls", [])
        conversation_id = tool_call_response.get("conversation_id")

        _LOGGER.info(f"=== TOOL EXECUTION DEBUG ===")
        _LOGGER.info(f"Tool calls to execute: {tool_calls}")
        _LOGGER.info(f"Conversation ID: {conversation_id}")

        if not chat_log.llm_api:
            _LOGGER.error("No LLM API available for tool execution")
            return "Error: No LLM API available for tool execution", None

        try:
            # Create assistant content with tool calls
            _LOGGER.info("Creating AssistantContent with tool calls")
            assistant_content = AssistantContent(
                agent_id=self.entity_id,
                content="",  # No text content, just tool calls
                tool_calls=[
                    llm.ToolInput(
                        id=tc.get("id", ""),
                        tool_name=tc["name"],
                        tool_args=tc["args"]
                    ) for tc in tool_calls
                ]
            )
            _LOGGER.info(f"AssistantContent created: {assistant_content}")

            # Execute tools and collect results
            tool_results = []
            _LOGGER.info("Starting tool execution loop...")
            try:
                async for tool_result in chat_log.async_add_assistant_content(assistant_content):
                    _LOGGER.info(f"Tool {tool_result.tool_name} executed: {tool_result.tool_result}")
                    tool_results.append({
                        "tool_call_id": tool_result.tool_call_id,
                        "tool_name": tool_result.tool_name,
                        "result": tool_result.tool_result
                    })
                _LOGGER.info(f"Tool execution loop completed. Collected {len(tool_results)} results")
            except Exception as tool_exec_error:
                _LOGGER.error(f"Error during tool execution loop: {tool_exec_error}", exc_info=True)
                raise

            _LOGGER.info(f"Tool results to send back: {tool_results}")

            # Send tool results back to LangChain for final response
            _LOGGER.info("Sending tool results back to LangChain for final response")
            request_body = {
                "prompt": original_query,
                "tools": tools,
                "tool_results": tool_results,
                "conversation_id": conversation_id
            }
            _LOGGER.info(f"Request body: {request_body}")

            response = await session.post(
                f"{url}/v1/completions",
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=timeout)
            )

            _LOGGER.info(f"Response status: {response.status}")

            if response.status == 200:
                response_body = await response.json()
                _LOGGER.info(f"Response body: {response_body}")

                final_response = response_body.get("response")
                if not final_response:
                    _LOGGER.warning(f"No 'response' field in response body: {response_body}")
                    return "Tool execution completed but no response received", None

                _LOGGER.info(f"Final response: {final_response}")
                continue_conversation = response_body.get("continue_conversation")
                return final_response, continue_conversation
            else:
                error_msg = f"Error getting final response: {response.status}"
                _LOGGER.error(error_msg)
                return error_msg, None

        except Exception as e:
            _LOGGER.error(f"Error executing tools: {e}", exc_info=True)
            return f"Error executing tools: {str(e)}", None

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()
