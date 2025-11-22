import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
# from langchain.chains.router import RouterChain
from langchain.chains import LLMChain
from pydantic import BaseModel
import json
import asyncio

from langchain_agent.src.router_agent_v2 import LangChainRouterAgentV2


def setup_file_logging():
    """
    Set up comprehensive file logging for easy debugging.

    Creates:
    - logs/langchain_agent.log - All logs (rotating, max 10MB, keep 5 backups)
    - logs/langchain_agent_errors.log - Error logs only
    - logs/conversations/YYYY-MM-DD.log - Daily conversation logs
    """
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create conversations subdirectory
    conversations_dir = log_dir / "conversations"
    conversations_dir.mkdir(exist_ok=True)

    # Detailed formatter with timestamps
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Simple formatter for conversation logs
    conversation_formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # 1. Main rotating log file - All logs
    main_file_handler = RotatingFileHandler(
        log_dir / "langchain_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)

    # 2. Error-only log file
    error_file_handler = RotatingFileHandler(
        log_dir / "langchain_agent_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)

    # 3. Daily conversation log file
    today = datetime.now().strftime("%Y-%m-%d")
    conversation_file_handler = logging.FileHandler(
        conversations_dir / f"{today}.log",
        encoding='utf-8'
    )
    conversation_file_handler.setLevel(logging.INFO)
    conversation_file_handler.setFormatter(conversation_formatter)

    # Create a separate logger for conversations
    conversation_logger = logging.getLogger('conversations')
    conversation_logger.setLevel(logging.INFO)
    conversation_logger.addHandler(conversation_file_handler)
    conversation_logger.propagate = False  # Don't propagate to root

    # 4. Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)

    # Log startup
    root_logger.info("=" * 80)
    root_logger.info("File logging initialized")
    root_logger.info(f"Main log: {log_dir / 'langchain_agent.log'}")
    root_logger.info(f"Error log: {log_dir / 'langchain_agent_errors.log'}")
    root_logger.info(f"Conversation log: {conversations_dir / f'{today}.log'}")
    root_logger.info("=" * 80)

    return logging.getLogger('conversations')


# Initialize file logging
conversation_logger = setup_file_logging()

# Set up module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()


class OpenAITextCompletionRequest(BaseModel):
    prompt: str
    tools: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[str] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    # model: str
    # max_tokens: str
    # temperature: str
    # top_p: str


class Choice(BaseModel):
    text: str
    finish_reason: str
    # index: int
    # logprobs: dict


class OpenAICompatibleResponse(BaseModel):
    choices: List[Choice]
    object: str
    # id
    # created
    # model
    # system_fingerprint
    # usage


# Example Request
# {
#   "model": "acon96/Home-3B-v3-GGUF",
#   "max_tokens": 128,
#   "temperature": 0.1,
#   "top_p": 1.0,
#   "prompt": "<|system|>\nYou are 'Al', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed with the information provided only.\nThe current time and date is 08:31 PM on Tuesday May 13, 2025\nServices: HassTurnOn(Any('name', 'area', 'floor', msg=None),domain,device_class) - Turns on/opens a device or entity, HassTurnOff(Any('name', 'area', 'floor', msg=None),domain,device_class) - Turns off/closes a device or entity, HassCancelAllTimers(area) - Cancels all timers, HassBroadcast(message) - Broadcast a message through the home, HassLightSet(Any('name', 'area', 'floor', msg=None),domain,color,temperature,temperature,brightness,brightness) - Sets the brightness percentage or color of a light, HassListAddItem(item,name) - Add item to a todo list, HassListCompleteItem(item,name) - Complete item on a todo list, HassMediaUnpause(Any('name', 'area', 'floor', msg=None),domain,device_class) - Resumes a media player, HassMediaPause(Any('name', 'area', 'floor', msg=None),domain,device_class) - Pauses a media player, HassMediaNext(Any('name', 'area', 'floor', msg=None),domain,device_class) - Skips a media player to the next item, HassMediaPrevious(Any('name', 'area', 'floor', msg=None),domain,device_class) - Replays the previous item for a media player, HassSetVolume(Any('name', 'area', 'floor', msg=None),domain,device_class,volume_level,volume_level,volume_level) - Sets the volume percentage of a media player, GetLiveContext() - Use this tool when the user asks a question about the CURRENT state, value, or mode of a specific device, sensor, entity, or area in the smart home, and the answer can be improved with real-time data not available in the static device overview list. \nDevices:\nperson.test_voice_assistant 'Daniel' = home\nconversation.home_assistant 'Home Assistant' = 2025-05-09T09:48:15.746238+00:00\nscene.test_office_scene 'Test office scene' = unknown\nscene.bright_evening 'Bright Evening' = unknown\nzone.home 'Home' = 1\nlight.fairy_lights_outlet 'Fairy lights Outlet' = on\nsun.sun 'Sun' = below_horizon\nsensor.sun_next_dawn 'Sun Next dawn' = 2025-05-13T20:12:07+00:00\nsensor.sun_next_dusk 'Sun Next dusk' = 2025-05-14T07:29:56+00:00\nsensor.sun_next_midnight 'Sun Next midnight' = 2025-05-13T13:51:12+00:00\nsensor.sun_next_noon 'Sun Next noon' = 2025-05-14T01:51:11+00:00\nsensor.sun_next_rising 'Sun Next rising' = 2025-05-13T20:39:08+00:00\nsensor.sun_next_setting 'Sun Next setting' = 2025-05-14T07:02:56+00:00\ntts.google_translate_en_com 'Google Translate en com' = unknown\nlight.letitias_bedside_lamp 'Letitia’s Bedside Lamp' = on;royalblue (63, 82, 255);100%\nlight.reading_light 'Reading Light' = on;darkorange (255, 147, 41);34%\nlight.mood_lamp 'Mood Lamp ' = on;darkorange (255, 149, 36);100%\nlight.tv_mood_lamp 'TV Mood Lamp' = on;blueviolet (126, 17, 255);100%\nlight.dans_bedside_lamp 'Dan’s Bedside Lamp' = on;royalblue (63, 82, 255);100%\nlight.office_lamp 'Office lamp' = on;darkorange (255, 146, 39);100%\nlight.living_room 'Living Room' = on;78%\nlight.bedroom 'Bedroom' = on;royalblue (63, 82, 255);100%\nlight.office 'Office' = on;darkorange (255, 146, 39);100%\nscene.living_room_read 'Living Room Read' = unknown;100%\nscene.living_room_concentrate 'Living Room Concentrate' = 2025-05-02T13:01:50.714705+00:00;100%\nscene.living_room_nightlight 'Living Room Nightlight' = 2025-05-02T13:01:52.870866+00:00;0%\nscene.office_nightlight 'Office Nightlight' = unknown;0%\nscene.bedroom_tropical_twilight 'Bedroom Tropical twilight' = unknown;48%\nscene.bedroom_laeti_reads 'Bedroom Laeti Reads' = unknown;100%\nscene.bedroom_dan_reads 'Bedroom Dan Reads' = unknown;100%\nscene.living_room_spring_blossom 'Living Room Spring blossom' = unknown;81%\nscene.living_room_bright 'Living Room Bright' = unknown;100%\nscene.bedroom_bright 'Bedroom Bright' = unknown;100%\nscene.office_relax 'Office Relax' = unknown;56%\nscene.living_room_relax 'Living Room Relax' = unknown;56%\nscene.bedroom_savanna_sunset 'Bedroom Savanna sunset' = 2025-05-03T06:03:12.113614+00:00;78%\nscene.living_room_energize 'Living Room Energize' = unknown;100%\nscene.living_room_friends_coming 'Living Room Friends coming' = 2025-05-02T13:01:52.457634+00:00;100%\nscene.office_concentrate 'Office Concentrate' = unknown;100%\nscene.living_room_savanna_sunset 'Living Room Savanna sunset' = unknown;80%\nscene.bedroom_read 'Bedroom Read' = unknown;100%\nscene.living_room_tv_evening 'Living Room TV evening' = 2025-05-02T13:01:54.720663+00:00;100%\nscene.bedroom_arctic_aurora 'Bedroom Arctic aurora' = 2025-05-08T03:34:15.394766+00:00;53%\nscene.office_read 'Office Read' = unknown;100%\nscene.living_room_dinner 'Living Room Dinner' = unknown;9%\nscene.bedroom_relax 'Bedroom Relax' = unknown;56%\nscene.living_room_tropical_twilight 'Living Room Tropical twilight' = unknown;43%\nscene.bedroom_spring_blossom 'Bedroom Spring blossom' = unknown;81%\nscene.living_room_arctic_aurora 'Living Room Arctic aurora' = unknown;30%\nscene.office_energize 'Office Energize' = unknown;100%\nscene.bedroom_nightlight 'Bedroom Nightlight' = unknown;0%\nscene.living_room_movie 'Living Room Movie' = unknown;0%\nscene.living_room_orange_isnt_the_new_black 'Living Room Orange isnt the new black' = unknown;62%\nscene.bedroom_concentrate 'Bedroom Concentrate' = unknown;100%\nscene.bedroom_dimmed 'Bedroom Dimmed' = unknown;30%\nscene.bedroom_energize 'Bedroom Energize' = unknown;100%\nscene.living_room_dimmed 'Living Room Dimmed' = unknown;30%\ntodo.shopping_list 'Shopping List' = 0\nsensor.backup_backup_manager_state 'Backup Backup Manager state' = idle\nsensor.backup_next_scheduled_automatic_backup 'Backup Next scheduled automatic backup' = unknown\nsensor.backup_last_successful_automatic_backup 'Backup Last successful automatic backup' = unknown\nmedia_player.home_assistant_voice_09dbdf_media_player 'Home Assistant Voice 09dbdf Media Player' = unavailable\nweather.forecast_home 'Forecast Home' = partlycloudy;17.6 °C;83%;13.0 km/h\nstt.faster_whisper 'faster-whisper' = unknown\ntts.piper 'piper' = 2025-05-09T09:47:23.806707+00:00\nmedia_player.spotify_daniel_reissenberger 'Spotify Daniel Reissenberger' = idle<|endoftext|>\n<|user|>\nwhat the weather outsid eyo<|endoftext|>\n<|assistant|>\n"
# }

# Example Response
# {
#   "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
#   "object": "text_completion",
#   "created": 1589478378,
#   "model": "gpt-3.5-turbo-instruct",
#   "system_fingerprint": "fp_44709d6fcb",
#   "choices": [
#     {
#       "text": "\n\nThis is indeed a test",
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "length"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 5,
#     "completion_tokens": 7,
#     "total_tokens": 12
#   }
# }

router_agent = LangChainRouterAgentV2()

class ToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any]

class OurResponse(BaseModel):
    response: Optional[str] = None
    type: str = "response"  # "response" or "tool_call"
    tool_calls: Optional[List[ToolCall]] = None
    conversation_id: Optional[str] = None
    continue_conversation: Optional[bool] = None

@app.post("/v1/completions", response_model=OurResponse)
async def process(req: OpenAITextCompletionRequest):
    # Log to both main and conversation logs
    conv_id = req.conversation_id or "new"
    logger.info(f"🌐 Incoming request: {len(req.prompt)} chars, has_tools={bool(req.tools)}, has_results={bool(req.tool_results)}")
    logger.debug(f"Full request: {req}")

    # Detailed conversation logging
    conversation_logger.info("=" * 80)
    conversation_logger.info(f"📥 NEW REQUEST | Conversation: {conv_id[:8]}...")
    conversation_logger.info(f"Query: {req.prompt[:200]}{'...' if len(req.prompt) > 200 else ''}")
    conversation_logger.info(f"Has tools: {bool(req.tools)} ({len(req.tools) if req.tools else 0} tools)")
    conversation_logger.info(f"Has tool results: {bool(req.tool_results)} ({len(req.tool_results) if req.tool_results else 0} results)")

    if req.tool_results:
        conversation_logger.info("Tool Results:")
        for i, result in enumerate(req.tool_results, 1):
            tool_name = result.get('tool_name', 'unknown')
            result_preview = str(result.get('result', ''))[:100]
            conversation_logger.info(f"  {i}. {tool_name}: {result_preview}{'...' if len(str(result.get('result', ''))) > 100 else ''}")

    # Use unified process method that handles both initial and continuation calls
    result = await router_agent.process(
        query=req.prompt,
        tools=req.tools,
        conversation_id=req.conversation_id,
        tool_results=req.tool_results
    )

    # Return appropriate response based on result type
    if result.get("type") == "tool_call":
        logger.info(f"  🔧 [process] Returning {len(result.get('tool_calls', []))} tool calls")

        # Log full tool call details to main log for debugging
        for i, tc in enumerate(result.get('tool_calls', []), 1):
            logger.info(f"  🔧   Tool call #{i}: {tc['name']}")
            logger.info(f"  🔧     ID: {tc.get('id', 'no-id')}")
            logger.info(f"  🔧     Args: {tc.get('args', {})}")

        logger.info(f"  ← [process] Returning HTTP 200 with tool calls")

        # Log tool calls to conversation log
        conversation_logger.info(f"🔧 TOOL CALLS REQUESTED: {len(result.get('tool_calls', []))}")
        for i, tc in enumerate(result.get('tool_calls', []), 1):
            args_preview = str(tc.get('args', {}))[:100]
            conversation_logger.info(f"  {i}. {tc['name']}({args_preview}{'...' if len(str(tc.get('args', {}))) > 100 else ''})")
        conversation_logger.info(f"Conversation ID: {result.get('conversation_id')}")
        conversation_logger.info("=" * 80 + "\n")

        return OurResponse(
            type="tool_call",
            tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
            conversation_id=result.get("conversation_id")
        )
    else:
        logger.info(f"  ✓ [process] Returning final response")
        logger.info(f"  ← [process] Returning HTTP 200 with response")

        # Log final response to conversation log
        response_text = result.get("response", "No response generated")
        continue_conversation = result.get("continue_conversation")
        conversation_logger.info(f"✅ FINAL RESPONSE:")
        conversation_logger.info(f"   {response_text[:300]}{'...' if len(response_text) > 300 else ''}")
        if continue_conversation is not None:
            conversation_logger.info(f"   continue_conversation: {continue_conversation}")
        conversation_logger.info("=" * 80 + "\n")

        return OurResponse(
            response=response_text,
            type="response",
            continue_conversation=continue_conversation
        )


# @app.post("/v1/completions", response_model=OpenAICompatibleResponse)
# def process(req: OpenAITextCompletionRequest):
#     print("REQUEST", req)
#     response_text = router_agent.route(req.prompt)
#     # return {"response": OpenAICompatibleResponse(
#     #     choices=[
#     #         Choice(text="Yoooooooooooo Mufka")
#     #     ]
#     # )}
#     return OpenAICompatibleResponse(
#         object="text_completion",
#         choices=[
#             Choice(text=response_text, finish_reason="length")
#         ]
#     )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "langchain-conversation-agent"}

@app.post("/test")
async def test_connection():
    """Test endpoint for Home Assistant integration."""
    return {"status": "ok", "message": "Connection successful"}


@app.post("/v1/completions/stream")
async def process_stream(req: OpenAITextCompletionRequest):
    """
    Streaming endpoint that uses LangGraph's native streaming to yield events.

    Uses Server-Sent Events (SSE) format.
    """
    import uuid

    async def generate():
        conv_id = req.conversation_id or str(uuid.uuid4())
        logger.info(f"🌐 Streaming request: {len(req.prompt)} chars, conv_id: {conv_id[:8]}...")

        try:
            # Use LangGraph's streaming capabilities
            thread_id = conv_id
            config = {"configurable": {"thread_id": thread_id}}

            if req.tool_results:
                # Resuming with tool results
                from langchain_core.messages import ToolMessage

                tool_messages = []
                for result in req.tool_results:
                    tool_messages.append(
                        ToolMessage(
                            content=str(result.get("result", "")),
                            tool_call_id=result.get("tool_call_id", "")
                        )
                    )

                state_update = {
                    "messages": tool_messages,
                    "validation_attempts": 1
                }

                # Stream events as they happen
                async for event in router_agent.graph.astream_events(state_update, config, version="v2"):
                    # Check for preliminary messages in state updates
                    if event["event"] == "on_chain_stream" and event.get("data"):
                        chunk = event["data"].get("chunk", {})

                        # Stream preliminary messages immediately
                        if preliminary_msgs := chunk.get("preliminary_messages"):
                            for msg in preliminary_msgs:
                                logger.info(f"Streaming preliminary: {msg}")
                                yield f"data: {json.dumps({'type': 'preliminary', 'content': msg})}\n\n"

                        # Stream any explicit streaming events
                        if streaming_events := chunk.get("streaming_events"):
                            for evt in streaming_events:
                                logger.info(f"Streaming event: {evt}")
                                yield f"data: {json.dumps(evt)}\n\n"

            else:
                # New conversation
                from langchain_core.messages import HumanMessage

                initial_state = {
                    "messages": [HumanMessage(content=req.prompt)],
                    "query": req.prompt,
                    "route_types": [],
                    "handler_responses": [],
                    "final_response": None,
                    "tools": req.tools,
                    "validation_attempts": 1,
                    "preliminary_messages": [],
                    "streaming_events": []
                }

                # Stream events as they happen
                async for event in router_agent.graph.astream_events(initial_state, config, version="v2"):
                    # Check for preliminary messages in state updates
                    if event["event"] == "on_chain_stream" and event.get("data"):
                        chunk = event["data"].get("chunk", {})

                        # Stream preliminary messages immediately
                        if preliminary_msgs := chunk.get("preliminary_messages"):
                            for msg in preliminary_msgs:
                                logger.info(f"Streaming preliminary: {msg}")
                                yield f"data: {json.dumps({'type': 'preliminary', 'content': msg})}\n\n"

                        # Stream any explicit streaming events
                        if streaming_events := chunk.get("streaming_events"):
                            for evt in streaming_events:
                                logger.info(f"Streaming event: {evt}")
                                yield f"data: {json.dumps(evt)}\n\n"

            # Get the final result
            result = await router_agent.process(
                query=req.prompt,
                tools=req.tools,
                conversation_id=conv_id,
                tool_results=req.tool_results
            )

            # Yield the final result
            if result.get("type") == "tool_call":
                yield f"data: {json.dumps({'type': 'tool_call', 'tool_calls': result['tool_calls'], 'conversation_id': result.get('conversation_id')})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'response', 'response': result.get('response'), 'continue_conversation': result.get('continue_conversation')})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )