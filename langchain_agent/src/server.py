import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
# from langchain.chains.router import RouterChain
from langchain.chains import LLMChain
from pydantic import BaseModel

from langchain_agent.src import LangChainRouterAgent

# Set up logging
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

router_agent = LangChainRouterAgent()

class ToolCall(BaseModel):
    id: str
    name: str
    args: Dict[str, Any]

class OurResponse(BaseModel):
    response: Optional[str] = None
    type: str = "response"  # "response" or "tool_call"
    tool_calls: Optional[List[ToolCall]] = None
    conversation_id: Optional[str] = None

@app.post("/v1/completions", response_model=OurResponse)
async def process(req: OpenAITextCompletionRequest):
    logger.info(f"🌐 Incoming request: {len(req.prompt)} chars, has_tools={bool(req.tools)}, has_results={bool(req.tool_results)}")
    logger.debug(f"Full request: {req}")

    # If this is a tool result, continue conversation
    if req.tool_results:
        logger.info(f"  ↳ [process] Continuing conversation {req.conversation_id[:8] if req.conversation_id else 'N/A'}...")
        response_text = await router_agent.route_with_tool_results(
            req.prompt, req.tools, req.tool_results, req.conversation_id
        )
        logger.info(f"  ✓ [process] Final response generated")
        logger.info(f"  ← [process] Returning HTTP 200 with response")
        return OurResponse(response=response_text, type="response")

    # Initial request with tools
    logger.info(f"  ↳ [process] Starting new conversation...")
    result = await router_agent.route_with_tools(req.prompt, req.tools)

    if result.get("type") == "tool_call":
        logger.info(f"  🔧 [process] Returning {len(result.get('tool_calls', []))} tool calls")
        logger.info(f"  ← [process] Returning HTTP 200 with tool calls")
        return OurResponse(
            type="tool_call",
            tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
            conversation_id=result.get("conversation_id")
        )
    else:
        logger.info(f"  ✓ [process] Returning final response")
        logger.info(f"  ← [process] Returning HTTP 200 with response")
        return OurResponse(
            response=result.get("response", "No response generated"),
            type="response"
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