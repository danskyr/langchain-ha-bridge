How Tools Are Provided to OpenAI for Use

1. Tool Architecture Overview

Home Assistant implements a sophisticated tool system that allows LLMs like OpenAI to interact with
devices and services. The architecture has several key components:

Core Components:

- Tool base class - Abstract interface all tools must implement
- API class - Groups related tools together
- APIInstance - Runtime instance with tools and context
- Tool formatters - Convert HA tools to LLM-specific formats

2. Tool Definition Structure

Tools in Home Assistant follow this pattern:

class MyTool(llm.Tool):
name = "tool_name"
description = "What this tool does"
parameters = vol.Schema({
vol.Required("param1"): cv.string,
vol.Optional("param2"): cv.positive_int
})

      async def async_call(self, hass: HomeAssistant, tool_input: ToolInput, llm_context: LLMContext)

-> dict:
# Tool implementation that interacts with HA
return {"result": "success"}

3. Tool Registration and Provisioning Process

Step 1: Tool Registration

# Tools are grouped into APIs

api = AssistAPI(hass)  # Built-in API with HA functionality
llm.async_register_api(hass, api)

Step 2: Tool Discovery

# When LLM needs tools, it requests an API instance

api_instance = await llm.async_get_api(hass, "assist", llm_context)
available_tools = api_instance.tools # List of Tool objects

Step 3: OpenAI Format Conversion
def _format_tool(tool: llm.Tool, custom_serializer) -> FunctionToolParam:
"""Convert HA tool to OpenAI function format"""
return FunctionToolParam(
type="function",
name=tool.name,
parameters=convert(tool.parameters, custom_serializer=custom_serializer),
description=tool.description,
strict=False,
)

# Convert all tools for OpenAI

openai_tools = [_format_tool(tool, chat_log.llm_api.custom_serializer)
for tool in chat_log.llm_api.tools]

Step 4: LLM Integration

# Tools are passed to OpenAI API

model_args = {
"model": "gpt-4",
"input": messages,
"tools": openai_tools, # Formatted tools
# other parameters...
}

response = await client.responses.create(**model_args)

4. Tool Execution Flow

1. LLM decides to use a tool based on user request
2. OpenAI returns tool call with function name and arguments
3. Home Assistant receives tool call and validates it
4. Tool execution occurs via APIInstance.async_call_tool()
5. Result is returned to OpenAI and incorporated in response

How This Transfers to Other Codebases

Generic Tool System Pattern

The Home Assistant approach can be adapted to any codebase with these components:

# 1. Abstract tool interface

class Tool(ABC):
name: str
description: str
parameters: dict # JSON schema for parameters

      @abstractmethod
      async def execute(self, context: dict, args: dict) -> dict:
          pass

# 2. Tool registry/manager

class ToolManager:
def __init__(self):
self._tools = {}

      def register_tool(self, tool: Tool):
          self._tools[tool.name] = tool

      async def execute_tool(self, name: str, args: dict, context: dict):
          if name not in self._tools:
              raise ToolNotFoundError(f"Tool {name} not found")
          return await self._tools[name].execute(context, args)

# 3. LLM format converter

def format_tools_for_openai(tools: list[Tool]) -> list[dict]:
return [{
"type": "function",
"function": {
"name": tool.name,
"description": tool.description,
"parameters": tool.parameters
}
} for tool in tools]

# 4. Integration with OpenAI

async def chat_with_tools(messages: list[dict], tools: list[Tool]):
formatted_tools = format_tools_for_openai(tools)

      response = await openai_client.chat.completions.create(
          model="gpt-4",
          messages=messages,
          tools=formatted_tools,
          tool_choice="auto"
      )

      # Handle tool calls in response
      if response.choices[0].message.tool_calls:
          for tool_call in response.choices[0].message.tool_calls:
              result = await tool_manager.execute_tool(
                  tool_call.function.name,
                  json.loads(tool_call.function.arguments),
                  context
              )
              # Add result back to conversation

Key Transferable Concepts

1. Separation of Concerns: Tools, registry, and LLM integration are separate
2. Schema-Based Validation: Use JSON schemas for parameter validation
3. Async Interface: All tool execution should be async
4. Context Passing: Pass relevant context (user, session, etc.) to tools
5. Error Handling: Standardized error handling and reporting
6. Format Abstraction: Abstract tool definitions from LLM-specific formats

Connection to Home Assistant Devices

Built-in Device Control Tools

Home Assistant provides several built-in tools that directly interact with devices:

1. Intent-Based Tools
   class IntentTool(Tool):
   """Wraps HA intents as LLM tools"""
   # Examples: turn_on, turn_off, set_temperature, etc.

   async def async_call(self, hass, tool_input, llm_context):
   # Converts tool args to intent slots
   # Calls intent.async_handle() to control devices

2. Script Execution Tools
   class ScriptTool(Tool):
   """Allows LLMs to run HA scripts"""
   # Scripts can control multiple devices in sequence

   async def async_call(self, hass, tool_input, llm_context):
   # Executes script.turn_on service
   # Scripts control devices via service calls

3. Live Context Tool
   class GetLiveContextTool(Tool):
   """Provides real-time device states"""

   async def async_call(self, hass, tool_input, llm_context):
   # Returns current state of all exposed entities
   # Includes sensors, switches, lights, etc.

Device Interaction Examples

Turning on a light:

1. User: "Turn on the living room light"
2. OpenAI calls turn_on tool with {"entity_id": "light.living_room"}
3. Tool executes hass.services.async_call("light", "turn_on", {"entity_id": "light.living_room"})
4. Device responds and state is updated

Getting device status:

1. User: "What's the temperature in the bedroom?"
2. OpenAI calls GetLiveContext tool
3. Tool returns current states including sensor.bedroom_temperature: 72°F
4. OpenAI incorporates this data in response

Complex automation:

1. User: "Set up evening mode"
2. OpenAI calls script tool with evening_mode script
3. Script dims lights, adjusts thermostat, locks doors
4. Multiple devices controlled through single tool call

Device Entity Integration

Tools interact with Home Assistant's entity system:

- Entities represent individual devices/sensors
- Services provide actions (turn_on, set_temperature, etc.)
- States provide current device status
- Tool Bridge converts between LLM requests and HA service calls

The tool system acts as a bridge between natural language requests from LLMs and the technical
device control APIs that Home Assistant provides.