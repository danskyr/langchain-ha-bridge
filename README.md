# LangChain Router Agent for Home Assistant

This project provides a smart conversation agent for Home Assistant that can:

1. Route user queries to the appropriate handler based on intent
2. Control smart home devices through natural language commands
3. Answer general questions using a more capable language model

## How It Works

The system uses a two-part architecture:

1. **LangChain Server**: A standalone FastAPI service that hosts the LangChain router agent
   - Runs independently from Home Assistant
   - Processes text queries and returns responses
   - Can be deployed anywhere (same machine, different machine, cloud)

2. **Home Assistant Proxy Component**: A lightweight custom component for Home Assistant
   - Forwards conversation requests from Home Assistant to the LangChain server
   - Returns responses back to Home Assistant

The LangChain agent itself uses a two-stage approach:

1. **Router**: Classifies the user's input as either a device control command or a general query
2. **Handlers**:
   - **Device Control**: Interprets commands to control smart home devices and generates Home Assistant service calls
   - **General Queries**: Uses a more capable language model to answer questions and provide information

## Installation

### 1. Set up the LangChain Server

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/langchain-ha-bridge.git
   cd langchain-ha-bridge
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

4. Start the server:
   ```bash
   poetry run python -m src.langchain_ha_bridge
   ```

   The server will start on http://0.0.0.0:8000

   Alternatively, you can use the example script:
   ```bash
   chmod +x examples/run_server.sh
   ./examples/run_server.sh
   ```

### 2. Install the Home Assistant Proxy Component

1. Copy the `custom_components/langchain_remote` directory to your Home Assistant configuration directory
2. Edit the `conversation.py` file to set the correct URL for your LangChain server:
   ```python
   self._url = "http://YOUR_SERVER:8000/process"  # Replace with your server's address
   ```
3. Restart Home Assistant
4. Go to Settings → Voice Assistants (or Conversation in older versions)
5. Click Add Assistant, pick "LangChain Remote Agent"

## Configuration Options

### LangChain Server Environment Variables

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |
| OPENAI_API_KEY | Yes | - | Your OpenAI API key |
| ROUTER_MODEL | No | gpt-3.5-turbo | The model to use for routing and device control |
| QUERY_MODEL | No | gpt-4 | The model to use for answering general queries |

## Usage

Once installed and configured, you can interact with the agent through any Home Assistant conversation interface:

- The conversation panel in the Home Assistant dashboard
- Voice assistants integrated with Home Assistant
- The Home Assistant mobile app

### Example Commands

#### Device Control
- "Turn on the living room lights"
- "Set the thermostat to 72 degrees"
- "Turn off all the lights in the house"

#### General Queries
- "What is home automation?"
- "How does a smart thermostat work?"
- "Tell me a joke about smart homes"

## Testing

### Testing the LangChain Server

You can test the LangChain server directly using curl:

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"text":"Turn on the living room lights"}'
```

Or using the provided Python example script:

```bash
python examples/test_server.py "Turn on the living room lights"
```

You can also use Python directly:

```python
import requests
response = requests.post(
    "http://localhost:8000/process",
    json={"text": "Turn on the living room lights"}
)
print(response.json())
```

### Testing the Home Assistant Integration

In Home Assistant:

1. Go to Developer Tools → Services
2. Choose `conversation.process`
3. Enter the following payload:
   ```yaml
   text: "Turn on the kitchen light"
   agent_id: langchain_remote
   ```
4. Click "Call Service" and check the response

## Troubleshooting

- Check the Home Assistant logs for errors
- Ensure your OpenAI API key is valid and has access to the models you're trying to use
- If you're getting timeout errors, try using a simpler model like "gpt-3.5-turbo" for both router_model and query_model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
