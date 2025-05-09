# LangChain Router Agent for Home Assistant

This custom component integrates LangChain with Home Assistant to provide a smart conversation agent that can:

1. Route user queries to the appropriate handler based on intent
2. Control smart home devices through natural language commands
3. Answer general questions using a more capable language model

## How It Works

The agent uses a two-stage approach:

1. **Router**: Classifies the user's input as either a device control command or a general query
2. **Handlers**:
   - **Device Control**: Interprets commands to control smart home devices and generates Home Assistant service calls
   - **General Queries**: Uses a more capable language model to answer questions and provide information

## Installation

### Manual Installation

1. Copy the `custom_components/langchain_agent` directory to your Home Assistant configuration directory
2. Restart Home Assistant
3. Add the following to your `configuration.yaml`:

```yaml
langchain_agent:
  openai_api_key: "your-openai-api-key"
  router_model: "gpt-3.5-turbo"  # Optional, defaults to gpt-3.5-turbo
  query_model: "gpt-4"  # Optional, defaults to gpt-4
  
conversation:
  integration: langchain_agent
```

4. Restart Home Assistant again

## Configuration Options

| Option | Type | Required | Default | Description |
| ------ | ---- | -------- | ------- | ----------- |
| openai_api_key | string | Yes | - | Your OpenAI API key |
| router_model | string | No | gpt-3.5-turbo | The model to use for routing and device control |
| query_model | string | No | gpt-4 | The model to use for answering general queries |

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

A test script is included to verify that the agent is working correctly. To use it:

1. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

2. Run the test script:
   ```bash
   python test_langchain_agent.py
   ```

## Troubleshooting

- Check the Home Assistant logs for errors
- Ensure your OpenAI API key is valid and has access to the models you're trying to use
- If you're getting timeout errors, try using a simpler model like "gpt-3.5-turbo" for both router_model and query_model

## License

This project is licensed under the MIT License - see the LICENSE file for details.