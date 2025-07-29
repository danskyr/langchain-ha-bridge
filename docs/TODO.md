# Big picture
1. Focus on getting good basis
2. Split out into implementing use-cases (that could even be done independently)

# Current
- Bootstrap clean routing so that we can easily extend it to more complex use-cases in the future
  - ~~Implement agent~~
  - ~~Add response Node (unifies responses in a consistent tone and format/feel)~~
  - ➡️ Verify how tools are being passed to `langchain_agent` (re-install HA plugin)
  - ➡️ Re-think routing and whether handlers should be tools
    - Might classify with proper classification model like [bart-large](https://huggingface.co/facebook/bart-large-mnli)
  - Continue to grow and mature the graph

# Use-cases
- Weather
- Spotify
- IoT devices

# Must have
- Pass HA information (i.e. entity states) in a stateful way
- Tool usage of tools exposed by HA (see `homeassistant/components/openai_conversation/entity.py:345`)
- Investigate how to initiate a conversation without a wake word
- Routing future
  - Smart LLM should also be able to take action on tools
  - Tools Route
  - Other features?
  - Tool-based/HA Apps features
    - Music Control
    - IoT Control (lights, etc.)
    - Weather

# Nice to have
- Auto identify langchain API host URL
