# Current
- Bootstrap clean routing so that we can easily extend it to more complex use-cases in the future
  - Implement agent
  - ➡️ Add response Node (unifies responses in a consistent tone and format/feel)
  - Able to answer basic questions (don't route everything to search)
  - News? (search for stuff)
  - Routes we want:
    - Smart LLM
    - Search


# Must have
- Tool usage (see `homeassistant/components/openai_conversation/entity.py:345`)
- Pass HA information (i.e. entity states) in a stateful way
- Investigate how to initiate a conversation without a wake word
- Routing future
  - Smart LLM should also be able to take action on tools
  - Tools Route
  - Other features?
  - Tool-based/HA Apps features
    - Music Control
    - IoT Control (lights, etc.)
    - Weather

# Should have

# Nice to have
- Auto identify langchain API host URL
