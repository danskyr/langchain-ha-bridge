# Next time
- Review last session report.
- Test out tools and identify bugs as well as opportunities for improvement.
- Get weather working. Looks like it may need to call the `GetLiveContext` tool, but the tool description does not seem to give enough context that weather could be retrieved or specifically what context can be retrieved. Do we have a list of entities/devices when we are called? Maybe we can split this into multiple psuedo tools to raise awareness of options to the LLM.

```
> Tell me about the state of my home.
< I couldn't find a suitable tool to check the state of your home right now. Let's try another request. What would you like to know about your home?
```


# Last time
- Saw tool calls with invalid args. Now we validate args and send back to LLM with error for it to correct. 2 retries allowed.
- Improved brevity of tool result summary responses to the user.
- Moved tavily search from a node in the graph to another tool in the list of tools. Tool calls to tavily are handled within the API and not sent back to HA as it is a local tool.



# Big picture
1. Build a framework that acts as a solid basis for extending the code base to any use-case
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
