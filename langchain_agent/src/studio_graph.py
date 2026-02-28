"""Entry point for LangGraph Studio.

Exposes the compiled graph from LangChainRouterAgentV2 as a module-level variable
so that `langgraph dev` can discover and serve it.

LangGraph Studio provides its own checkpointer, so we compile
the graph without one here.
"""

from langchain_agent.src.router_agent_v2 import LangChainRouterAgentV2

agent = LangChainRouterAgentV2()
graph = agent._build_graph()
