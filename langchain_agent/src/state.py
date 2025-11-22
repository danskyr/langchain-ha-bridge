from typing import TypedDict, Optional, Dict, Any, List, Annotated
import operator
from langchain_core.messages import BaseMessage


class RouterState(TypedDict):
    """State for the router agent with automatic message merging."""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    route_types: List[str]
    handler_responses: Annotated[List[Dict[str, Any]], operator.add]
    final_response: Optional[str]
    tools: Optional[List[Dict[str, Any]]]
    validation_attempts: int
    continue_conversation: Optional[bool]
