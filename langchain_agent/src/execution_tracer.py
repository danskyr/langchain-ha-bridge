import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeExecution:
    name: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class ExecutionTrace:
    nodes: List[NodeExecution] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    conversation_id: str = ""

    @property
    def total_duration_s(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0


GRAPH_STRUCTURE = {
    "router": {"next": ["iot_handler", "general_handler", "announcement"], "parallel": True},
    "iot_handler": {"next": ["aggregator"], "parallel": False},
    "general_handler": {"next": ["aggregator"], "parallel": False},
    "announcement": {"next": ["aggregator"], "parallel": False},
    "aggregator": {"next": ["agent"], "parallel": False},
    "agent": {"next": ["tool_call_validation"], "parallel": False},
    "tool_call_validation": {"next": ["agent", "local_tools", "formatter", "END"], "parallel": False},
    "local_tools": {"next": ["agent"], "parallel": False},
    "formatter": {"next": ["END"], "parallel": False},
}

PARALLEL_NODES = {"iot_handler", "general_handler", "announcement"}


class ExecutionTracer:
    def __init__(self) -> None:
        self.enabled = os.getenv("LOG_EXECUTION_PATH", "true").lower() == "true"
        self.current_trace: Optional[ExecutionTrace] = None
        self._node_start_times: Dict[str, float] = {}

    def start_trace(self, conversation_id: str = "") -> None:
        if not self.enabled:
            return
        self.current_trace = ExecutionTrace(conversation_id=conversation_id)
        self._node_start_times = {}

    def record_node_start(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self.current_trace:
            return
        self._node_start_times[name] = time.time()
        self.current_trace.nodes.append(
            NodeExecution(name=name, start_time=time.time(), metadata=metadata or {})
        )

    def record_node_end(self, name: str) -> None:
        if not self.enabled or not self.current_trace:
            return
        end_time = time.time()
        for node in reversed(self.current_trace.nodes):
            if node.name == name and node.end_time is None:
                node.end_time = end_time
                break

    def end_trace(self) -> Optional[ExecutionTrace]:
        if not self.enabled or not self.current_trace:
            return None
        self.current_trace.end_time = time.time()
        return self.current_trace

    def render_ascii(self) -> str:
        if not self.current_trace or not self.current_trace.nodes:
            return ""

        visited = {n.name for n in self.current_trace.nodes}
        node_metadata = {n.name: n.metadata for n in self.current_trace.nodes}
        node_times = {n.name: n.duration_ms for n in self.current_trace.nodes}

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"EXECUTION PATH | Conversation: {self.current_trace.conversation_id[:12]}...")
        lines.append("=" * 80)
        lines.append("")

        lines.append("  START")
        lines.append("    │")
        lines.append("    ▼")

        if "router" in visited:
            lines.extend(self._render_node("router", node_metadata, node_times))
            lines.append("     │")

            parallel_visited = [n for n in ["iot_handler", "general_handler", "announcement"] if n in visited]
            if parallel_visited:
                lines.extend(self._render_parallel_nodes(parallel_visited, node_metadata, node_times))
                lines.append("")

        if "aggregator" in visited:
            lines.extend(self._render_centered_node("aggregator", node_metadata, node_times))
            lines.append("")

        if "agent" in visited:
            lines.extend(self._render_centered_node("agent", node_metadata, node_times))
            lines.append("")

        if "tool_call_validation" in visited:
            lines.extend(self._render_centered_node("tool_call_validation", node_metadata, node_times))
            lines.append("")

        if "local_tools" in visited:
            lines.extend(self._render_centered_node("local_tools", node_metadata, node_times))
            lines.append("                        │")
            lines.append("                        ▼")
            if "agent" in visited:
                lines.append("                  (back to agent)")
                lines.append("")

        if "formatter" in visited:
            lines.extend(self._render_centered_node("formatter", node_metadata, node_times))
            lines.append("")

        end_reason = self._get_end_reason(node_metadata)
        lines.append(f"                    [END] {end_reason}")
        lines.append("")

        tool_calls = self._count_tool_calls(node_metadata)
        lines.append(f"Total: {self.current_trace.total_duration_s:.2f}s | Nodes: {len(visited)} | Tool calls: {tool_calls}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _render_node(self, name: str, metadata: Dict[str, Dict], times: Dict[str, float]) -> List[str]:
        width = max(len(name) + 2, 11)
        meta_str = self._format_metadata(name, metadata.get(name, {}))
        time_str = f" ({times.get(name, 0):.0f}ms)" if times.get(name, 0) > 0 else ""

        lines = [
            f"┌{'─' * width}┐",
            f"│ {name.center(width - 2)} │{meta_str}{time_str}",
            f"└{'─' * (width // 2)}┬{'─' * (width - width // 2 - 1)}┘",
        ]
        return lines

    def _render_centered_node(self, name: str, metadata: Dict[str, Dict], times: Dict[str, float]) -> List[str]:
        width = max(len(name) + 4, 14)
        meta_str = self._format_metadata(name, metadata.get(name, {}))
        time_str = f" ({times.get(name, 0):.0f}ms)" if times.get(name, 0) > 0 else ""

        padding = " " * 14
        lines = [
            f"{padding}     │",
            f"{padding}     ▼",
            f"{padding}┌{'─' * width}┐",
            f"{padding}│ {name.center(width - 2)} │{meta_str}{time_str}",
            f"{padding}└{'─' * width}┘",
        ]
        return lines

    def _render_parallel_nodes(self, nodes: List[str], metadata: Dict[str, Dict], times: Dict[str, float]) -> List[str]:
        if not nodes:
            return []

        widths = [max(len(n) + 2, 12) for n in nodes]

        lines: List[str] = []

        if len(nodes) > 1:
            branch = "     ├" + "─" * 8
            for i in range(len(nodes) - 1):
                branch += "┬" + "─" * (widths[i] + 2)
            branch += "┐"
            lines.append(branch)

        arrows = "     "
        for i, (n, w) in enumerate(zip(nodes, widths)):
            if i == 0:
                arrows += "▼" + " " * (w + 2)
            else:
                arrows += "▼" + " " * (w + 2)
        lines.append(arrows.rstrip())

        top_line = "  "
        for w in widths:
            top_line += "┌" + "─" * w + "┐  "
        lines.append(top_line.rstrip())

        mid_line = "  "
        for n, w in zip(nodes, widths):
            mid_line += "│" + n.center(w) + "│  "
        lines.append(mid_line.rstrip())

        bot_line = "  "
        for w in widths:
            bot_line += "└" + "─" * (w // 2) + "┬" + "─" * (w - w // 2 - 1) + "┘  "
        lines.append(bot_line.rstrip())

        if len(nodes) > 1:
            merge = "     "
            for i, w in enumerate(widths):
                if i == 0:
                    merge += "└" + "─" * (w // 2 + 1)
                elif i == len(widths) - 1:
                    merge += "─" * (w // 2 + 1) + "┘"
                else:
                    merge += "┴" + "─" * (w + 2)
            lines.append(merge)
            lines.append("                         │")

        return lines

    def _format_metadata(self, node_name: str, meta: Dict[str, Any]) -> str:
        if not meta:
            return ""

        if node_name == "router" and "route_types" in meta:
            return f"  route_types: {meta['route_types']}"
        if node_name == "agent" and "tool_calls" in meta:
            tools = [tc.get("name", "?") for tc in meta.get("tool_calls", [])]
            return f"  tools: {tools}"
        if node_name == "tool_call_validation" and "decision" in meta:
            return f"  → {meta['decision']}"

        return ""

    def _get_end_reason(self, metadata: Dict[str, Dict]) -> str:
        validation_meta = metadata.get("tool_call_validation", {})
        decision = validation_meta.get("decision", "")
        if decision == "ha_tools":
            return "(returning tool calls to HA)"
        elif decision == "formatter":
            return "(formatted response)"
        return ""

    def _count_tool_calls(self, metadata: Dict[str, Dict]) -> int:
        agent_meta = metadata.get("agent", {})
        return len(agent_meta.get("tool_calls", []))
