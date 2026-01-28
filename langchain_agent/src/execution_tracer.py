import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STALE_TRACE_TTL_SECONDS = 300  # 5 minutes


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
    rounds: int = 1

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
        self._active_traces: Dict[str, ExecutionTrace] = {}
        self._trace_timestamps: Dict[str, float] = {}

    def _cleanup_stale_traces(self) -> None:
        now = time.time()
        stale_keys = [
            k for k, ts in self._trace_timestamps.items()
            if now - ts > STALE_TRACE_TTL_SECONDS
        ]
        for k in stale_keys:
            self._active_traces.pop(k, None)
            self._trace_timestamps.pop(k, None)

    def start_trace(self, conversation_id: str = "") -> None:
        if not self.enabled:
            return

        self._cleanup_stale_traces()

        if conversation_id and conversation_id in self._active_traces:
            self.current_trace = self._active_traces[conversation_id]
            self.current_trace.rounds += 1
            self._trace_timestamps[conversation_id] = time.time()
        else:
            self.current_trace = ExecutionTrace(conversation_id=conversation_id)
            if conversation_id:
                self._active_traces[conversation_id] = self.current_trace
                self._trace_timestamps[conversation_id] = time.time()

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

    def record_ha_roundtrip(self, tool_calls: List[Dict[str, Any]]) -> None:
        if not self.enabled or not self.current_trace:
            return
        tool_names = [tc.get("tool_name", tc.get("name", "unknown")) for tc in tool_calls]
        self.current_trace.nodes.append(
            NodeExecution(
                name="ha_tool_execution",
                start_time=time.time(),
                end_time=time.time(),
                metadata={"tool_names": tool_names}
            )
        )

    def end_trace(self, final: bool = True) -> Optional[ExecutionTrace]:
        if not self.enabled or not self.current_trace:
            return None

        if not final:
            self._trace_timestamps[self.current_trace.conversation_id] = time.time()
            return None

        self.current_trace.end_time = time.time()
        conv_id = self.current_trace.conversation_id
        self._active_traces.pop(conv_id, None)
        self._trace_timestamps.pop(conv_id, None)
        return self.current_trace

    def render_ascii(self) -> str:
        if not self.current_trace or not self.current_trace.nodes:
            return ""

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"EXECUTION PATH | Conversation: {self.current_trace.conversation_id[:12]}...")
        lines.append("=" * 80)
        lines.append("")
        lines.append("  START")
        lines.append("    │")
        lines.append("    ▼")

        nodes = self.current_trace.nodes
        i = 0
        # Render the initial router + parallel handlers section
        while i < len(nodes) and nodes[i].name == "router":
            self._append_node_lines(lines, nodes[i])
            lines.append("     │")
            i += 1

            # Collect parallel handler nodes
            parallel_batch: List[NodeExecution] = []
            while i < len(nodes) and nodes[i].name in PARALLEL_NODES:
                parallel_batch.append(nodes[i])
                i += 1
            if parallel_batch:
                lines.extend(self._render_parallel_executions(parallel_batch))
                lines.append("")
            break

        # Render remaining nodes sequentially
        while i < len(nodes):
            node = nodes[i]
            if node.name == "ha_tool_execution":
                lines.extend(self._render_ha_boundary(node))
                lines.append("")
            else:
                lines.extend(self._render_centered_execution(node))
                lines.append("")
            i += 1

        end_reason = self._get_end_reason_from_nodes(nodes)
        lines.append(f"                    [END] {end_reason}")
        lines.append("")

        tool_calls = self._count_tool_calls_from_nodes(nodes)
        rounds = self.current_trace.rounds
        node_count = len(nodes)
        lines.append(
            f"Total: {self.current_trace.total_duration_s:.2f}s | "
            f"Rounds: {rounds} | Nodes: {node_count} | Tool calls: {tool_calls}"
        )
        lines.append("=" * 80)

        return "\n".join(lines)

    def _append_node_lines(self, lines: List[str], node: NodeExecution) -> None:
        width = max(len(node.name) + 2, 11)
        meta_str = self._format_metadata(node.name, node.metadata)
        time_str = f" ({node.duration_ms:.0f}ms)" if node.duration_ms > 0 else ""

        lines.append(f"┌{'─' * width}┐")
        lines.append(f"│ {node.name.center(width - 2)} │{meta_str}{time_str}")
        lines.append(f"└{'─' * (width // 2)}┬{'─' * (width - width // 2 - 1)}┘")

    def _render_centered_execution(self, node: NodeExecution) -> List[str]:
        width = max(len(node.name) + 4, 14)
        meta_str = self._format_metadata(node.name, node.metadata)
        time_str = f" ({node.duration_ms:.0f}ms)" if node.duration_ms > 0 else ""

        padding = " " * 14
        return [
            f"{padding}     │",
            f"{padding}     ▼",
            f"{padding}┌{'─' * width}┐",
            f"{padding}│ {node.name.center(width - 2)} │{meta_str}{time_str}",
            f"{padding}└{'─' * width}┘",
        ]

    def _render_ha_boundary(self, node: NodeExecution) -> List[str]:
        tool_names = node.metadata.get("tool_names", [])
        label = f"   HA executes: {', '.join(tool_names)}   "
        width = max(len(label), 20)
        label = label.center(width)

        padding = " " * 14
        return [
            f"{padding}     │",
            f"{padding}     ▼",
            f"{padding}╔{'═' * width}╗",
            f"{padding}║{label}║",
            f"{padding}╚{'═' * width}╝",
        ]

    def _render_parallel_executions(self, nodes: List[NodeExecution]) -> List[str]:
        if not nodes:
            return []

        widths = [max(len(n.name) + 2, 12) for n in nodes]
        lines: List[str] = []

        if len(nodes) > 1:
            branch = "     ├" + "─" * 8
            for i in range(len(nodes) - 1):
                branch += "┬" + "─" * (widths[i] + 2)
            branch += "┐"
            lines.append(branch)

        arrows = "     "
        for _i, (_n, w) in enumerate(zip(nodes, widths)):
            arrows += "▼" + " " * (w + 2)
        lines.append(arrows.rstrip())

        top_line = "  "
        for w in widths:
            top_line += "┌" + "─" * w + "┐  "
        lines.append(top_line.rstrip())

        mid_line = "  "
        for n, w in zip(nodes, widths):
            mid_line += "│" + n.name.center(w) + "│  "
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

    def _get_end_reason_from_nodes(self, nodes: List[NodeExecution]) -> str:
        for node in reversed(nodes):
            if node.name == "tool_call_validation":
                decision = node.metadata.get("decision", "")
                if decision == "ha_tools":
                    return "(returning tool calls to HA)"
                elif decision == "formatter":
                    return "(formatted response)"
                break
        return ""

    def _count_tool_calls_from_nodes(self, nodes: List[NodeExecution]) -> int:
        count = 0
        for node in nodes:
            if node.name == "agent" and "tool_calls" in node.metadata:
                count += len(node.metadata["tool_calls"])
        return count
