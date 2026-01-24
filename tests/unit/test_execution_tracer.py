"""Tests for langchain_agent.src.execution_tracer module."""
import os
import time
import pytest
from langchain_agent.src.execution_tracer import (
    NodeExecution,
    ExecutionTrace,
    ExecutionTracer,
    GRAPH_STRUCTURE,
    PARALLEL_NODES,
)


class TestNodeExecution:
    """Tests for the NodeExecution dataclass."""

    def test_creation(self):
        """NodeExecution should be creatable with required fields."""
        node = NodeExecution(name="router", start_time=1000.0)
        assert node.name == "router"
        assert node.start_time == 1000.0
        assert node.end_time is None
        assert node.metadata == {}

    def test_duration_ms_no_end_time(self):
        """Duration should be 0 if no end time."""
        node = NodeExecution(name="router", start_time=1000.0)
        assert node.duration_ms == 0.0

    def test_duration_ms_with_end_time(self):
        """Duration should be calculated correctly."""
        node = NodeExecution(name="router", start_time=1000.0, end_time=1001.5)
        assert node.duration_ms == 1500.0

    def test_metadata_default(self):
        """Metadata should default to empty dict."""
        node = NodeExecution(name="test", start_time=0)
        assert node.metadata == {}
        node.metadata["key"] = "value"
        assert node.metadata["key"] == "value"

    def test_metadata_custom(self):
        """Custom metadata should be stored."""
        node = NodeExecution(
            name="agent",
            start_time=0,
            metadata={"tool_calls": [{"name": "HassTurnOn"}]}
        )
        assert node.metadata["tool_calls"][0]["name"] == "HassTurnOn"


class TestExecutionTrace:
    """Tests for the ExecutionTrace dataclass."""

    def test_creation(self):
        """ExecutionTrace should be creatable with defaults."""
        trace = ExecutionTrace()
        assert trace.nodes == []
        assert trace.end_time is None
        assert trace.conversation_id == ""
        assert trace.start_time > 0

    def test_total_duration_no_end(self):
        """Total duration should be 0 if not ended."""
        trace = ExecutionTrace()
        assert trace.total_duration_s == 0.0

    def test_total_duration_with_end(self):
        """Total duration should be calculated correctly."""
        trace = ExecutionTrace(start_time=1000.0)
        trace.end_time = 1002.5
        assert trace.total_duration_s == 2.5

    def test_nodes_list(self):
        """Nodes should be appendable."""
        trace = ExecutionTrace()
        trace.nodes.append(NodeExecution(name="router", start_time=0))
        trace.nodes.append(NodeExecution(name="agent", start_time=0.1))
        assert len(trace.nodes) == 2

    def test_conversation_id(self):
        """Conversation ID should be settable."""
        trace = ExecutionTrace(conversation_id="test-123")
        assert trace.conversation_id == "test-123"


class TestExecutionTracer:
    """Tests for the ExecutionTracer class."""

    def test_init_enabled_by_default(self):
        """Tracer should be enabled by default (env var set in conftest)."""
        tracer = ExecutionTracer()
        assert tracer.enabled is True

    def test_start_trace(self):
        """Start trace should initialize current trace."""
        tracer = ExecutionTracer()
        tracer.start_trace("conv-123")
        assert tracer.current_trace is not None
        assert tracer.current_trace.conversation_id == "conv-123"

    def test_record_node_start(self):
        """Record node start should add node to trace."""
        tracer = ExecutionTracer()
        tracer.start_trace()
        tracer.record_node_start("router", {"route_types": ["iot"]})
        assert len(tracer.current_trace.nodes) == 1
        assert tracer.current_trace.nodes[0].name == "router"

    def test_record_node_end(self):
        """Record node end should set end time."""
        tracer = ExecutionTracer()
        tracer.start_trace()
        tracer.record_node_start("router")
        time.sleep(0.01)  # Small delay to ensure measurable duration
        tracer.record_node_end("router")
        assert tracer.current_trace.nodes[0].end_time is not None
        assert tracer.current_trace.nodes[0].duration_ms > 0

    def test_end_trace(self):
        """End trace should set end time and return trace."""
        tracer = ExecutionTracer()
        tracer.start_trace("test-conv")
        tracer.record_node_start("router")
        tracer.record_node_end("router")
        trace = tracer.end_trace()
        assert trace is not None
        assert trace.end_time is not None
        assert trace.total_duration_s > 0

    def test_disabled_tracer(self, monkeypatch):
        """Disabled tracer should not record anything."""
        monkeypatch.setenv("LOG_EXECUTION_PATH", "false")
        tracer = ExecutionTracer()
        assert tracer.enabled is False
        tracer.start_trace()
        assert tracer.current_trace is None
        tracer.record_node_start("router")
        result = tracer.end_trace()
        assert result is None

    def test_record_without_trace(self):
        """Recording without starting trace should not fail."""
        tracer = ExecutionTracer()
        # Don't start trace
        tracer.record_node_start("router")  # Should not raise
        tracer.record_node_end("router")  # Should not raise

    def test_multiple_nodes(self):
        """Multiple nodes should be recorded in order."""
        tracer = ExecutionTracer()
        tracer.start_trace()
        tracer.record_node_start("router")
        tracer.record_node_end("router")
        tracer.record_node_start("iot_handler")
        tracer.record_node_end("iot_handler")
        tracer.record_node_start("agent")
        tracer.record_node_end("agent")

        assert len(tracer.current_trace.nodes) == 3
        assert tracer.current_trace.nodes[0].name == "router"
        assert tracer.current_trace.nodes[1].name == "iot_handler"
        assert tracer.current_trace.nodes[2].name == "agent"


class TestExecutionTracerAscii:
    """Tests for ASCII rendering of execution traces."""

    def test_render_ascii_empty(self):
        """Empty trace should return empty string."""
        tracer = ExecutionTracer()
        tracer.start_trace()
        result = tracer.render_ascii()
        assert result == ""

    def test_render_ascii_with_nodes(self):
        """Trace with nodes should render ASCII art."""
        tracer = ExecutionTracer()
        tracer.start_trace("test-conv-123456")
        tracer.record_node_start("router", {"route_types": ["iot"]})
        tracer.record_node_end("router")
        tracer.record_node_start("iot_handler")
        tracer.record_node_end("iot_handler")
        tracer.record_node_start("aggregator")
        tracer.record_node_end("aggregator")
        tracer.record_node_start("formatter")
        tracer.record_node_end("formatter")
        tracer.end_trace()

        result = tracer.render_ascii()
        assert "router" in result
        assert "iot_handler" in result
        assert "formatter" in result
        assert "START" in result
        assert "END" in result

    def test_render_ascii_with_parallel_nodes(self):
        """Parallel nodes should be rendered correctly."""
        tracer = ExecutionTracer()
        tracer.start_trace("test-123")
        tracer.record_node_start("router")
        tracer.record_node_end("router")
        tracer.record_node_start("iot_handler")
        tracer.record_node_end("iot_handler")
        tracer.record_node_start("general_handler")
        tracer.record_node_end("general_handler")
        tracer.record_node_start("aggregator")
        tracer.record_node_end("aggregator")
        tracer.end_trace()

        result = tracer.render_ascii()
        assert "iot_handler" in result
        assert "general_handler" in result


class TestGraphStructure:
    """Tests for the GRAPH_STRUCTURE constant."""

    def test_graph_structure_exists(self):
        """GRAPH_STRUCTURE should be defined."""
        assert GRAPH_STRUCTURE is not None
        assert isinstance(GRAPH_STRUCTURE, dict)

    def test_all_nodes_defined(self):
        """All expected nodes should be in the structure."""
        expected_nodes = [
            "router", "iot_handler", "general_handler", "announcement",
            "aggregator", "agent", "tool_call_validation", "local_tools", "formatter"
        ]
        for node in expected_nodes:
            assert node in GRAPH_STRUCTURE

    def test_parallel_nodes_constant(self):
        """PARALLEL_NODES should contain the parallel handlers."""
        assert "iot_handler" in PARALLEL_NODES
        assert "general_handler" in PARALLEL_NODES
        assert "announcement" in PARALLEL_NODES
