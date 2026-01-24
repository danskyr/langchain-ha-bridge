"""Tests for custom_components.langchain_conversation.utils module."""
import pytest
from langchain_conversation.utils import get_host_from_url


class TestGetHostFromUrl:
    """Tests for the get_host_from_url function."""

    def test_simple_url(self):
        """Should extract host from simple URL."""
        result = get_host_from_url("http://localhost:8000")
        assert result == "localhost:8000"

    def test_https_url(self):
        """Should extract host from HTTPS URL."""
        result = get_host_from_url("https://example.com")
        assert result == "example.com"

    def test_url_with_path(self):
        """Should extract host ignoring path."""
        result = get_host_from_url("http://192.168.1.100:8000/v1/completions")
        assert result == "192.168.1.100:8000"

    def test_url_with_port(self):
        """Should include port in result."""
        result = get_host_from_url("http://myserver:3000")
        assert result == "myserver:3000"

    def test_url_without_port(self):
        """Should work without explicit port."""
        result = get_host_from_url("https://api.example.com/endpoint")
        assert result == "api.example.com"

    def test_ip_address(self):
        """Should work with IP address."""
        result = get_host_from_url("http://10.0.0.1:8080")
        assert result == "10.0.0.1:8080"

    def test_invalid_url_returns_empty(self):
        """Invalid URL returns empty netloc (no exception raised by urlparse)."""
        result = get_host_from_url("not a url")
        assert result == ""  # urlparse returns empty netloc for invalid URLs

    def test_empty_string_returns_empty(self):
        """Empty string returns empty netloc."""
        result = get_host_from_url("")
        assert result == ""

    def test_none_like_returns_empty(self):
        """Malformed URLs return empty netloc."""
        result = get_host_from_url("://")
        assert result == ""  # urlparse returns empty netloc

    def test_subdomain(self):
        """Should include subdomain."""
        result = get_host_from_url("https://api.prod.example.com")
        assert result == "api.prod.example.com"

    def test_url_with_credentials(self):
        """Should handle URL with credentials."""
        result = get_host_from_url("http://user:pass@example.com:8000")
        assert result == "user:pass@example.com:8000"

    def test_localhost(self):
        """Should work with localhost."""
        result = get_host_from_url("http://localhost")
        assert result == "localhost"

    def test_ipv6_address(self):
        """Should handle IPv6 address."""
        result = get_host_from_url("http://[::1]:8000")
        assert result == "[::1]:8000"
