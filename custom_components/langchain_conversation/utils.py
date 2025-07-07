def get_host_from_url(self, url):
    """Extract host from URL for title."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return "Remote Service"
