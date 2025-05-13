"""Main entry point for the LangChain server."""
import uvicorn
from .server import app

def main():
    """Run the server."""
    print("We startin' the serva")
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
