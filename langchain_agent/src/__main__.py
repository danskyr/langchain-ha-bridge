"""Main entry point for the LangChain server."""
import uvicorn


def main():
    """Run the server."""
    print("We startin' the serva")
    uvicorn.run(
        "langchain_agent.src.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=['langchain_agent/src']
    )


if __name__ == "__main__":
    main()
