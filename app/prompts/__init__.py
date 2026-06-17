from langfuse import get_client


def get(name: str) -> str:
    """Fetch the production-labelled prompt from LangFuse."""
    return get_client().get_prompt(name).compile()
