"""
Shared factory for creating LangChain agents and LLMs.

Available builders:
- build_gemini_agent()          — Google Gemini agent
- build_grok_llm()              — xAI Grok (requires XAI_API_KEY)
- build_openrouter_llm()        — OpenRouter (requires OPENROUTER_API_KEY, paid)
- build_groq_llm()              — Groq free tier (requires GROQ_API_KEY, 9K req/day)
- build_ollama_llm()            — Ollama local (free, requires local Ollama running)
- build_huggingface_llm()       — HuggingFace Inference (requires HUGGINGFACEHUB_API_TOKEN)
- build_reviewer_llm_with_fallback() — Auto-fallback through Groq → Ollama → HuggingFace
"""

import os
from typing import Callable, Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()


def build_gemini_agent(
    tools: Sequence[BaseTool | Callable],
    system_prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.1,
):
    """
    Build and return a compiled LangChain agent backed by a Gemini model.

    Args:
        tools:         List of LangChain tools (decorated with @tool or BaseTool subclasses).
        system_prompt: System instructions that define the agent's persona and capabilities.
        model:         Gemini model identifier (default: gemini-2.5-flash).
        temperature:   Sampling temperature for the LLM (default: 0.1).

    Returns:
        A compiled LangChain agent graph ready to be invoked with {"messages": [...]}.

    Example::

        from agent_factory import build_gemini_agent
        from langchain_core.tools import tool

        @tool
        def my_tool(x: str) -> str:
            \"\"\"Does something useful.\"\"\"
            return f"result: {x}"

        agent = build_gemini_agent(
            tools=[my_tool],
            system_prompt="You are a helpful assistant.",
        )
        response = agent.invoke({"messages": [HumanMessage(content="Hello")]})
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )

    return create_agent(llm, list(tools), system_prompt=system_prompt)


def build_grok_llm(
    model: str = "grok-3",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at xAI's Grok API endpoint.
    Uses XAI_API_KEY from the environment.

    Args:
        model:       Grok model name (default: grok-3).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A ChatOpenAI client configured for xAI.

    Example::

        from agent_factory import build_grok_llm
        llm = build_grok_llm()
        response = llm.invoke([HumanMessage(content="Review this trade...")])
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "XAI_API_KEY environment variable is not set. "
            "Get a key at https://console.x.ai/"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        temperature=temperature,
    )


def build_openrouter_llm(
    model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointed at OpenRouter's free tier.
    Uses OPENROUTER_API_KEY from the environment.

    Free models available:
    - nvidia/nemotron-3-ultra-550b-a55b:free (NVIDIA Nemotron 3 Ultra 550B) — default
    - meta-llama/llama-3-8b-instruct:free (Llama 3 8B)
    - mistralai/mistral-7b-instruct:free (Mistral 7B)
    - openchat/openchat-3.5:free (OpenChat 3.5)

    Args:
        model:       OpenRouter model name (default: nvidia/nemotron-3-ultra-550b-a55b:free).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A ChatOpenAI client configured for OpenRouter.

    Example::

        from agent_factory import build_openrouter_llm
        llm = build_openrouter_llm()
        response = llm.invoke([HumanMessage(content="Review this trade...")])
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a free key at https://openrouter.ai/"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def build_groq_llm(
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
):
    """
    Return a ChatGroq instance using Groq's free tier (9K requests/day).
    Uses GROQ_API_KEY from the environment. Model: llama-3.3-70b-versatile (currently active).

    Args:
        model:       Groq model name (default: mixtral-8x7b-32768).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A ChatGroq LLM client.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    return ChatGroq(model=model, api_key=api_key, temperature=temperature)


def build_ollama_llm(
    model: str = "qwen3:8b-q4_k_m",
    temperature: float = 0.1,
):
    """
    Return an Ollama LLM instance (completely free, runs locally).
    Requires Ollama to be installed and running: ollama serve

    Args:
        model:       Ollama model name (default: qwen3:8b-q4_k_m). Also: mistral, llama3.2, neural-chat, etc.
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A ChatOllama LLM client, or None if Ollama is not running.
    """
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        return None

    try:
        # Try to instantiate and test connection
        llm = Ollama(model=model, temperature=temperature, base_url="http://localhost:11434")
        # Quick validation - if this fails, Ollama is not running
        return llm
    except Exception:
        return None


def build_huggingface_llm(
    model: str = "meta-llama/Llama-2-7b-chat-hf",
    temperature: float = 0.1,
):
    """
    Return a HuggingFace Inference API LLM (free tier available).
    Uses HUGGINGFACEHUB_API_TOKEN from the environment.

    Args:
        model:       HuggingFace model ID (default: meta-llama/Llama-2-7b-chat-hf).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A HuggingFaceHub LLM client, or None if token not set.
    """
    try:
        from langchain_community.llms import HuggingFaceHub
    except ImportError:
        return None

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None

    try:
        return HuggingFaceHub(
            repo_id=model,
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": temperature},
        )
    except Exception:
        return None


def build_reviewer_llm_with_fallback(temperature: float = 0.1):
    """
    Build a reviewer LLM with automatic fallback chain.
    Tries free options in order:
      1. Groq (free tier: 9K req/day) - skipped if model deprecated
      2. Ollama (local, completely free)
      3. HuggingFace (free tier)

    Args:
        temperature: Sampling temperature (default: 0.1).

    Returns:
        A working LLM client, or raises error if none available.

    Raises:
        EnvironmentError: If no free LLM option is available.
    """
    # Try Groq first (fastest, generous free tier)
    print("  Trying Groq (free tier)...", flush=True)
    groq_llm = build_groq_llm(temperature=temperature)
    if groq_llm:
        # Validate Groq works by attempting a simple invocation
        try:
            from langchain_core.messages import HumanMessage
            _ = groq_llm.invoke([HumanMessage(content="test")])
            print("  ✓ Using Groq", flush=True)
            return groq_llm
        except Exception as e:
            if "decommissioned" in str(e) or "model" in str(e).lower():
                print(f"  Groq model unavailable. Trying Ollama...", flush=True)
            else:
                print(f"  Groq error: {str(e)[:80]}", flush=True)

    # Try Ollama second (local, completely free, no rate limits)
    print("  Trying Ollama (local)...", flush=True)
    ollama_llm = build_ollama_llm(temperature=temperature)
    if ollama_llm:
        try:
            from langchain_core.messages import HumanMessage
            _ = ollama_llm.invoke([HumanMessage(content="test")])
            print("  ✓ Using Ollama (local)", flush=True)
            return ollama_llm
        except Exception as e:
            print(f"  Ollama invoke failed: {str(e)[:60]}...", flush=True)

    # Try HuggingFace third (free tier available)
    print("  Trying HuggingFace (free tier)...", flush=True)
    hf_llm = build_huggingface_llm(temperature=temperature)
    if hf_llm:
        try:
            from langchain_core.messages import HumanMessage
            _ = hf_llm.invoke([HumanMessage(content="test")])
            print("  ✓ Using HuggingFace", flush=True)
            return hf_llm
        except Exception as e:
            print(f"  HuggingFace invoke failed: {str(e)[:60]}...", flush=True)

    # No free option available
    raise EnvironmentError(
        "No free LLM option available. Configure one of:\n"
        "  1. Install Ollama (https://ollama.ai) - run 'ollama pull mistral && ollama serve'\n"
        "  2. HUGGINGFACEHUB_API_TOKEN (get at https://huggingface.co/settings/tokens)\n"
        "  3. For Groq, current models are deprecated - check https://console.groq.com/docs/deprecations"
    )
