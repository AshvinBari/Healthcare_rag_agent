# hospital_agent.py
import os
from typing import Any
from tavily import TavilyClient

# LangChain imports (for optional standalone agent)
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Load Tavily & OpenAI keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Defensive Tavily client init (may be None if key missing)
_tavily_client = None
if TAVILY_API_KEY:
    try:
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    except Exception as e:
        # keep client None and surface errors later
        _tavily_client = None


def hospital_search(query: str) -> str:
    """
    Query Tavily to find hospitals/clinics/emergency centers.
    Returns a formatted human-friendly string.
    If Tavily key is missing or an error happens, returns a helpful message.
    """
    if not TAVILY_API_KEY or _tavily_client is None:
        return "Tavily API key missing or Tavily client initialization failed. Set TAVILY_API_KEY in your environment to enable hospital search."

    try:
        # Build a concise search prompt
        search_prompt = (
            f"Find hospitals, clinics, or emergency centers for: {query}. "
            "Return up to 5 results with name, short description, address or city (if available), and a link or contact if available."
        )

        # SDK call — defensive parsing of response shapes
        resp = _tavily_client.search(search_prompt, max_results=6, include_raw_content=False)

        if not resp:
            return "No results returned by Tavily for that query."

        # Normalise many possible shapes
        results = []
        if isinstance(resp, dict):
            # try common keys
            for k in ("results", "data", "items"):
                if k in resp and isinstance(resp[k], (list, tuple)):
                    results = resp[k]
                    break
            if not results:
                # maybe resp itself is a single result dict
                results = [resp]
        elif isinstance(resp, (list, tuple)):
            results = list(resp)
        else:
            results = [resp]

        if len(results) == 0:
            return "No hospital results found."

        out_lines = []
        for idx, item in enumerate(results[:5], start=1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or "Untitled"
                url = item.get("url") or item.get("link") or ""
                snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
                # Some SDK results include structured fields for address/phone
                address = item.get("address") or item.get("location") or ""
                phone = item.get("phone") or item.get("contact") or ""
            else:
                # fallback string representation
                title = str(item)[:80]
                url = ""
                snippet = ""
                address = ""
                phone = ""

            line = f"🏥 {idx}. {title}\n"
            if snippet:
                line += f"{snippet}\n"
            if address:
                line += f"📍 {address}\n"
            if phone:
                line += f"📞 {phone}\n"
            if url:
                line += f"🔗 {url}\n"
            line += "\n"
            out_lines.append(line)

        return "\n".join(out_lines)

    except Exception as e:
        return f"Tavily search failed: {e}"


# Create a LangChain Tool object that other code (app.py) can import.
hospital_tool = Tool(
    name="HospitalFinder",
    func=hospital_search,
    description="Find hospitals, clinics and emergency centers using Tavily Search API."
)


# Optional: provide a standalone agent for testing (uses OpenAI key)
def create_standalone_hospital_agent():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment for standalone hospital agent.")
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0, model_name="gpt-4o-mini")
    agent = initialize_agent(
        tools=[hospital_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent


if __name__ == "__main__":
    # Quick CLI test: run the standalone agent interactively
    print("Hospital Finder CLI — this uses Tavily (TAVILY_API_KEY) and a small agent (OPENAI_API_KEY required).")
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY not set. The hospital tool will not function.")
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Standalone agent cannot run but hospital_search can still be tested directly.")

    # If OpenAI key present, create agent; otherwise let user call hospital_search directly.
    agent = None
    if OPENAI_API_KEY:
        try:
            agent = create_standalone_hospital_agent()
        except Exception as e:
            print("Could not create standalone agent:", e)
            agent = None

    while True:
        q = input("\nEnter hospital query (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if agent:
            try:
                print("Agent thinking...")
                out = agent.run(q)
                print("\n" + out + "\n")
            except Exception as e:
                print("Agent error:", e)
                print("Falling back to direct hospital_search()")
                print(hospital_search(q))
        else:
            print(hospital_search(q))
