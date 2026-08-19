import os
import asyncio
import logging
from pathlib import Path
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.cloud import storage
from utils0 import getDocs
from utils1 import loadFile, fetch_from_qdrant, fetch_from_qdrant2

_ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")

MODEL_FAST = os.environ.get("MODEL_FAST", "gemini-3.6-flash")
MODEL_RESEARCH = os.environ.get("MODEL_RESEARCH", "gemini-3.6-flash")


def _validate_env() -> None:
    missing = [k for k in ("GOOGLE_API_KEY",) if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set them before running: export GOOGLE_API_KEY=<your-key>"
        )


_validate_env()

# --- Agent 1: Keyword Extractor ---
keyword_extractor_agent = LlmAgent(
    name="KeywordExtractorAgent",
    model=MODEL_FAST,
    instruction='''
        Extract exactly ONE legal keyword that best represents the core context of the case
        summary provided in the user message.
        The keyword must be a recognized legal term and directly reflect the subject matter
        or nature of the legal case.
        Return only the keyword — no explanation, formatting, or punctuation.
        If no valid legal keyword is found, return: NO_KEYWORDS_FOUND
    ''',
    description="Analyzes legal case summaries to extract core keywords.",
    output_key="extracted_keywords",
)


# --- Agent 2: Document Fetcher ---
document_fetcher_agent = LlmAgent(
    name="DocumentFetcherAgent",
    model=MODEL_FAST,
    instruction='''
        You have access to state['extracted_keywords'] which contains keywords.
        Call the `getDocs` tool with it as the `tags` argument (in a list) ONLY ONCE.
        Do NOT generate any explanations or additional text.
        If input is "NO_KEYWORDS_FOUND", respond with:
        Error: No keywords provided for document fetching.
    ''',
    description="Uses extracted keywords to fetch legal documents.",
    tools=[getDocs],
    output_key="document_fetch_status",
)

# --- Initialiser Pipeline: Extract keyword → Fetch documents
initialiser_pipeline = SequentialAgent(
    name="InitialiserPipeline",
    description="Extracts keyword and fetches legal documents.",
    sub_agents=[
        keyword_extractor_agent,
        document_fetcher_agent
    ]
)

# --- Research Agents for prosecution and defence
prosecution_research_agent = LlmAgent(
    name="Prosecution_Research",
    model=MODEL_RESEARCH,
    instruction=loadFile(str(_ROOT / "instructions" / "forInstruction.txt")),
    output_key="for_prosecutor",
    tools=[fetch_from_qdrant, fetch_from_qdrant2],
)

defence_research_agent = LlmAgent(
    name="Defence_Research",
    model=MODEL_RESEARCH,
    instruction=loadFile(str(_ROOT / "instructions" / "againstInstruction.txt")),
    output_key="for_defence",
    tools=[fetch_from_qdrant, fetch_from_qdrant2],
)

parallel_research_agent = ParallelAgent(
    name="ParallelVectorResearch",
    description="Runs prosecution and defence research in parallel.",
    sub_agents=[prosecution_research_agent, defence_research_agent]
)

# --- Root Agent: Orchestrates everything
root_agent = SequentialAgent(
    name="LegalResearchOrchestrator",
    description="Runs the full legal research pipeline.",
    sub_agents=[
        initialiser_pipeline,
        parallel_research_agent
    ]
)


async def save_raw_text_async_gcs(
    raw_str: str,
    author: str,
    run_prefix: str = "",
    gcs_client: storage.Client = None,
) -> None:
    if not raw_str:
        logger.warning("Empty string passed to save_raw_text_async_gcs.")
        return

    if not gcs_client:
        logger.warning("GCS client not provided (GCS_BUCKET_NAME unset?), skipping upload.")
        return

    safe_author = author.replace(" ", "_").replace("/", "_")
    prefix = f"{run_prefix}/" if run_prefix else ""
    gcs_path = f"raw_texts/{prefix}{safe_author}.txt"

    try:
        bucket = gcs_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        await asyncio.to_thread(blob.upload_from_string, raw_str.strip(), content_type="text/plain")
        logger.info("Saved to GCS: %s", gcs_path)
    except Exception as e:
        logger.error("Failed to save to GCS: %s: %s", gcs_path, e)


async def run_pipeline(case_summary: str) -> None:
    """
    Entry point for the legal research pipeline.
    Pass the full case summary text as `case_summary`.
    """
    from datetime import datetime
    logging.basicConfig(level=logging.INFO)

    gcs_client = storage.Client() if BUCKET_NAME else None
    runner = InMemoryRunner(agent=root_agent, app_name="lawAgents")
    run_prefix = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    session = await runner.session_service.create_session(
        app_name="lawAgents",
        user_id="user_1",
        state={
            "role": "researchRoom",
            "case_summary": case_summary,
        }
    )
    logger.info("Session created: %s", session.id)

    # Pass the case summary as the triggering user message so
    # KeywordExtractorAgent can read it from conversation history.
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text=case_summary)]),
    ):
        if event.content and event.content.parts:
            raw_str = event.content.parts[0].text
            if raw_str and event.author in ["Prosecution_Research", "Defence_Research"]:
                await save_raw_text_async_gcs(raw_str, event.author, run_prefix, gcs_client)


if __name__ == "__main__":
    # Replace this string with the actual case summary when running directly.
    sample_case = "The accused is charged with criminal breach of trust under Section 406 IPC."
    asyncio.run(run_pipeline(sample_case))