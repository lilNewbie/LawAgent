import logging
import multiprocessing

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from google.adk.tools import LongRunningFunctionTool

from scrapper import KanoonSpider

logger = logging.getLogger(__name__)

SCRAPE_TIMEOUT = 300  # seconds


def _run_scrapy_spider(tags_list: list, result_queue: multiprocessing.Queue, log_level_str: str):
    """Run KanoonSpider in an isolated process and put collected items into result_queue."""
    try:
        logging.basicConfig(
            level=getattr(logging, log_level_str, logging.INFO),
            format="[Scrapy Worker %(process)d] %(levelname)s: %(message)s",
        )

        collected_items = []

        settings = get_project_settings()
        settings.set("LOG_LEVEL", log_level_str, priority="cmdline")
        settings.set("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", priority="cmdline")
        settings.set("DOWNLOAD_DELAY", 1.0, priority="cmdline")
        settings.set("AUTOTHROTTLE_ENABLED", True, priority="cmdline")

        process = CrawlerProcess(settings)

        # Collect items yielded by the spider via the item_scraped signal.
        def collect_item(item, response, spider):
            collected_items.append(dict(item))

        crawler = process.create_crawler(KanoonSpider)
        crawler.signals.connect(collect_item, signal=scrapy.signals.item_scraped)

        logging.info(f"Scrapy process starting crawl for tags: {tags_list}")
        process.crawl(crawler, search_tags=tags_list)
        process.start()
        logging.info(f"Scrapy process finished. Collected {len(collected_items)} items.")

        result_queue.put(collected_items)

    except Exception as e:
        error_message = f"Scrapy process failed: {e}"
        logging.error(error_message, exc_info=True)
        result_queue.put(error_message)


def _upsert_items_to_qdrant(items: list) -> int:
    """Chunk, embed, and upsert scraped items into the COLLECTION_SCRAPED Qdrant collection.

    Imports are lazy so that the Scrapy subprocess (which forks this module) does not pay
    the cost of loading the embedding model and Qdrant client.
    """
    import uuid
    from qdrant_client.models import PointStruct
    from utils1 import (
        _embed_model, _qdrant_client, _ensure_collection, strip_html, chunk_text,
        COLLECTION_SCRAPED,
    )

    _ensure_collection(_qdrant_client, COLLECTION_SCRAPED)

    # First pass: collect all (source_url, chunks) pairs and build one flat chunk list.
    item_meta: list[tuple[str, list[str]]] = []
    all_chunks: list[str] = []

    for item in items:
        text = strip_html(
            item.get("html") or item.get("text") or item.get("content") or ""
        )
        source_url = item.get("source_url", "")
        if not text:
            continue
        chunks = chunk_text(text)
        item_meta.append((source_url, chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Single batch encode — most efficient use of SentenceTransformers.
    all_vectors = _embed_model.encode(all_chunks, convert_to_numpy=True)

    # Second pass: build PointStructs using the flat vector array via an offset counter.
    records: list[PointStruct] = []
    offset = 0
    for source_url, chunks in item_meta:
        for i, chunk in enumerate(chunks):
            records.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}#{i}")),
                    vector=all_vectors[offset].tolist(),
                    payload={
                        "chunk_text": chunk,
                        "source_url": source_url,
                        "chunk_index": i,
                    },
                )
            )
            offset += 1

    _qdrant_client.upsert(collection_name=COLLECTION_SCRAPED, points=records)
    logger.info("Upserted %d chunks into '%s'", len(records), COLLECTION_SCRAPED)
    return len(records)


def getdocs(tags: str) -> str:
    # Ensure tags is always a list, not a bare string.
    tags_list = [tags] if isinstance(tags, str) else list(tags)
    logger.info("ADK calling getdocs with tags: %s", tags_list)

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    log_level_str = "INFO"

    p = multiprocessing.Process(
        target=_run_scrapy_spider,
        args=(tags_list, result_queue, log_level_str),
    )
    p.start()
    p.join(timeout=SCRAPE_TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return f"Error: scraping timed out after {SCRAPE_TIMEOUT}s"

    result = result_queue.get()

    if isinstance(result, str):
        # Error path: the subprocess put an error message string.
        logger.error("Scrapy error: %s", result)
        return result

    # Success path: result is a list of scraped item dicts.
    count = _upsert_items_to_qdrant(result)
    return f"Scraping complete. {len(result)} items scraped, {count} chunks stored."


# Register getdocs as a LongRunningFunctionTool
getDocs = LongRunningFunctionTool(func=getdocs)
