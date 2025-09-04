import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
)

vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME_CRAWL"), embedding=embeddings
)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("🚀 VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batch)} ({len(batch)}) documents)"
            )

        except Exception as e:
            log_error(
                f"VectorStore Indexing: Failed to add batch {batch_num}: {str(e)}"
            )
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processes {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""

    log_header("🚀  DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentation relevant to ai agents",
            "max_depth": 5,
        }
    )

    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )

    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("🚀 DOCUMENTATION CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline completed successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f"URLs Mapped: {len(tavily_crawl_results)}")
    log_info(f"Documents Extracted: {len(all_docs)}")
    log_info(f"Chunks Created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
