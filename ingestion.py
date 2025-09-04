import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
from dotenv import load_dotenv

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", show_progress_bar = True, chunk_size = 50, retry_min_seconds =10)

vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding = embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=30, max_pages=1500)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
  """Split URLs into chunks of specified size."""
  chunks = []
  for i in range(0, len(urls), chunk_size):
    chunk = urls[i : i + chunk_size]
    chunks.append(chunk)
  return chunks

async def extract_batch(urls: List[str], batch_num : int) -> List[Dict[str, Any]]:
  """Extract documents from a batch of URLs"""
  try:
    log_info(
      f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs", 
      Colors.BLUE
    )
    docs = await tavily_extract.ainvoke(input = {"urls": urls})
    log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents")
    return docs
  except Exception as e:
    log_error(f"TavilyExtract: Failed to extract batch {batch_num}: {str(e)}")
    return []


async def async_extract(url_batches: List[List[str]]):
  log_header("DOCUMENT EXTRACTION PHASE")
  log_info(
    f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches", 
    Colors.DARKCYAN
  )

  tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

  results = await asyncio.gather(*tasks, return_exceptions=True)
  
  all_pages = []
  failed_batches = 0

  for i, result in enumerate(results):
    if isinstance(result, Exception):
      log_error(f"TavilyExtract: Failed to extract batch {i + 1}: {str(result)}")
      failed_batches += 1
    else:
      for extracted_page in result.get("results", []):
        document = Document(page_content=extracted_page["raw_content"], metadata={"source": extracted_page["url"]})
        all_pages.append(document)

  log_success(f"TavilyExtract: Successfully extracted {len(all_pages)} documents from {len(url_batches) - failed_batches} batches")

  if failed_batches > 0:
    log_warning(f"TavilyExtract: Failed to extract {failed_batches} batches")
  return all_pages


async def index_documents_async(documents: List[Document], batch_size: int = 50):
  """Process documents in batches asynchronously."""
  log_header("🚀 VECTOR STORAGE PHASE")
  log_info(
    f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store", 
    Colors.DARKCYAN
  )
  
  # Create batches
  batches = [
    documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
  ]

  log_info(
    f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
  )

  async def add_batch(batch: List[Document], batch_num: int):
    try:
      await vector_store.aadd_documents(batch)
      log_success(f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batch)} ({len(batch)}) documents)")

    except Exception as e:
      log_error(f"VectorStore Indexing: Failed to add batch {batch_num}: {str(e)}")
      return False
    return True

  # Process batches concurrently
  tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
  results = await asyncio.gather(*tasks, return_exceptions=True)

  # Count successful batches
  successful = sum(1 for result in results if result is True)

  if successful == len(batches):
    log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
  else:
    log_warning(f"VectorStore Indexing: Processes {successful}/{len(batches)} batches successfully")



async def main():
  """Main async function to orchestrate the entire process."""

  log_header("🚀  DOCUMENTATION INGESTION PIPELINE")

  log_info(
    "TavilyMap: Starting to map documentation structure from from https://python.langchain.com/",
    Colors.PURPLE 
  )

  site_map = tavily_map.invoke({
    "url": "https://python.langchain.com/",
  })

  log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site")

  url_batches = chunk_urls(site_map['results'], chunk_size=20)
  log_info(
    f"URL Processing: Split {len(site_map['results'])}) URLs into {len(url_batches)} batches", 
    Colors.BLUE
  )

  # Extract documents from  URLs
  all_docs = await async_extract(url_batches)

  # Split documents into chunks
  log_header("🚀  DOCUMENTATION CHUNKING PHASE")
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
  log_info(f"URLs Mapped: {len(site_map['results'])}")
  log_info(f"Documents Extracted: {len(all_docs)}")
  log_info(f"Chunks Created: {len(splitted_docs)}")


if __name__ == "__main__":
  asyncio.run(main())
