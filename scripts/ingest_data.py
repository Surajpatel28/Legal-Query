"""
BNS Data Ingestion Pipeline

Scrapes, cleans, chunks, and uploads Bharatiya Nyaya Sanhita sections to Qdrant.

Usage:
    python -m scripts.ingest_data
"""
import json
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import COLLECTION_NAME
from src.database import get_qdrant_client, get_embeddings, get_vectorstore
from qdrant_client import models

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def scrape_single_url(url, section_type="BNS"):
    """
    Scrape a single BNS section
    
    Args:
        url: URL to scrape
        section_type: Law code type (default: "BNS")
    
    Returns:
        Dictionary with scraped data or None if error
    """
    try:
        section = url.split('/')[-2]
        print(f"Scraping section {section}...")

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        p_elements = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        h2_elements = [h2.get_text().strip() for h2 in soup.find_all('h2') if h2.get_text().strip()]
        h3_elements = [h3.get_text().strip() for h3 in soup.find_all('h3') if h3.get_text().strip()]
        li_elements = [li.get_text().strip() for li in soup.find_all('li') if li.get_text().strip()]

        combined_content = '\n'.join(h2_elements + h3_elements + p_elements + li_elements)

        return {
            'law_code': section_type,
            'section': section,
            'content': combined_content,
            'source_url': url
        }
    except Exception as e:
        print(f"Error scraping section {section}: {str(e)}")
        return None

def scrape_sections(start=1, end=358):
    """
    Scrape BNS sections from devgan.in
    
    Args:
        start: Starting section number (default: 1)
        end: Ending section number (default: 358)
    
    Returns:
        List of scraped section data
    """
    # Generate URLs
    base_url = "https://devgan.in/bns/section/{}/"
    urls = [base_url.format(i) for i in range(start, end + 1)]

    # Scrape with 10 threads at once (fast but not overwhelming the server)
    print(f"Starting to scrape {len(urls)} sections (from {start} to {end})...\n")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(scrape_single_url, urls))

    # Remove None values (errors)
    scraped_data = [d for d in results if d is not None]
    failed_count = len(results) - len(scraped_data)

    print(f"\nSuccessfully scraped {len(scraped_data)} sections")
    if failed_count > 0:
        print(f"Failed: {failed_count} sections")

    # Save raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DIR / "bns_sections.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")
    return scraped_data

def clean_data(raw_data):
    """
    Clean scraped data by removing navigation elements and normalizing whitespace
    
    Args:
        raw_data: List of scraped section dictionaries
    
    Returns:
        List of cleaned section data
    """
    print(f"Cleaning {len(raw_data)} sections...")

    # Define cleaning patterns
    pattern_nav = re.compile(r'\b(?:Home|Top|Back|Prev|Index|Next)\b', re.IGNORECASE)

    def clean_text(text):
        text = text.replace("\n", " ").replace("\t", " ")
        text = pattern_nav.sub(" ", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Clean all sections
    cleaned_data = []
    for item in raw_data:
        cleaned_item = item.copy()
        cleaned_item["content"] = clean_text(item["content"])
        cleaned_data.append(cleaned_item)

    # Save cleaned data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "cleaned_final.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned and saved {len(cleaned_data)} sections to {output_file}")
    return cleaned_data

def chunk_documents(data, chunk_size=2000, overlap=400):
    """
    Split documents into chunks for embedding
    
    Args:
        data: List of cleaned section data
        chunk_size: Maximum chunk size in characters (default: 2000)
        overlap: Overlap between chunks (default: 400)
    
    Returns:
        List of Document chunks
    """
    print(f"Chunking {len(data)} sections with size={chunk_size}, overlap={overlap}...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Create a Langchain document for each JSON record 
    documents = []
    for record in data:
        document = Document(
            page_content=record['content'],
            metadata={'section': record['section']}
        )
        documents.append(document)

    chunks = splitter.split_documents(documents)

    section_chunk_count = {}
    for chunk in chunks:
        section = chunk.metadata['section']
        
        # Count chunks per section
        if section not in section_chunk_count:
            section_chunk_count[section] = 0
        
        # Create deterministic integer ID: section_number * 1000 + chunk_index
        # Example: Section 103, chunk 2 → 103002
        section_num = int(section)
        chunk_index = section_chunk_count[section]
        chunk.id = section_num * 1000 + chunk_index
        
        section_chunk_count[section] += 1
    
    print(f"Created {len(chunks)} chunks from {len(data)} sections")
    return chunks

def upload_to_qdrant(docs, vectorstore, client):
    """
    Upload/update document chunks to Qdrant vector database
    
    Args:
        docs: List of Document chunks
        vectorstore: QdrantVectorStore instance
        client: QdrantClient instance
    """
    print(f"\nUploading {len(docs)} chunks to Qdrant...")
    
    # Check if collection exists
    if verify_collection(client, COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists!")
        
        # Ask user what to do
        response = input("Do you want to delete and recreate it? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print(f"Deleting collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection deleted")
        else:
            print("Operation cancelled. Exiting program.")
            exit(0)
    
    # Create new collection
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    
    # Extract IDs from documents
    ids = [doc.id for doc in docs]
    
    # Use vectorstore with our deterministic IDs
    vectorstore.add_documents(documents=docs, ids=ids)
    
    print(f"✓ Successfully uploaded {len(docs)} documents to Qdrant")

def verify_collection(client, collection_name):
    """
    Check if a collection exists in Qdrant
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to check
    
    Returns:
        Boolean indicating if collection exists
    """
    return client.collection_exists(collection_name=collection_name)

def run_full_pipeline(start=1, end=358):
    """
    Run the complete data ingestion pipeline
    
    Args:
        start: Starting section number
        end: Ending section number
    """
    print("STARTING FULL INGESTION PIPELINE")
    
    # Step 1: Scrape
    print("\n[1/4] Scraping sections...")
    scraped_data = scrape_sections(start, end)
    
    # Step 2: Clean
    print("\n[2/4] Cleaning data...")
    cleaned_data = clean_data(scraped_data)
    
    # Step 3: Chunk
    print("\n[3/4] Chunking documents...")
    chunks = chunk_documents(cleaned_data)
    
    # Step 4: Upload (lazy load clients)
    print("\n[4/4] Uploading to Qdrant...")
    print("Initializing Qdrant client and embeddings...")
    client = get_qdrant_client()
    embedding = get_embeddings()
    vectorstore = get_vectorstore(client, embedding)
    
    upload_to_qdrant(chunks, vectorstore, client)
    
    print("PIPELINE COMPLETE")
    
if __name__ == "__main__":
    run_full_pipeline()