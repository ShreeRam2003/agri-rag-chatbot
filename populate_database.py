# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import hashlib
import time
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma


# Ensure absolute path for database storage
CHROMA_PATH = os.path.abspath("chroma_db")
DATA_PATH = os.path.abspath("data")  # Ensure absolute path for PDFs

print(f"Using ChromaDB path: {CHROMA_PATH}")
print(f"Looking for documents in: {DATA_PATH}")

def main():
    """Main function to handle document processing and database operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load and process documents
    documents = load_documents()
    if not documents:
        print("No documents found! Check the 'data' directory.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")
    chunks = split_documents(documents)
    
    if not chunks:
        print("No chunks created! Check document splitting parameters.")
        return
        
    add_to_chroma_with_batches(chunks)


def load_documents():
    """Loads all PDFs from the 'data' directory."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at {DATA_PATH}")
        return []

    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def split_documents(documents: list[Document]):
    """Splits documents into smaller chunks for better embeddings."""
    if not documents:
        return []
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")  
        if chunks:
            print(f"First chunk preview: {chunks[0].page_content[:200]}")  # Preview first 200 characters

        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []


def add_to_chroma_with_batches(chunks):
    """Adds document chunks to ChromaDB using smaller batches with recovery mechanisms."""
    try:
        print("Loading embedding function...")
        embedding_function = get_embedding_function()
        
        # Create database directory if it doesn't exist
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        print("Loading embedding function...")
        print("Connecting to ChromaDB...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        try:
            # Check current document count
            existing_data = db.get(include=["metadatas"])
            existing_count = len(existing_data.get("metadatas", []))
            print(f"Existing documents in DB before adding: {existing_count}")
        except Exception as e:
            print(f"Warning when checking documents: {e}")
            existing_count = 0
        
        # Generate unique IDs for chunks
        print("Generating unique IDs for document chunks...")
        new_chunks = generate_unique_chunk_ids(chunks)
        
        if not new_chunks:
            print("No chunks to add.")
            return
            
        # Add documents in small batches with multiple retries
        batch_size = 20  # Use smaller batches
        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        print(f"Adding {len(new_chunks)} new documents to ChromaDB")
        
        added_count = 0
        failed_batches = []
        
        # Process batches
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Try multiple times for each batch
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Add the batch
                    start_time = time.time()
                    db.add_documents(batch, ids=batch_ids)
                    db.persist()
                    end_time = time.time()
                    
                    # Success!
                    added_count += len(batch)
                    duration = end_time - start_time
                    print(f"Batch {batch_num}/{total_batches} added (Attempt {attempt+1}/{max_retries}) in {duration:.2f} seconds")
                    print(f"Progress: {added_count}/{len(new_chunks)} documents ({(added_count/len(new_chunks)*100):.1f}%)")
                    
                    # Verify the addition by checking a count
                    try:
                        verify_data = db.get(include=["metadatas"])
                        verify_count = len(verify_data.get("metadatas", []))
                        print(f"Current document count in DB: {verify_count}")
                        
                        # Success! Break out of retry loop
                        break
                    except Exception as ve:
                        print(f"Warning: Could not verify counts after batch {batch_num}: {ve}")
                        # Continue anyway since documents might have been added
                        break
                    
                except Exception as e:
                    print(f"Error in batch {batch_num}, attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        # Last attempt failed, record this batch as failed
                        failed_batches.append((batch_num, batch, batch_ids))
                        print(f"Batch {batch_num} failed after {max_retries} attempts. Continuing with next batch.")
                    else:
                        # Sleep briefly before retrying
                        print(f"Retrying batch {batch_num} in 2 seconds...")
                        time.sleep(2)
        
        # Try failed batches one more time with a fresh connection
        if failed_batches:
            print(f"Attempting to process {len(failed_batches)} failed batches with a fresh connection...")
            # Close existing connection
            del db
            
            # Create fresh connection
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
            for batch_num, batch, batch_ids in failed_batches:
                try:
                    print(f"Retrying failed batch {batch_num}...")
                    db.add_documents(batch, ids=batch_ids)
                    db.persist()
                    added_count += len(batch)
                    print(f"Previously failed batch {batch_num} added successfully")
                except Exception as e:
                    print(f"Failed batch {batch_num} could not be added: {e}")
        
        # Final verification
        try:
            # Create a fresh connection for final verification
            del db
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
            final_data = db.get(include=["metadatas"])
            final_count = len(final_data.get("metadatas", []))
            print(f"Final document count in DB: {final_count}")
            print(f"Process completed: {final_count} total documents in database")
            print(f"Added approximately {final_count - existing_count} new documents")
            
            if final_count < len(new_chunks) + existing_count:
                print(f"Note: {(len(new_chunks) + existing_count) - final_count} documents may not have been added")
                
        except Exception as e:
            print(f"Error during final verification: {e}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


def generate_unique_chunk_ids(chunks):
    """Generates truly unique IDs for document chunks based on content hash and metadata."""
    updated_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Get source and page from metadata
        source = chunk.metadata.get("source", "unknown")
        page = str(chunk.metadata.get("page", "0"))
        
        # Create a hash from content to ensure uniqueness even with similar chunks
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        
        # Create a unique ID combining source, page, chunk index and content hash
        chunk_id = f"{source}:{page}:{i}:{content_hash}"
        
        # Create new document with updated metadata
        updated_chunks.append(Document(
            page_content=chunk.page_content,
            metadata={**chunk.metadata, "id": chunk_id}
        ))

    return updated_chunks


def clear_database():
    """Deletes the ChromaDB database."""
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error clearing database: {e}")


# Initialize and check database status
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error occurred: {e}")
        traceback.print_exc()
        print("Process terminated unexpectedly.")