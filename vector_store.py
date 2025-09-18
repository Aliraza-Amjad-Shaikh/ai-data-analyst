# vector_store.py
import os
import time
import shutil
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

class DataAnalysisVectorStore:
    """
    A class to handle the creation and management of a vector store for the data profile.
    """
    def __init__(self, openai_api_key: str):
        """
        Initialize the vector store with OpenAI embeddings and a local ChromaDB persistence directory.
        """
        # Initialize the embedding function
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Define the path for the database. It will be created inside a 'chroma_db' folder.
        self.persist_directory = "chroma_db"
        
        # This will be our LangChain Chroma client object
        self.vectorstore = None

    def _chunk_profile_text(self, profile_text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Splits the large profile text into smaller, meaningful chunks for better retrieval.
        Uses LangChain's RecursiveTextSplitter for smart splitting.
        """
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Number of characters per chunk
            chunk_overlap=200,  # Overlap between chunks to maintain context
            length_function=len,
        )
        
        # Create a LangChain Document from the profile text
        doc = Document(page_content=profile_text, metadata=metadata)
        
        # Split the document into chunks
        chunks = text_splitter.split_documents([doc])
        
        return chunks

    def create_and_persist_vectorstore(self, profile_text: str, metadata: Dict[str, Any]) -> None:
        """
        Main function to create the vector store from the profile text and save it to disk.
        """
        print("🧠 Creating vector store from data profile...")
        
        # 1. Split the large profile text into chunks
        chunks = self._chunk_profile_text(profile_text, metadata)
        print(f"   Created {len(chunks)} text chunks for vectorization.")
        
        # 2. FILTER METADATA: Remove any complex metadata types that ChromaDB can't handle.
        from langchain_community.vectorstores.utils import filter_complex_metadata
        
        # This function will remove any metadata values that are not simple types (str, int, float, bool)
        filtered_chunks = filter_complex_metadata(chunks)
        print("   ✅ Filtered out complex metadata values (like lists).")
        
        # 3. Create the vector store from the FILTERED chunks.
        #    This will automatically generate embeddings for each chunk and store them persistently.
        self.vectorstore = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            ids=[str(uuid.uuid4()) for _ in range(len(filtered_chunks))]
        )
        
        print(f"   ✅ Vector store created and persisted to '{self.persist_directory}'.")

    def get_retriever(self):
        """
        Returns a retriever object from the vector store to be used for querying.
        Must be called after create_and_persist_vectorstore.
        """
        if self.vectorstore is None:
            # If we're restarting the app, we need to load the existing store from disk.
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        # Create a retriever that fetches the top 3 most relevant chunks
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def clear_vectorstore(self):
        """
        Deletes the persisted vector store directory.
        Useful for testing or when a new file is uploaded.
        Handles Windows file locking issues with retry logic.
        """
        if not os.path.exists(self.persist_directory):
            print("   ℹ️ No existing vector store found to clear.")
            return
            
        print("   ♻️ Attempting to clear vector store...")
        
        # Try multiple times to handle Windows file locking
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(self.persist_directory)
                print("   ✅ Vector store cleared successfully.")
                return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    wait_time = 1 * (attempt + 1)  # Wait 1s, 2s, 3s, etc.
                    print(f"   ⏳ File locked (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed to clear vector store after {max_retries} attempts: {e}")
                    # Try a more forceful approach on final attempt
                    try:
                        os.system(f'rmdir /s /q "{self.persist_directory}"')
                        print("   ✅ Vector store cleared using force method.")
                    except:
                        print("   ❌ Could not clear vector store. Manual cleanup may be required.")
            except Exception as e:
                print(f"   ❌ Error clearing vector store: {e}")
                break