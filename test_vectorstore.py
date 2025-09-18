# test_vectorstore.py
import os
from dotenv import load_dotenv
from data_loader import load_and_profile_csv
from vector_store import DataAnalysisVectorStore

# Load environment variables (your API key)
load_dotenv()

# Simple test without running the full Streamlit app
if __name__ == "__main__":
    # Get API Key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file. Please check Step 0.")
    
    test_file_path = "test.csv"  # Use the same test file from Step 1

    try:
        # 1. Simulate file upload and run Step 1 (Data Loading & Profiling)
        print("📂 Step 1: Loading and profiling CSV file...")
        with open(test_file_path, "rb") as f:
            class MockUploadedFile:
                def __init__(self, file_path):
                    self.name = file_path
                    with open(file_path, 'rb') as file:
                        self.value = file.read()
                def getvalue(self):
                    return self.value
                    
            uploaded_file = MockUploadedFile(test_file_path)
            df, profile, metadata = load_and_profile_csv(uploaded_file)
        print("   ✅ Profiling complete.")

        # 2. Initialize the Vector Store class
        print("\n🔗 Step 2: Initializing Vector Store...")
        vector_store_manager = DataAnalysisVectorStore(openai_api_key=api_key)
        
        # Optional: Clear any previous vector store for a clean test
        vector_store_manager.clear_vectorstore()
        
        # 3. Create and persist the vector store from our profile
        vector_store_manager.create_and_persist_vectorstore(profile, metadata)
        
        # 4. Test the retriever
        print("\n❓ Testing retriever with a sample query...")
        retriever = vector_store_manager.get_retriever()
        sample_query = "What are the column names?"
        relevant_docs = retriever.invoke(sample_query)
        
        print(f"   Query: '{sample_query}'")
        print(f"   Found {len(relevant_docs)} relevant chunk(s):")
        for i, doc in enumerate(relevant_docs):
            print(f"   Chunk {i+1} (Relevance Score ~): {doc.page_content[:100]}...")
        
        print("\n🎉 Step 2 completed successfully! Vector database is ready.")

    except FileNotFoundError:
        print(f"❌ Error: File '{test_file_path}' not found. Please run the test from Step 1 first.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")