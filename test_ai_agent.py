# test_ai_agent.py
import os
from dotenv import load_dotenv
from data_loader import load_and_profile_csv
from vector_store import DataAnalysisVectorStore
from ai_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

def test_complete_pipeline():
    """Test the complete pipeline from CSV upload to AI response."""
    api_key = os.getenv("OPENAI_API_KEY")
    test_file_path = "test.csv"  # Your test file

    try:
        print("🚀 Testing Complete AI Pipeline...\n")
        
        # 1. Load and profile CSV (Step 1)
        print("1. Loading CSV...")
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
        print("   ✅ CSV loaded\n")

        # 2. Create vector store (Step 2)
        print("2. Creating vector store...")
        vector_store = DataAnalysisVectorStore(api_key)
        vector_store.create_and_persist_vectorstore(profile, metadata)
        retriever = vector_store.get_retriever()
        print("   ✅ Vector store ready\n")

        # 3. Create AI agent (Step 3)
        print("3. Initializing AI agent...")
        agent = DataAnalysisAgent(df)
        print("   ✅ AI agent ready\n")

        # 4. Test questions
        test_questions = [
            "What are the column names?",
            "What is the average price?",
            "What is the highest number of units sold?",
        ]

        for question in test_questions:
            print(f"❓ Question: {question}")
            
            # Retrieve relevant context from vector DB
            relevant_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Get answer from AI agent
            answer = agent.generate_answer(question, context)
            print(f"🤖 Answer: {answer}")
            print("-" * 50 + "\n")

        print("🎉 Pipeline test completed successfully!")

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    test_complete_pipeline()