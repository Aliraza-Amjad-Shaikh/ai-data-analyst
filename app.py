# app.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Import our custom modules
from data_loader import load_and_profile_csv
from vector_store import DataAnalysisVectorStore
from ai_agent import DataAnalysisAgent

# Load environment variables
load_dotenv()

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# --- CUSTOM CSS FOR A CLEANER LOOK ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0px;
    }
    .code-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #6c757d;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- APP TITLE ---
st.markdown('<h1 class="main-header">🤖 AI Data Analyst</h1>', unsafe_allow_html=True)
st.markdown("### Upload your CSV. Get instant insights. No code required.")
st.markdown("---")

# --- INITIALIZE SESSION STATE ---
# This keeps our data persistent across interactions in the Streamlit app
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: FILE UPLOAD & PROCESSING ---
with st.sidebar:
    st.header("📁 Step 1: Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset for analysis"
    )
    
    process_button = st.button(
        "🚀 Process File",
        type="primary",
        use_container_width=True
    )
    
    if process_button and uploaded_file is not None:
        with st.spinner("Analyzing your data structure..."):
            try:
                # Step 1: Load and profile the CSV
                df, profile, metadata = load_and_profile_csv(uploaded_file)
                st.session_state.df = df
                
                # Step 2: Create vector store
                api_key = os.getenv("OPENAI_API_KEY")
                vector_store = DataAnalysisVectorStore(api_key)
                vector_store.clear_vectorstore()  # Clear old data
                vector_store.create_and_persist_vectorstore(profile, metadata)
                st.session_state.vector_store = vector_store
                
                # Step 3: Create AI agent
                st.session_state.agent = DataAnalysisAgent(df)
                
                st.session_state.messages = []
                st.success("✅ Data processed and AI agent is ready!")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    st.markdown("---")
    st.header("💡 Example Questions")
    st.markdown("""
    - *What are the column names?*
    - *What is the average [column_name]?*
    - *Show me the distribution of [column_name]*
    - *What is the correlation between [col1] and [col2]?*
    - *Find the top 5 highest values in [column_name]*
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using LangChain, OpenAI, and Streamlit by ~ Aliraza Amjad Shaikh")

# --- MAIN CHAT INTERFACE ---
# Only show if a file has been processed
if st.session_state.df is not None:
    # Display data overview
    with st.expander("📊 Data Overview", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Total Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("File Name", uploaded_file.name)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant context
                    retriever = st.session_state.vector_store.get_retriever()
                    relevant_docs = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Generate and execute answer
                    answer = st.session_state.agent.generate_answer(prompt, context)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    # Welcome screen before file upload
    st.markdown("""
    <div style='text-align: center; padding: 50px 0px;'>
        <h2>Welcome to your AI Data Analyst! 👋</h2>
        <p style='font-size: 1.2em;'>Get instant insights from your data using natural language.</p>
        <br>
        <h3>How to use:</h3>
        <ol style='text-align: left; display: inline-block;'>
            <li>Upload a CSV file using the sidebar</li>
            <li>Click "Process File" to analyze your data</li>
            <li>Start asking questions in natural language</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample data
    sample_data = {
        'Product': ['Widget A', 'Widget B', 'Widget C'],
        'Price': [19.99, 24.99, 15.50],
        'Units_Sold': [100, 85, 120],
        'Customer_Rating': [4.5, 4.2, 4.8]
    }
    sample_df = pd.DataFrame(sample_data)
    
    with st.expander("👉 Click here to see what kind of data you can analyze"):
        st.dataframe(sample_df, use_container_width=True)
        st.caption("Example of a compatible CSV structure")

# --- FOOTER ---
st.markdown("---")
st.caption("💡 Tip: Process your file first, then ask questions about the data columns and relationships.")