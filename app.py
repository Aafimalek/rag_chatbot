import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import time
import random

from dotenv import load_dotenv

# Set page config for wide mode and better appearance
st.set_page_config(
    page_title="PDF Chat with Llama3.3",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

## load the GROQ And Google API KEY 
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

# Main title with larger font
st.markdown("# 📄 PDF Chat with Llama3.3")
st.markdown("### Upload your PDF documents and ask questions about their content using advanced AI!")
st.markdown("---")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def create_embeddings_with_retry(documents, embeddings_model, max_retries=3, batch_size=50):
    """Create embeddings with retry logic and rate limiting"""
    
    # Process documents in smaller batches to avoid hitting rate limits
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Display progress
                progress_text = f"Processing batch {batch_idx + 1}/{len(batches)} (documents {batch_idx * batch_size + 1}-{min((batch_idx + 1) * batch_size, len(documents))})"
                st.write(f"🔄 {progress_text}")
                
                # Create vector store for this batch
                if batch_idx == 0:
                    # Create initial vector store
                    vectors = FAISS.from_documents(batch, embeddings_model)
                else:
                    # Add to existing vector store
                    batch_vectors = FAISS.from_documents(batch, embeddings_model)
                    vectors.merge_from(batch_vectors)
                
                # Add delay between batches to respect rate limits
                if batch_idx < len(batches) - 1:  # Don't delay after last batch
                    delay = random.uniform(2, 4)  # Random delay between 2-4 seconds
                    st.write(f"⏳ Waiting {delay:.1f}s before next batch...")
                    time.sleep(delay)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    if retry_count < max_retries:
                        # Exponential backoff for rate limiting errors
                        wait_time = (2 ** retry_count) + random.uniform(1, 3)
                        st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time:.1f}s... (Attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries. Please wait a few minutes and try again, or consider using fewer/smaller documents.")
                else:
                    # For non-rate-limiting errors, don't retry
                    raise e
        
        # Update the global vectors variable for each successful batch
        if batch_idx == 0:
            st.session_state.vectors = vectors
        else:
            st.session_state.vectors = vectors
    
    return True

def vector_embedding(uploaded_files, embedding_choice):
    """Process uploaded PDF files and create vector embeddings"""
    
    if uploaded_files:
        try:
            # Choose embedding model based on user selection
            if "Google Gemini" in embedding_choice:
                st.write("🔗 Using Google Gemini embeddings...")
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                use_batching = True  # Use batching for API-based embeddings
            else:  # HuggingFace
                st.write("🤗 Using HuggingFace embeddings (local processing)...")
                try:
                    st.session_state.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    use_batching = False  # No need for batching with local embeddings
                except ImportError as e:
                    if "sentence_transformers" in str(e):
                        raise Exception("HuggingFace embeddings require additional packages. Please run: pip install sentence-transformers langchain-huggingface")
                    else:
                        raise e
            
            # Process all uploaded files
            all_docs = []
            
            st.write("📄 Loading PDF files...")
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Loading file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Create a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Load the PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                all_docs.extend(docs)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            st.write(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} file(s)")
            
            # Split documents into chunks
            st.write("✂️ Splitting documents into chunks...")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
            
            st.write(f"✅ Created {len(st.session_state.final_documents)} document chunks")
            
            # Create vector store with or without batching
            st.write("🧠 Creating embeddings (this may take a few minutes)...")
            
            if use_batching:
                # Use batching for API-based embeddings (Google)
                total_docs = len(st.session_state.final_documents)
                if total_docs <= 50:
                    batch_size = 25
                elif total_docs <= 200:
                    batch_size = 30
                else:
                    batch_size = 40
                
                st.info(f"📊 Processing {total_docs} chunks in batches of {batch_size} to respect API limits")
                
                success = create_embeddings_with_retry(
                    st.session_state.final_documents, 
                    st.session_state.embeddings,
                    max_retries=3,
                    batch_size=batch_size
                )
                
                if success:
                    st.write("✅ Vector embeddings created successfully!")
                    return True
                else:
                    return False
            else:
                # Process all at once for local embeddings (HuggingFace)
                st.info("📊 Processing all documents at once (local processing)")
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents, 
                    st.session_state.embeddings
                )
                st.write("✅ Vector embeddings created successfully!")
                return True
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                st.error("❌ **API Rate Limit Exceeded**")
                st.error("**Solutions:**")
                st.error("• Wait 5-10 minutes and try again")
                st.error("• Try uploading fewer or smaller PDF files")
                st.error("• Switch to HuggingFace embeddings (no API limits)")
                st.info("💡 **Tip:** The Google Embeddings API has daily quotas. If you continue to see this error, you may need to wait until your quota resets or upgrade your API plan.")
            elif "sentence_transformers" in error_msg or "HuggingFace embeddings require" in error_msg:
                st.error("❌ **Missing Dependencies for HuggingFace Embeddings**")
                st.error("**To fix this issue, run these commands:**")
                st.code("pip install sentence-transformers langchain-huggingface", language="bash")
                st.info("💡 **Alternative:** Use Google Gemini embeddings instead (requires internet and API key)")
            else:
                st.error(f"❌ **Error processing documents:** {error_msg}")
            return False
    return False

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Embedding model selection
    st.markdown("## ⚙️ Configuration")
    
    embedding_option = st.selectbox(
        "Choose Embedding Model:",
        options=[
            "Google Gemini (Free tier - may hit quotas)",
            "HuggingFace (Local - no API limits)"
        ],
        index=0,
        help="Google Gemini: More accurate but has API quotas. HuggingFace: Runs locally with no limits but requires more processing time."
    )
    
    # File upload section with larger text
    st.markdown("## 📂 Upload PDF Documents")
    st.markdown("##### Drag and drop your PDF files or click to browse")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze",
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"🎉 **{len(uploaded_files)} file(s) uploaded successfully!**")
        
        # Show uploaded file names in a nicer format
        with st.expander("📋 **View Uploaded Files**", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                file_size_mb = file.size / (1024 * 1024)
                st.markdown(f"**{i}.** `{file.name}` - *{file_size_mb:.2f} MB*")
        
        # Process documents button with better styling
        st.markdown("---")
        if st.button("🚀 **Process Documents**", type="primary", use_container_width=True):
            with st.spinner("🔄 **Processing documents... This may take a few moments.**"):
                try:
                    if vector_embedding(uploaded_files, embedding_option):
                        st.success("🎉 **Documents processed successfully! You can now ask questions.**")
                        st.balloons()
                    else:
                        st.error("❌ **Error processing documents. Please try again.**")
                except Exception as e:
                    st.error(f"❌ **Error processing documents:** {str(e)}")

with col2:
    # Status section
    st.markdown("## 📊 Status")
    if "vectors" in st.session_state:
        st.success("✅ **Documents Ready**")
        st.info(f"📑 **{len(st.session_state.final_documents)}** document chunks indexed")
        st.metric("Document Chunks", len(st.session_state.final_documents))
    else:
        st.info("⏳ **Waiting for documents...**")
        st.markdown("Upload and process PDFs to get started")

# Question input section with larger font
st.markdown("---")
st.markdown("## 💬 Ask Questions About Your Documents")
st.markdown("##### Type your question below and get AI-powered answers based on your uploaded content")

prompt1 = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the main topic discussed in the document?",
    label_visibility="collapsed"
)

# Question answering section with enhanced UI
if prompt1:
    if "vectors" in st.session_state:
        try:
            with st.spinner("🧠 **Generating AI response...**"):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                response_time = time.process_time() - start
                
                # Create columns for answer section
                st.markdown("---")
                answer_col1, answer_col2 = st.columns([3, 1])
                
                with answer_col1:
                    # Display the answer with enhanced styling
                    st.markdown("## 🤖 AI Response")
                    
                    # Answer container with background
                    with st.container():
                        st.markdown("### Answer:")
                        st.markdown(f"**{response['answer']}**")
                
                with answer_col2:
                    # Response metrics
                    st.markdown("## 📈 Metrics")
                    st.metric("Response Time", f"{response_time:.2f}s")
                    st.metric("Sources Used", len(response["context"]))
                
                # Show relevant document chunks with better formatting
                st.markdown("---")
                with st.expander("📄 **View Source Documents**", expanded=False):
                    st.markdown("#### Relevant sections from your documents:")
                    
                    for i, doc in enumerate(response["context"]):
                        with st.container():
                            st.markdown(f"##### 📑 Source {i+1}")
                            st.markdown(f"```\n{doc.page_content}\n```")
                            if i < len(response["context"]) - 1:
                                st.markdown("---")
                        
        except Exception as e:
            st.error(f"❌ **Error generating response:** {str(e)}")
    else:
        # Enhanced warning message
        st.markdown("---")
        st.warning("⚠️ **Please upload and process documents first before asking questions!**")
        st.markdown("##### 👆 Use the upload section above to get started")

# Enhanced Sidebar with instructions
with st.sidebar:
    st.markdown("# 📖 How to Use")
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    #### **Step 1:** Upload PDFs 📤
    Drag and drop or browse for PDF files
    
    #### **Step 2:** Process Documents 🔄
    Click the process button to analyze your files
    
    #### **Step 3:** Ask Questions 💭
    Type your questions about the content
    
    #### **Step 4:** Get AI Answers 🤖
    Receive intelligent responses based on your documents
    """)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    if "vectors" in st.session_state:
        st.success("✅ **System Ready!**")
        st.metric("Documents Processed", "Ready")
        st.metric("Total Chunks", len(st.session_state.final_documents))
        st.info("🎯 **You can now ask questions!**")
    else:
        st.warning("⏳ **Waiting for Documents**")
        st.info("📋 Upload PDF files to begin")
    
    st.markdown("---")
    st.markdown("### 🔧 Features")
    st.markdown("""
    - **Multiple PDF Support** 📚
    - **Multiple Embedding Models** 🧠
    - **Rate Limit Handling** ⚡
    - **Source Tracking** 🔍
    - **Real-time Responses** 💨
    """)
    
    st.markdown("---")
    st.markdown("### 🧠 Embedding Models")
    st.markdown("""
    **Google Gemini:**
    - ✅ High accuracy
    - ⚠️ Has API quotas
    - 🌐 Requires internet
    
    **HuggingFace:**
    - ✅ No API limits
    - ✅ Runs locally
    - ⚡ Good performance
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use HuggingFace if Google API hits quotas
    - Upload multiple PDFs for broader knowledge
    - Ask specific questions for better answers
    - Check source documents for verification
    """)
    
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("🦙 **Llama 3.3** | 🔍 **Google Embeddings** | ⚡ **GROQ**")




