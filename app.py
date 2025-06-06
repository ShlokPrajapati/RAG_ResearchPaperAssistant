import os
import re
import time
import hashlib
import numpy as np
import streamlit as st
import faiss
import PyPDF2
from dotenv import load_dotenv
from langchain_community.document_loaders import ArxivLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Debug helper function
def debug_print(message, data=None):
    """Print debug information with timestamps"""
    timestamp = time.strftime("%H:%M:%S")
    if data:
        st.sidebar.write(f"[{timestamp}] {message}")
        st.sidebar.write(f"Data: {str(data)[:100]}...")  # Show only first 100 chars
    else:
        st.sidebar.write(f"[{timestamp}] {message}")

# Initialize models with caching
@st.cache_resource
def load_models():
    debug_print("Loading models...")
    start_time = time.time()
    models = {
        'embedder': SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),
        'groq': ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    }
    debug_print(f"Models loaded in {time.time() - start_time:.2f} seconds")
    return models

models = load_models()

# Initialize session state
def init_session():
    vars = {
        'pdf_index': None,
        'pdf_metadata': [],
        'processed_pdf_hashes': set(),  # Track processed PDFs by hash
        'arxiv_index': None,
        'arxiv_metadata': [],
        'pdf_chat_history': [],
        'arxiv_chat_history': [],
        'processed_arxiv_queries': set(),  # Track processed queries to avoid duplicates
        'clear_arxiv': False,  # Flag to clear arXiv data when needed
        'debug_mode': False  # Toggle for debug info
    }
    for k, v in vars.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ----------- PDF Processing -----------

def get_file_hash(file_content, filename):
    """Generate a unique hash for the file to avoid duplicate processing"""
    file_identifier = f"{filename}_{len(file_content)}"
    return hashlib.md5(file_identifier.encode()).hexdigest()

@st.cache_data
def extract_pdf_text(file_content, filename):
    """Cache PDF text extraction to improve performance"""
    debug_print(f"Extracting text from PDF: {filename}")
    start_time = time.time()
    
    import io
    pdf_file = io.BytesIO(file_content)
    
    text = ""
    try:
        pdf = PyPDF2.PdfReader(pdf_file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        debug_print(f"Error reading PDF: {str(e)}")
        return ""

    text = re.sub(r'\s+', ' ', text)
    debug_print(f"PDF extraction completed in {time.time() - start_time:.2f} seconds")
    return text

def process_pdf(file):
    """Process PDF only if it hasn't been processed before"""
    # Create a copy of the file content for caching
    file_content = file.getvalue()
    file_hash = get_file_hash(file_content, file.name)
    
    # Check if this PDF has already been processed
    if file_hash in st.session_state.processed_pdf_hashes:
        debug_print(f"PDF {file.name} already processed, skipping...")
        return False  # Indicate no new processing was done
    
    debug_print(f"Processing new PDF: {file.name}")
    
    # Extract text
    text = extract_pdf_text(file_content, file.name)
    if not text.strip():
        st.error(f"Could not extract text from {file.name}")
        return False

    debug_print(f"Processing PDF chunks for {file.name}")
    start_time = time.time()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    debug_print(f"Created {len(chunks)} chunks")

    if not chunks:
        st.error(f"No content chunks created from {file.name}")
        return False

    # Create embeddings
    embed_start = time.time()
    debug_print(f"Generating embeddings for {len(chunks)} chunks")
    try:
        embeddings = models['embedder'].encode(chunks)
        debug_print(f"Embeddings generated in {time.time() - embed_start:.2f} seconds")
    except Exception as e:
        debug_print(f"Error generating embeddings: {str(e)}")
        st.error(f"Error processing {file.name}: {str(e)}")
        return False

    # Initialize index if needed
    if st.session_state.pdf_index is None:
        st.session_state.pdf_index = faiss.IndexFlatL2(embeddings.shape[1])
        debug_print("Created new FAISS index for PDFs")

    # Add to index
    index_start = time.time()
    st.session_state.pdf_index.add(embeddings.astype('float32'))
    debug_print(f"Added to index in {time.time() - index_start:.2f} seconds")

    # Store metadata with unique IDs to avoid duplication
    current_index = len(st.session_state.pdf_metadata)
    for i, chunk in enumerate(chunks):
        st.session_state.pdf_metadata.append({
            'id': current_index + i,  # Unique ID
            'source': file.name,
            'file_hash': file_hash,  # Track which file this chunk belongs to
            'text': chunk,
            'page': i // 3 + 1
        })

    # Mark this PDF as processed
    st.session_state.processed_pdf_hashes.add(file_hash)
    
    debug_print(f"Total PDF processing time: {time.time() - start_time:.2f} seconds")
    return True  # Indicate successful new processing

def summarize_pdf(file):
    debug_print(f"Summarizing PDF: {file.name}")
    start_time = time.time()
    
    # Use cached text extraction
    file_bytes = file.getvalue()
    text = extract_pdf_text(file_bytes, file.name)
    
    if not text.strip():
        return "Could not extract text from the PDF for summarization."
    
    # Use only first part of document for summary
    text_for_summary = text[:3000]
    
    debug_print("Requesting summary from Groq")
    try:
        response = models['groq'].invoke([
            {"role": "system", "content": "You are a research paper summarization expert."},
            {"role": "user", "content": f"Summarize the following research paper based on the retrieved context. Include:\n- Paper Title\n- Author(s)\n- Date or Publication Info (if available)\n- Key Evaluation Metrics Used and Their Results\n- Summary of the Approach/Methodology\nProvide a concise summary using only the information from the retrieved documents. If any requested information is missing, state 'Not provided in the retrieved documents.':\n{text_for_summary}"}
        ])
        debug_print(f"Summary completed in {time.time() - start_time:.2f} seconds")
        return response.content
    except Exception as e:
        debug_print(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

# ----------- arXiv Processing -----------

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def fetch_arxiv_papers(query, max_results):
    """Cache arXiv results to improve performance"""
    debug_print(f"Fetching papers from arXiv: '{query}', max: {max_results}")
    start_time = time.time()
    
    try:
        loader = ArxivLoader(query=query, max_results=max_results, load_all_available_meta=True)
        docs = loader.load()
        debug_print(f"Fetched {len(docs)} papers in {time.time() - start_time:.2f} seconds")
        return docs
    except Exception as e:
        debug_print(f"Error during arXiv fetch: {str(e)}")
        return []

def process_arxiv(query, max_results):
    debug_print(f"Processing arXiv query: '{query}', max results: {max_results}")
    start_time = time.time()
    
    # Check if we've already processed this query to avoid duplicates
    if f"{query}:{max_results}" in st.session_state.processed_arxiv_queries:
        debug_print("Query already processed, skipping")
        return 0
    
    st.session_state.processed_arxiv_queries.add(f"{query}:{max_results}")
    
    try:
        # Use cached fetching
        docs = fetch_arxiv_papers(query, max_results)
        
        if not docs:
            return 0
        
        # Process and chunk the documents before storing
        debug_print(f"Splitting {len(docs)} documents into chunks")
        chunk_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Smaller chunks to manage token limits
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunked_docs = []
        for doc in docs:
            # Store original metadata
            meta = doc.metadata
            
            # Split content into smaller chunks
            chunks = text_splitter.split_text(doc.page_content)
            debug_print(f"Document '{meta.get('Title', 'Unknown')}' split into {len(chunks)} chunks")
            
            # Create new document objects for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = type(doc)(
                    page_content=chunk,
                    metadata={
                        **meta,
                        "chunk": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        debug_print(f"Chunking completed in {time.time() - chunk_start:.2f} seconds")
        
        # Use the chunked documents for the vector store
        debug_print(f"Creating embeddings for {len(chunked_docs)} chunks")
        embed_start = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        if st.session_state.arxiv_index is None:
            st.session_state.arxiv_index = FAISS.from_documents(chunked_docs, embeddings)
            debug_print("Created new FAISS index for arXiv papers")
        else:
            st.session_state.arxiv_index.add_documents(chunked_docs)
            debug_print("Added documents to existing FAISS index")
        debug_print(f"Embeddings and indexing completed in {time.time() - embed_start:.2f} seconds")
        
        # Store metadata for display - ensure we don't duplicate
        paper_titles = set([p['title'] for p in st.session_state.arxiv_metadata])
        added_count = 0
        
        for doc in docs:
            title = doc.metadata.get('Title', 'Unknown Title')
            if title not in paper_titles:
                st.session_state.arxiv_metadata.append({
                    'title': title,
                    'authors': doc.metadata.get('Authors', 'Unknown Authors'),
                    'published': doc.metadata.get('Published', 'Unknown Date'),
                    'url': doc.metadata.get('entry_id', '#'),
                    'summary': doc.metadata.get('Summary', 'No summary available')[:300] + '...'  # Limited summary
                })
                paper_titles.add(title)
                added_count += 1
            
        debug_print(f"Total arXiv processing time: {time.time() - start_time:.2f} seconds")
        return added_count
    except Exception as e:
        debug_print(f"Error fetching papers: {str(e)}")
        return 0

def arxiv_qa(query):
    debug_print(f"Processing arXiv QA: '{query}'")
    start_time = time.time()
    
    try:
        # Get relevant documents
        retriever = st.session_state.arxiv_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        debug_print(f"Retrieved {len(docs)} relevant documents in {time.time() - start_time:.2f} seconds")
        
        # Track unique papers to avoid duplicate information
        unique_papers = {}
        
        # Process and limit the content to avoid token limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Extract and prepare limited context
        context_chunks = []
        for doc in docs:
            # Use title as unique identifier
            title = doc.metadata.get('Title', 'Unknown')
            
            # Only process this paper if we haven't seen it before
            # or if we have fewer than 2 chunks from it
            if title not in unique_papers or unique_papers[title] < 2:
                # Add document metadata as context
                if title not in unique_papers:
                    meta = f"Title: {title}\n" \
                           f"Authors: {doc.metadata.get('Authors', 'Unknown')}\n" \
                           f"Published: {doc.metadata.get('Published', 'Unknown')}\n"
                    context_chunks.append(meta)
                    unique_papers[title] = 0
                
                # Add limited content from each document
                doc_chunks = text_splitter.split_text(doc.page_content)
                # Take only one chunk from this document to avoid repetition
                if doc_chunks and unique_papers[title] < 2:
                    context_chunks.append(doc_chunks[0])
                    unique_papers[title] += 1
        
        # Combine limited context
        limited_context = "\n\n---\n\n".join(context_chunks)
        
        # Ensure context isn't too large - hard limit at 4000 chars
        if len(limited_context) > 4000:
            limited_context = limited_context[:4000] + "...(truncated)"
        
        debug_print(f"Context prepared with {len(context_chunks)} chunks, {len(limited_context)} chars")
        
        # Query Groq with the limited context
        response_start = time.time()
        response = models['groq'].invoke([
            {"role": "system", "content": "You are a research assistant. Answer questions based only on the provided context from arXiv papers. If the information isn't in the context, say 'The information is not found in the retrieved papers.'"},
            {"role": "user", "content": f"Context from arXiv papers:\n\n{limited_context}\n\nQuestion: {query}\n\nProvide a concise answer using only information from the context above."}
        ])
        debug_print(f"Groq response received in {time.time() - response_start:.2f} seconds")
        
        # Return both the answer and source documents
        debug_print(f"Total QA processing time: {time.time() - start_time:.2f} seconds")
        return response.content, docs
        
    except Exception as e:
        debug_print(f"Error in arXiv QA: {str(e)}")
        return "I encountered an error while processing your question. Please try a simpler question or restart the application.", []

# ----------- PDF QA Function -----------

def pdf_qa(query):
    """Process PDF QA without generating new embeddings"""
    debug_print(f"Processing PDF QA: '{query}' - NO embedding generation")
    start_time = time.time()
    
    if st.session_state.pdf_index is None:
        return "No PDFs have been processed yet. Please upload and process some PDFs first."
    
    try:
        # Embed ONLY the query (not the PDFs again)
        query_embed = models['embedder'].encode([query])
        debug_print("Query embedded (PDFs already processed)")
        
        # Search for similar chunks in existing index
        search_start = time.time()
        _, indices = st.session_state.pdf_index.search(query_embed.astype('float32'), 3)
        debug_print(f"Search completed in {time.time() - search_start:.2f} seconds")
        
        # Track already seen text chunks to avoid duplication
        seen_chunks = set()
        context_parts = []
        
        # Build context from unique chunks
        for i in indices[0]:
            if i < len(st.session_state.pdf_metadata):  # Ensure index is valid
                chunk_text = st.session_state.pdf_metadata[i]['text']
                # Hash the chunk to check for duplicates
                chunk_hash = hash(chunk_text[:100])  # Use first 100 chars as a signature
                
                if chunk_hash not in seen_chunks:
                    seen_chunks.add(chunk_hash)
                    source_info = f"[From: {st.session_state.pdf_metadata[i]['source']}, Page: {st.session_state.pdf_metadata[i]['page']}]"
                    context_parts.append(f"{source_info}\n{chunk_text}")
        
        # Combine context parts
        context = "\n\n---\n\n".join(context_parts)
        debug_print(f"Context prepared with {len(context_parts)} unique chunks")
        
        if not context.strip():
            return "No relevant information found in the uploaded PDFs for your question."
        
        # Query Groq with context
        response_start = time.time()
        response = models['groq'].invoke([
            {"role": "system", "content": "You are a RAG-Powered Research Paper Assistant. Answer user questions using only the context from retrieved research papers. Provide clear, concise, and accurate responses based solely on the retrieved documents. If no relevant information is found in the retrieved context, respond with 'No relevant information found in the retrieved documents.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ])
        debug_print(f"Groq response received in {time.time() - response_start:.2f} seconds")
        
        debug_print(f"Total PDF QA processing time: {time.time() - start_time:.2f} seconds")
        return response.content
        
    except Exception as e:
        debug_print(f"Error in PDF QA: {str(e)}")
        return "I encountered an error while processing your question. Please try again or upload different PDFs."

# ----------- Streamlit UI -----------

def main():
    st.title("📚 RAG-Powered Research Paper Assistant")
    
    # Debug mode toggle in sidebar
    st.sidebar.title("Debug Settings")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    
    if st.session_state.debug_mode:
        st.sidebar.subheader("Debug Information")
        # Show processing status
        st.sidebar.write(f"Processed PDFs: {len(st.session_state.processed_pdf_hashes)}")
        st.sidebar.write(f"Total chunks: {len(st.session_state.pdf_metadata)}")
        if st.session_state.pdf_index:
            st.sidebar.write(f"Index size: {st.session_state.pdf_index.ntotal}")
    
    tab1, tab2, tab3 = st.tabs(["PDF Assistant", "Summarization", "arXiv Researcher"])

    # -------- Tab 1: PDF Assistant --------
    with tab1:
        st.header("PDF Chat Assistant")

        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="pdf_upload")
        
        if uploaded_files:
            progress_bar = st.progress(0)
            processed_count = 0
            skipped_count = 0
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    was_processed = process_pdf(file)
                    if was_processed:
                        processed_count += 1
                    else:
                        skipped_count += 1
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if processed_count > 0:
                st.success(f"Processed {processed_count} new PDF(s)")
            if skipped_count > 0:
                st.info(f"Skipped {skipped_count} already processed PDF(s)")
            progress_bar.empty()

        # Show current status
        if st.session_state.pdf_metadata:
            unique_files = list(set([meta['source'] for meta in st.session_state.pdf_metadata]))
            st.caption(f"📄 Currently indexed: {len(unique_files)} PDF(s)")
            with st.expander("View indexed PDFs"):
                for filename in unique_files:
                    chunks_count = len([m for m in st.session_state.pdf_metadata if m['source'] == filename])
                    st.write(f"• {filename} ({chunks_count} chunks)")
            
            # Add clear button
            if st.button("🗑️ Clear All PDFs", type="secondary"):
                st.session_state.pdf_index = None
                st.session_state.pdf_metadata = []
                st.session_state.processed_pdf_hashes = set()
                st.session_state.pdf_chat_history = []
                st.success("All PDF data cleared!")
                st.rerun()

        # Chat interface
        if st.session_state.pdf_index:
            # Display chat history first
            for msg in st.session_state.pdf_chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            user_input = st.chat_input("Ask a question about your PDFs...")
            if user_input:
                # Add user message
                st.session_state.pdf_chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Get assistant response
                with st.spinner("Finding relevant information..."):
                    answer = pdf_qa(user_input)
                
                # Add and display assistant response
                st.session_state.pdf_chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.write(answer)

    # -------- Tab 2: Summarization --------
    with tab2:
        st.header("Document Summarization")

        sum_file = st.file_uploader("Upload a PDF to Summarize", type="pdf", key="sum_upload")
        if sum_file:
            with st.spinner("Summarizing..."):
                summary = summarize_pdf(sum_file)
                st.subheader("Summary Points")
                st.write(summary)

    # -------- Tab 3: arXiv Research Assistant --------
    with tab3:
        st.header("arXiv Research Assistant")

        # Input area
        col1, col2 = st.columns([3, 1])
        with col1:
            arxiv_query = st.text_input("Search Topic (arXiv)", key="arxiv_query")
        with col2:
            max_results = st.number_input("Max Papers", min_value=1, max_value=5, value=2, key="max_papers")
            
        # Action buttons
        col3, col4 = st.columns([1, 1])
        with col3:
            fetch_button = st.button("Fetch Papers", key="fetch_button")
        with col4:
            clear_button = st.button("Clear All Papers", key="clear_button")
            
        # Handle clear button
        if clear_button:
            st.session_state.arxiv_index = None
            st.session_state.arxiv_metadata = []
            st.session_state.arxiv_chat_history = []
            st.session_state.processed_arxiv_queries = set()
            st.session_state.clear_arxiv = True
            st.success("All paper data cleared. You can start a new search.")
            st.rerun()

        # Only fetch papers when the button is clicked
        if fetch_button and arxiv_query:
            with st.spinner("Fetching from arXiv..."):
                papers_count = process_arxiv(arxiv_query, max_results)
                if papers_count > 0:
                    st.success(f"Fetched {papers_count} papers about '{arxiv_query}'")
                else:
                    st.warning("No new papers found for this query. Try a different search term.")
        
        # Display fetched papers
        if st.session_state.arxiv_metadata:
            st.subheader(f"Retrieved Papers ({len(st.session_state.arxiv_metadata)})")
            for paper in st.session_state.arxiv_metadata:
                with st.expander(f"**{paper['title']}**", expanded=False):
                    st.caption(f"Authors: {paper['authors']}")
                    st.caption(f"Published: {paper['published']}")
                    st.markdown(f"[View on arXiv]({paper['url']})")
                    st.markdown("**Summary:**")
                    st.markdown(paper.get('summary', 'No summary available'))

        # QA Interface - Add a clear visual separator
        st.divider()
        st.subheader("Ask Questions About the Papers")

        if st.session_state.arxiv_index is not None:
            # Add a message about token limits
            st.info("Note: Questions are answered using a limited context from each paper to stay within token limits.")
            
            # Display existing chat history
            for message in st.session_state.arxiv_chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # New question input
            user_query = st.chat_input("Ask about the fetched arXiv papers...")
            if user_query:
                # Add user message to history and display
                st.session_state.arxiv_chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)
                    
                # Process query
                with st.spinner("Analyzing papers..."):
                    try:
                        answer, sources = arxiv_qa(user_query)
                        
                        # Add assistant response to history and display
                        st.session_state.arxiv_chat_history.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.write(answer)
                            
                            # Show sources in an expander to save space
                            with st.expander("View Source Papers", expanded=False):
                                source_papers = {}
                                
                                # Group chunks by their original paper
                                for doc in sources:
                                    title = doc.metadata.get('Title', 'Unknown Title')
                                    if title not in source_papers:
                                        source_papers[title] = {
                                            'authors': doc.metadata.get('Authors', 'Unknown Authors'),
                                            'published': doc.metadata.get('Published', 'Unknown Date'),
                                            'url': doc.metadata.get('entry_id', '#')
                                        }
                                
                                # Display each source paper once
                                for title, details in source_papers.items():
                                    st.markdown(f"**{title}**")
                                    st.caption(f"Authors: {details['authors']}")
                                    st.caption(f"Published: {details['published']}")
                                    st.markdown(f"[View on arXiv]({details['url']})")
                                    st.divider()
                                    
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.info("Try using a shorter question or retrieving fewer papers to stay within token limits.")

if __name__ == "__main__":
    main()