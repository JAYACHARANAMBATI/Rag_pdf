import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant

# Load environment variables
load_dotenv()

# Setup Streamlit page configuration
st.set_page_config(
    page_title="Enhanced PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and create a retriever"""
    with st.spinner("Processing PDF..."):
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and split the PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split the documents into chunks optimized for Gemini
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
            
            # Use Qdrant in-memory for vector storage
            vectorstore = Qdrant.from_documents(
                docs,
                embeddings,
                location=":memory:",
                collection_name="pdf_documents"
            )
            
            # Create retriever with MMR for better diversity
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10
                }
            )
            
            # Remove temporary file
            os.unlink(tmp_path)
            
            return retriever, docs
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None, None

def determine_response_style(question):
    """Determine the appropriate response style based on the question type"""
    question = question.lower()
    
    if any(word in question for word in ["summary", "overview", "summarize"]):
        return "summary"
    elif any(word in question for word in ["list", "steps", "how to", "process"]):
        return "list"
    elif any(word in question for word in ["compare", "difference", "similar"]):
        return "comparison"
    elif any(word in question for word in ["why", "reason", "explain"]):
        return "explanation"
    elif any(word in question for word in ["what is", "define", "meaning"]):
        return "definition"
    elif any(word in question for word in ["table", "chart", "data"]):
        return "tabular"
    else:
        return "general"

def format_response(response_text, style):
    """Format the response based on the determined style"""
    if style == "summary":
        return f"📌 **Summary:**\n\n{response_text}\n\n[Let me know if you'd like more details on any part]"
    elif style == "list":
        # Convert numbered or bullet points to markdown list
        lines = response_text.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '- ', '* ')):
                formatted_lines.append(line)
            else:
                formatted_lines.append(f"- {line}")
        return "📋 **Step-by-Step:**\n\n" + "\n".join(formatted_lines)
    elif style == "comparison":
        return "⚖️ **Comparison:**\n\n" + response_text
    elif style == "explanation":
        return "🔍 **Explanation:**\n\n" + response_text
    elif style == "definition":
        return "📖 **Definition:**\n\n" + response_text
    elif style == "tabular":
        # Try to format as a table if the response has clear columns
        if '|' in response_text or all(':' in line for line in response_text.split('\n')[:3]):
            return "📊 **Data Presentation:**\n\n" + response_text
        return "📊 **Information:**\n\n" + response_text
    else:
        return "📄 **Response:**\n\n" + response_text

def get_llm_response(user_question, retriever):
    """Get response from LLM using retrieval chain with Gemini"""
    try:
        # Determine response style based on question
        response_style = determine_response_style(user_question)
        
        # Create LLM with Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            max_output_tokens=2000,
            top_k=40,
            top_p=0.95
        )
        
        # Enhanced prompt template with style guidance
        system_prompt = f"""You are an expert AI assistant analyzing PDF documents. Use the following context to answer questions:

        Context: {{context}}

        Current Question: {user_question}
        Detected Response Style: {response_style}

        Guidelines:
        1. Provide accurate, well-structured responses
        2. For summaries, focus on key points
        3. For lists, use clear numbering or bullets
        4. For comparisons, use parallel structure
        5. For explanations, provide logical flow
        6. For definitions, be concise but comprehensive
        7. For tabular data, use markdown tables if appropriate
        8. If unsure, say "I couldn't find that in the document"
        9. Maintain professional tone
        10. Highlight key points when appropriate
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Get response
        response = rag_chain.invoke({"input": user_question})
        raw_answer = response["answer"]
        
        # Format the response based on style
        formatted_answer = format_response(raw_answer, response_style)
        
        return formatted_answer
    
    except Exception as e:
        return f"⚠️ **Error:** {str(e)}. Please try again or rephrase your question."

# App title
st.title("📚 Enhanced PDF RAG Assistant")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and not st.session_state.document_processed:
        st.session_state.retriever, docs = process_pdf(uploaded_file)
        if st.session_state.retriever:
            st.session_state.document_processed = True
            
            # Document summary
            st.success("Document processed successfully!")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Pages:** {len(docs)}")
            
            # Add welcome message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"📚 I've analyzed '{uploaded_file.name}'."
            })
    
    if st.session_state.document_processed:
        if st.button("🔄 Start New Session"):
            st.session_state.document_processed = False
            st.session_state.messages = []
            st.session_state.retriever = None
            st.rerun()

# Main chat interface
if st.session_state.document_processed:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user question
    if user_question := st.chat_input("Ask about the PDF..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                assistant_response = get_llm_response(user_question, st.session_state.retriever)
                st.markdown(assistant_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
else:
    st.info("📤 Please upload a PDF document to begin analysis")

# Footer
st.sidebar.markdown("---")
