import streamlit as st
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from chatbot import Chatbot
import traceback

load_dotenv()

def main():
    st.set_page_config(page_title="AI Documentation Assistant", page_icon="🤖")
    
    st.title("🤖 AI Documentation Assistant")
    st.sidebar.title("Document Management")
    
    # Initialize processors
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot()
    
    # Sidebar for document upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documentation",
        type=['pdf', 'txt', 'docx', 'md'],
        accept_multiple_files=True,
        help="Supports PDF (with images), TXT, DOCX, and Markdown files"
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.sidebar:
                with st.spinner("Processing documents (including images)..."):
                    for file in uploaded_files:
                        try:
                            documents = st.session_state.doc_processor.process_file(file)
                            
                            # Count text vs image documents
                            text_docs = len([d for d in documents if d.metadata.get('type') == 'text'])
                            image_docs = len([d for d in documents if d.metadata.get('type') == 'image'])
                            
                            st.write(f"✅ {file.name}: {text_docs} text chunks, {image_docs} images")
                            
                        except Exception as e:
                            traceback.print_exc()
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                            
                st.success("Document processing completed!")
    
    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your documentation"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()