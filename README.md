# 🤖 AI Documentation Assistant

An intelligent chatbot that processes documentation files (including images) and provides contextual answers using advanced AI. Upload PDFs, Word docs, or text files and get instant answers from your documents!

## ✨ Key Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, Markdown
- 🖼️ **Image Processing**: Extract text from images in PDFs using OCR
- 🧠 **AI-Powered**: Local language models + vector search
- 🔍 **Smart Search**: Find relevant content across all your documents
- 💬 **Interactive Chat**: Ask questions in natural language
- 🌐 **Web Interface**: Easy-to-use Streamlit interface
- 🏠 **Privacy-First**: Everything runs locally on your machine

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```
*Or directly:* `streamlit run main.py`

### 3. Upload Documents
- Click "Upload Documentation" in the sidebar
- Select PDF, DOCX, TXT, or Markdown files
- Click "Process Documents"

### 4. Start Asking Questions!
- Type your question in the chat box
- Get instant answers based on your documents
- Includes source citations and image content

## 📖 How It Works

1. **Document Processing**: Extracts text and images from your files
2. **OCR & AI Analysis**: Reads text from images and generates descriptions
3. **Vector Storage**: Creates searchable embeddings of all content
4. **Smart Retrieval**: Finds most relevant content for your questions
5. **AI Response**: Generates natural language answers using local LLM

## Configuration

Edit `.env` file to customize:
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `LLM_MODEL_PATH`: Path to your local language model
- `CHROMA_DB_PATH`: Vector database storage path
- `DOCS_PATH`: Default documents directory

## Model Downloads

For better responses, download a local language model:
- [Llama-2-7B-Chat GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Mistral-7B-Instruct GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

## Architecture

- **Document Processor**: Extracts and chunks text from various file formats
- **Image Processor**: Extracts images from PDFs, performs OCR, and generates descriptions using CLIP
- **Vector Store**: Uses ChromaDB for storing document embeddings (text + image content)
- **Chatbot**: Implements RAG to generate contextual responses from both text and image content
- **UI**: Streamlit interface for file uploads and chat

## Image Processing Capabilities

- **OCR**: Extracts text from images using EasyOCR
- **Image Classification**: Uses CLIP model to understand image content
- **Structured Data**: Detects tables, charts, and diagrams
- **Multi-format**: Supports images embedded in PDFs

## License

MIT License