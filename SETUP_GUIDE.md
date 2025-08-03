# 🚀 Quick Setup Guide

## Step-by-Step Installation

### 1. Prerequisites
- ✅ Python 3.12+ installed
- ✅ Git (optional)
- ✅ 8GB+ RAM recommended
- ✅ 2GB+ free disk space

### 2. Install Dependencies
Open command prompt and run:
```bash
cd C:\Users\haans\help-chatbot
pip install -r requirements.txt
```

### 3. Create Directories
The app will create these automatically, but you can create them manually:
```bash
mkdir models
mkdir documents
mkdir vector_db
```

### 4. Optional: Download LLM Model
For better responses, download a local language model:

**Recommended Models:**
- **Llama-2-7B-Chat**: Good quality, 4GB RAM needed
- **Mistral-7B**: Fast and efficient
- **TinyLlama**: Lightweight, 1GB RAM

**Download Links:**
- [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Mistral-7B-Instruct-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Place the .gguf file in the `models/` directory.

### 5. Create Configuration (Optional)
Create `.env` file:
```env
# Optional: Path to your downloaded model
LLM_MODEL_PATH=models/llama-2-7b-chat.q4_0.gguf

# Default embedding model (auto-downloaded)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector database location
VECTOR_DB_PATH=./vector_db
```

## 🎯 First Time Usage

### 1. Start the Application
```bash
python run.py
```

### 2. Access the Interface
- Browser opens automatically to `http://localhost:8501`
- If not, manually open the URL

### 3. Upload Your First Document
1. **Click** "Upload Documentation" in the sidebar
2. **Select** a PDF, Word doc, or text file
3. **Click** "Process Documents"
4. **Wait** for processing to complete

### 4. Ask Questions
- Type: "What is this document about?"
- Type: "Summarize the main points"
- Type: "How does [specific topic] work?"

## 🔧 Troubleshooting

### Common Installation Issues

#### Problem: `llama-cpp-python` won't install
**Solution:**
1. Install Visual Studio Build Tools
2. Use Developer Command Prompt
3. Or skip LLM: Comment out llama-cpp-python in requirements.txt

#### Problem: Out of memory errors
**Solution:**
1. Use smaller model (7B instead of 13B+)
2. Close other applications
3. Reduce document chunk size

#### Problem: OCR not working
**Solution:**
1. Install Visual C++ Redistributables
2. Check image quality (should be high resolution)
3. Restart application

#### Problem: Streamlit won't start
**Solution:**
1. Check if port 8501 is available
2. Try: `streamlit run main.py --server.port 8502`
3. Check firewall settings

### Performance Tips

#### For Better Speed:
- Use SSD storage for vector database
- Close unnecessary applications
- Use smaller embedding models

#### For Better Quality:
- Use higher quality documents (300+ DPI for PDFs)
- Upload multiple related documents
- Use descriptive filenames

## 📋 Testing Your Setup

### 1. Test Dependencies
```bash
python -c "import streamlit, langchain, chromadb; print('Core packages OK')"
python -c "import cv2, easyocr, torch; print('AI packages OK')"
```

### 2. Test Document Processing
1. Upload a simple PDF or text file
2. Check for processing completion message
3. Verify no error messages appear

### 3. Test Chat Functionality
1. Ask: "What documents do you have?"
2. Should list your uploaded documents
3. Ask a specific question about your content

## 🎨 Interface Overview

### Sidebar (Left Panel)
- **Upload Documentation**: Add new files
- **Process Documents**: Convert files to searchable format
- **Status Messages**: Processing feedback

### Main Area (Right Panel)
- **Chat History**: Previous questions and answers
- **Input Box**: Type your questions here
- **Send Button**: Submit questions (or press Enter)

### Response Format
- **Answer**: AI-generated response
- **Sources**: Which documents were used
- **Images**: Descriptions of relevant images/charts

## 🚨 Important Notes

### Privacy & Security
- ✅ All processing happens on your computer
- ✅ No data sent to external servers
- ✅ Documents stay on your machine
- ✅ No internet required (after setup)

### File Recommendations
- **PDFs**: Up to 50MB, prefer text-based over scanned
- **Images**: High resolution for better OCR
- **Text Files**: UTF-8 encoding preferred
- **Word Docs**: .docx format (not .doc)

### System Requirements
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB+ RAM, SSD storage
- **Optimal**: 16GB+ RAM for large documents

## 🎯 Next Steps

### Once Setup is Complete:
1. **Upload Documentation**: Start with your most important docs
2. **Test Thoroughly**: Try different question types
3. **Optimize**: Adjust settings based on your needs
4. **Scale Up**: Add more documents as needed

### Advanced Usage:
1. **Custom Models**: Try different LLM models
2. **Batch Processing**: Upload multiple files at once
3. **Query Optimization**: Learn effective question patterns
4. **Performance Tuning**: Adjust chunk sizes and parameters

---

**Need Help?** Check [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical information, or refer to the troubleshooting section above.

**Ready to Start?** Run `python run.py` and begin uploading your documents!