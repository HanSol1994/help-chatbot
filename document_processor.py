import os
import tempfile
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
import docx2txt
import fitz  # PyMuPDF for better PDF handling
from vector_store import VectorStore
from image_processor import ImageProcessor

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        self.vector_store = VectorStore()
        self.image_processor = ImageProcessor()
    
    def process_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file and extract text content and images"""
        text_content = ""
        all_documents = []
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                # Extract text using PyMuPDF for better handling
                text_content = self._extract_pdf_text_pymupdf(tmp_file_path)
                
                # Extract and process images
                image_documents = self._process_pdf_images(tmp_file_path, uploaded_file.name)
                all_documents.extend(image_documents)
                
            elif file_extension == 'txt' or file_extension == 'md':
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file_extension == 'docx':
                text_content = docx2txt.process(tmp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create text document chunks
            if text_content.strip():
                text_documents = self._create_documents(text_content, uploaded_file.name)
                all_documents.extend(text_documents)
            
            # Store in vector database
            if all_documents:
                self.vector_store.add_documents(all_documents)
            
            return all_documents
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_pdf_text_pymupdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF for better handling"""
        text = ""
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
            pdf_document.close()
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {e}")
            # Fallback to PyPDF2
            text = self._extract_pdf_text(file_path)
        return text
    
    def _process_pdf_images(self, file_path: str, source_name: str) -> List[Document]:
        """Process images from PDF and create documents"""
        image_documents = []
        
        try:
            images_data = self.image_processor.extract_images_from_pdf(file_path)
            
            for img_data in images_data:
                # Create searchable content combining OCR text and description
                content = f"""
Image from {img_data['source']} in {source_name}:

Description: {img_data['description']}

Extracted Text: {img_data['ocr_text']}

Image Type: {img_data['type']}
Dimensions: {img_data['width']}x{img_data['height']}
                """.strip()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "type": "image",
                        "image_source": img_data['source'],
                        "description": img_data['description'],
                        "ocr_text": img_data['ocr_text'],
                        "image_data": img_data['image_data'],  # Base64 encoded image
                        "width": img_data['width'],
                        "height": img_data['height']
                    }
                )
                image_documents.append(doc)
                
        except Exception as e:
            print(f"Error processing PDF images: {e}")
        
        return image_documents
    
    def _create_documents(self, text: str, source_name: str) -> List[Document]:
        """Split text into chunks and create Document objects"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "type": "text",
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents