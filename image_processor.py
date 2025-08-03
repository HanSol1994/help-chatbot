import os
import io
import base64
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

class ImageProcessor:
    def __init__(self):
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize CLIP model for image understanding
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_available = True
        except Exception as e:
            print(f"Warning: CLIP model not available: {e}")
            self.clip_available = False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF and process them"""
        images_data = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Process the image
                        processed_data = self.process_image(
                            pil_image, 
                            source=f"Page {page_num + 1}, Image {img_index + 1}"
                        )
                        
                        if processed_data:
                            images_data.append(processed_data)
                    
                    pix = None
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
        
        return images_data
    
    def process_image(self, image: Image.Image, source: str = "Unknown") -> Optional[Dict[str, Any]]:
        """Process a single image to extract text and generate description"""
        try:
            # Convert PIL image to numpy array for OCR
            img_array = np.array(image)
            
            # Extract text using OCR
            ocr_results = self.ocr_reader.readtext(img_array)
            extracted_text = " ".join([result[1] for result in ocr_results if result[2] > 0.5])
            
            # Generate image description using CLIP (if available)
            description = self.generate_image_description(image) if self.clip_available else "Image content"
            
            # Encode image as base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "source": source,
                "type": "image",
                "ocr_text": extracted_text,
                "description": description,
                "image_data": img_base64,
                "width": image.width,
                "height": image.height
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def generate_image_description(self, image: Image.Image) -> str:
        """Generate description of image content using CLIP"""
        if not self.clip_available:
            return "Image description not available"
        
        try:
            # Predefined categories for image classification
            categories = [
                "a diagram", "a chart", "a graph", "a table", "a screenshot",
                "a flowchart", "a technical drawing", "a photo", "a schematic",
                "code snippet", "text document", "a map", "an interface",
                "a workflow", "an architecture diagram"
            ]
            
            # Process image and text
            inputs = self.clip_processor(
                text=categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top prediction
            top_idx = probs.argmax().item()
            confidence = probs[0][top_idx].item()
            
            if confidence > 0.3:
                return f"This appears to be {categories[top_idx]} (confidence: {confidence:.2f})"
            else:
                return "Image with technical or document content"
                
        except Exception as e:
            print(f"Error generating image description: {e}")
            return "Image content"
    
    def extract_structured_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract structured data from images (tables, charts, etc.)"""
        try:
            img_array = np.array(image)
            
            # Use OCR to detect text regions
            ocr_results = self.ocr_reader.readtext(img_array, detail=1)
            
            structured_data = {
                "text_regions": [],
                "potential_table": False,
                "potential_chart": False
            }
            
            # Analyze text regions for patterns
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:
                    structured_data["text_regions"].append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence
                    })
            
            # Simple heuristics for detecting tables and charts
            if len(structured_data["text_regions"]) > 5:
                # Check for table-like patterns (aligned text)
                y_positions = [region["bbox"][0][1] for region in structured_data["text_regions"]]
                if len(set([round(y/10)*10 for y in y_positions])) < len(y_positions) * 0.7:
                    structured_data["potential_table"] = True
            
            return structured_data
            
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            return {"text_regions": [], "potential_table": False, "potential_chart": False}