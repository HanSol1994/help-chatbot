import os
from typing import List, Dict, Any
from vector_store import VectorStore
from llama_cpp import Llama

class Chatbot:
    def __init__(self):
        self.vector_store = VectorStore()
        self.model_path = os.getenv("LLM_MODEL_PATH", "models/llama-2-7b-chat.gguf")
        
        # Initialize local LLM (you'll need to download a model)
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=4,
                verbose=False,
                n_gpu_layers=0
            )
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            self.llm = None
    
    def get_response(self, query: str) -> str:
        """Generate response based on query and retrieved documents"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=5)
            
            if not relevant_docs:
                return "I don't have any relevant documentation to answer your question. Please upload some documents first."
            
            # Create context from retrieved documents
            context = self._create_context(relevant_docs)
            
            # Generate response
            if self.llm:
                response = self._generate_with_llm(query, context)
            else:
                response = self._generate_fallback_response(query, relevant_docs)
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _create_context(self, docs: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved documents"""
        context_parts = []
        for doc in docs:
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']
            context_parts.append(f"From {source}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate response using local LLM"""
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documentation. Follow these guidelines:

1. Answer directly and completely based on the context
2. If information is not in the context, clearly state "I don't have information about that in the provided documentation"
3. Be specific and cite relevant details from the context
4. Provide complete, well-formed sentences
5. If multiple documents are relevant, synthesize the information

Context:
{context}

Question: {query}

Provide a complete and helpful answer:"""
        
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            stop=["Question:", "Context:", "\n\nQuestion:", "\n\nContext:"],
            echo=False,
            repeat_penalty=1.1
        )
        
        return response['choices'][0]['text'].strip()
    
    def _generate_fallback_response(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """Fallback response when LLM is not available"""
        response = f"Based on the documentation, here are the most relevant sections for your query '{query}':\n\n"
        
        for i, doc in enumerate(docs, 1):
            metadata = doc['metadata']
            source = metadata.get('source', 'Unknown')
            doc_type = metadata.get('type', 'text')
            content = doc['content']
            
            if doc_type == 'image':
                # Special handling for image documents
                image_source = metadata.get('image_source', 'Unknown location')
                description = metadata.get('description', 'No description')
                ocr_text = metadata.get('ocr_text', 'No text extracted')
                
                response += f"{i}. **Image from {source}** ({image_source}):\n"
                response += f"   - Description: {description}\n"
                if ocr_text.strip():
                    response += f"   - Text in image: {ocr_text}\n"
                response += "\n"
            else:
                # Regular text content
                content_preview = content[:300] + "..." if len(content) > 300 else content
                response += f"{i}. From {source}:\n{content_preview}\n\n"
        
        response += "Note: Please install a local language model for more sophisticated answers."
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        collection_info = self.vector_store.get_collection_info()
        return {
            "documents_count": collection_info["count"],
            "model_loaded": self.llm is not None,
            "model_path": self.model_path
        }