import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain.schema import Document

class VectorStore:
    def __init__(self, db_path: str = "./vector_db"):
        self.db_path = db_path
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documentation2",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Generate IDs
        ids = [f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i)}" 
               for i, doc in enumerate(documents)]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name
        }