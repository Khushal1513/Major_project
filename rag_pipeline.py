import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from typing import List, Dict

# Load environment variables
load_dotenv()

class IndianLawRAG:
    def __init__(self):
        """Initialize the RAG pipeline with Pinecone and Groq"""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embedding model
        embedding_model_name = os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Indian law keywords for validation
        self.indian_law_keywords = [
            'ipc', 'indian penal code', 'constitution', 'article', 'section',
            'crpc', 'criminal procedure code', 'supreme court', 'high court',
            'fundamental rights', 'directive principles', 'amendment',
            'bharatiya nyaya sanhita', 'bns','right', 'rights',
            'bharatiya nagarik suraksha sanhita', 'bnss',
            'law', 'legal', 'india', 'indian', 'act', 'rights',
            'criminal', 'civil', 'procedure', 'evidence act', 'contract act'
        ]
    
    def is_indian_law_query(self, query: str) -> bool:
        """Check if the query is related to Indian laws"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.indian_law_keywords)
    
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text"""
        embeddings = self.embedding_model.encode(
            text,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context from Pinecone"""
        query_embedding = self.get_embeddings(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = []
        for match in results['matches']:
            contexts.append({
                'text': match['metadata'].get('text', ''),
                'score': match['score'],
                'source': match['metadata'].get('source', 'Unknown'),
                'section': match['metadata'].get('section', '')
            })
        
        return contexts
    
    def generate_response(self, query: str, contexts: List[Dict]) -> str:
        """Generate response using Groq API with retrieved context"""
        context_text = "\n\n".join([
            f"[Source: {ctx['source']} - {ctx['section']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        system_prompt = """You are an expert AI assistant specialized in Indian Constitution and Laws, including IPC, CrPC, BNS, BNSS, and other Indian legal statutes.

Your responsibilities:
1. Provide accurate legal explanations
2. Cite sections/articles when relevant
3. Explain concepts clearly
4. Use only the provided legal context
"""

        user_prompt = f"""Based on the following legal context, answer the question.

CONTEXT:
{context_text}

USER QUESTION:
{query}

Answer:"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """Main query method for the RAG pipeline"""
        if not self.is_indian_law_query(user_query):
            return {
                'answer': (
                    "I can only answer questions related to Indian Constitution "
                    "and Laws (IPC, CrPC, BNS, BNSS, etc.)."
                ),
                'contexts': [],
                'is_valid_query': False
            }
        
        contexts = self.retrieve_context(user_query, top_k)
        
        if not contexts or all(ctx['text'] == '' for ctx in contexts):
            return {
                'answer': (
                    "No relevant legal information found. "
                    "Please rephrase or mention specific sections/articles."
                ),
                'contexts': [],
                'is_valid_query': True
            }
        
        answer = self.generate_response(user_query, contexts)
        
        return {
            'answer': answer,
            'contexts': contexts,
            'is_valid_query': True
        }


def get_rag_pipeline():
    """Factory function"""
    return IndianLawRAG()