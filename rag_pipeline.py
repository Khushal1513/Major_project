"""
RAG Pipeline - Retrieval-Augmented Generation System
Combines semantic search with LLM inference for accurate IPC answers
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from ctransformers import AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Retrieved context from vector search"""
    section_number: str
    section_title: str
    text: str
    score: float
    chunk_index: int


@dataclass
class IPCAnswer:
    """Structured answer for IPC queries"""
    question: str
    answer: str
    section_references: List[str]
    retrieved_contexts: List[RetrievedContext]
    confidence: str
    metadata: dict


class VectorRetriever:
    """Retrieve relevant IPC sections using semantic search"""
    
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "ipc-rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_name = index_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convert query to embedding vector"""
        embedding = self.embedding_model.encode(query)
        return embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        """Retrieve top-k relevant contexts for query"""
        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Parse results
            contexts = []
            for match in results['matches']:
                context = RetrievedContext(
                    section_number=match['metadata'].get('section_number', 'Unknown'),
                    section_title=match['metadata'].get('section_title', 'Unknown'),
                    text=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_index=match['metadata'].get('chunk_index', 0)
                )
                contexts.append(context)
            
            logger.info(f"Retrieved {len(contexts)} contexts for query: '{query[:50]}...'")
            return contexts
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []


class LLaMAGenerator:
    """Generate human-understandable answers using local LLaMA model"""
    
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.Q4_K_M.gguf",
        model_type: str = "llama",
        max_tokens: int = 512,
        temperature: float = 0.3
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Loading LLaMA model from: {model_path}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                max_new_tokens=max_tokens,
                temperature=temperature,
                context_length=2048,
                gpu_layers=0  # Set to 50+ if you have GPU
            )
            logger.info("LLaMA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLaMA model: {e}")
            raise
    
    def create_prompt(self, question: str, contexts: List[RetrievedContext]) -> str:
        """Create structured prompt for LLaMA"""
        
        # Build context section
        context_text = ""
        for idx, ctx in enumerate(contexts, 1):
            context_text += f"\n--- Context {idx} (Section {ctx.section_number}) ---\n"
            context_text += f"Title: {ctx.section_title}\n"
            context_text += f"Text: {ctx.text}\n"
        
        # Create prompt
        prompt = f"""You are an expert legal assistant specializing in the Indian Penal Code (IPC). Your task is to provide accurate, clear, and helpful answers based ONLY on the provided legal contexts.

LEGAL CONTEXTS:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the contexts above
2. Structure your answer clearly with:
   - Brief explanation of the relevant IPC section(s)
   - Key points and definitions
   - Punishment details (if applicable)
   - Classification (bailable/non-bailable, cognizable/non-cognizable if mentioned)
   - Simple example to illustrate (if appropriate)
3. Cite specific section numbers when referencing laws
4. If the contexts don't fully answer the question, say so clearly
5. Use simple, understandable language for general public
6. Be precise and avoid speculation

ANSWER:"""
        
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Generate answer using LLaMA model"""
        try:
            logger.info("Generating answer...")
            response = self.model(prompt)
            logger.info("Answer generated successfully")
            return response.strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating answer: {str(e)}"
    
    def extract_section_references(self, answer: str) -> List[str]:
        """Extract section numbers mentioned in the answer"""
        import re
        sections = re.findall(r'Section\s+(\d+[A-Z]*)', answer, re.IGNORECASE)
        return list(set(sections))


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(
        self,
        pinecone_api_key: str,
        llama_model_path: str,
        index_name: str = "ipc-rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        # Initialize retriever
        self.retriever = VectorRetriever(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            embedding_model=embedding_model
        )
        
        # Initialize generator
        self.generator = LLaMAGenerator(
            model_path=llama_model_path
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        return_contexts: bool = True
    ) -> IPCAnswer:
        """Answer a question using RAG pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing question: {question}")
        logger.info(f"{'='*60}")
        
        # Step 1: Retrieve relevant contexts
        contexts = self.retriever.retrieve(question, top_k=top_k)
        
        if not contexts:
            return IPCAnswer(
                question=question,
                answer="I couldn't find relevant information in the IPC database to answer this question.",
                section_references=[],
                retrieved_contexts=[],
                confidence="low",
                metadata={"error": "No contexts retrieved"}
            )
        
        # Step 2: Create prompt
        prompt = self.generator.create_prompt(question, contexts)
        
        # Step 3: Generate answer
        answer = self.generator.generate(prompt)
        
        # Step 4: Extract section references
        section_refs = self.generator.extract_section_references(answer)
        
        # Add context sections
        for ctx in contexts:
            if ctx.section_number not in section_refs:
                section_refs.append(ctx.section_number)
        
        # Determine confidence based on retrieval scores
        avg_score = np.mean([ctx.score for ctx in contexts])
        if avg_score > 0.8:
            confidence = "high"
        elif avg_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Create structured answer
        ipc_answer = IPCAnswer(
            question=question,
            answer=answer,
            section_references=sorted(set(section_refs)),
            retrieved_contexts=contexts if return_contexts else [],
            confidence=confidence,
            metadata={
                "num_contexts": len(contexts),
                "avg_retrieval_score": float(avg_score),
                "top_score": float(contexts[0].score) if contexts else 0.0
            }
        )
        
        logger.info(f"Answer generated with {len(section_refs)} section references")
        logger.info(f"Confidence: {confidence}")
        
        return ipc_answer
    
    def batch_answer(self, questions: List[str], top_k: int = 5) -> List[IPCAnswer]:
        """Answer multiple questions"""
        answers = []
        for question in questions:
            answer = self.answer_question(question, top_k=top_k)
            answers.append(answer)
        return answers


def main():
    """Test the RAG pipeline"""
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
    
    if not PINECONE_API_KEY:
        logger.error("Please set PINECONE_API_KEY environment variable")
        return
    
    if not os.path.exists(LLAMA_MODEL_PATH):
        logger.error(f"LLaMA model not found at: {LLAMA_MODEL_PATH}")
        logger.info("Please download a GGUF model and set LLAMA_MODEL_PATH")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        pinecone_api_key=PINECONE_API_KEY,
        llama_model_path=LLAMA_MODEL_PATH,
        index_name="ipc-rag"
    )
    
    # Test questions
    test_questions = [
        "What is the punishment for theft under IPC?",
        "Explain Section 302 of IPC",
        "What constitutes murder according to Indian Penal Code?",
        "Is defamation a criminal offense in India?"
    ]
    
    # Answer questions
    for question in test_questions:
        answer = pipeline.answer_question(question)
        
        print(f"\n{'='*80}")
        print(f"Q: {answer.question}")
        print(f"{'='*80}")
        print(f"\nA: {answer.answer}")
        print(f"\nSection References: {', '.join(answer.section_references)}")
        print(f"Confidence: {answer.confidence}")
        print(f"\nTop Retrieved Sections:")
        for ctx in answer.retrieved_contexts[:3]:
            print(f"  - Section {ctx.section_number}: {ctx.section_title} (score: {ctx.score:.3f})")


if __name__ == "__main__":
    main()