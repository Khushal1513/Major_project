"""
RAG Pipeline - Retrieval-Augmented Generation System
Combines semantic search with LLM inference for accurate IPC answers
Updated for Mistral Model Support
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from ctransformers import AutoModelForCausalLM
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Optional Groq integration
try:
    from groq_client import generate as groq_generate
except Exception:
    groq_generate = None

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
        
        # Initialize embedding model with fallback
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model, device='cuda')
            logger.info("Embedding model loaded on CUDA")
        except Exception as e:
            logger.warning(f"Could not load on CUDA: {e}")
            logger.info("Falling back to CPU...")
            self.embedding_model = SentenceTransformer(embedding_model, device='cpu')
            logger.info("Embedding model loaded on CPU")
        
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(index_name)
            
            # Verify index has data
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"Connected to Pinecone index: {index_name}")
            logger.info(f"Index contains {vector_count} vectors")
            
            if vector_count == 0:
                logger.warning("Warning: Index is empty! Please run text_to_embeddings.py first")
                
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convert query to embedding vector"""
        try:
            embedding = self.embedding_model.encode(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
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
            for match in results.get('matches', []):
                context = RetrievedContext(
                    section_number=match['metadata'].get('section_number', 'Unknown'),
                    section_title=match['metadata'].get('section_title', 'Unknown'),
                    text=match['metadata'].get('text', ''),
                    score=match['score'],
                    chunk_index=match['metadata'].get('chunk_index', 0)
                )
                contexts.append(context)
            
            logger.info(f"Retrieved {len(contexts)} contexts for query: '{query[:50]}...'")
            
            if contexts:
                logger.info(f"Top match score: {contexts[0].score:.3f}")
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []


class MistralGenerator:
    """Generate human-understandable answers using local Mistral model"""
    
    def __init__(
        self,
        model_path: str = r"D:\AI_Class\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type: str = "mistral",
        max_tokens: int = 512,
        temperature: float = 0.3,
        gpu_layers: int = 0
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please download a GGUF model and place it at this path.\n"
                f"Current expected path: {model_path}"
            )
        
        logger.info(f"Loading Mistral model from: {model_path}")
        logger.info(f"GPU layers: {gpu_layers} (set to 50+ for GPU acceleration)")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                max_new_tokens=max_tokens,
                temperature=temperature,
                context_length=4096,  # Mistral supports larger context
                gpu_layers=gpu_layers,
                threads=4
            )
            logger.info("Mistral model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            logger.error("Make sure you have ctransformers installed: pip install ctransformers")
            raise
    
    def create_prompt(self, question: str, contexts: List[RetrievedContext], max_context_length: int = 2000) -> str:
        """Create structured prompt for Mistral with Instruct format"""
        
        # Build context section with length limit
        context_text = ""
        total_length = 0
        contexts_used = 0
        
        for idx, ctx in enumerate(contexts, 1):
            ctx_block = f"\n--- Context {idx} (Section {ctx.section_number}) ---\n"
            ctx_block += f"Title: {ctx.section_title}\n"
            ctx_block += f"Text: {ctx.text}\n"
            
            # Check if adding this context exceeds limit
            if total_length + len(ctx_block) > max_context_length:
                logger.warning(f"Context length limit reached. Using {contexts_used}/{len(contexts)} contexts")
                break
            
            context_text += ctx_block
            total_length += len(ctx_block)
            contexts_used += 1
        
        # Mistral Instruct format: [INST] instruction [/INST]
        prompt = f"""[INST] You are an expert legal assistant specializing in the Indian Penal Code (IPC). Your task is to provide accurate, clear, and helpful answers based ONLY on the provided legal contexts.

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
3. Cite specific section numbers when referencing laws
4. If the contexts don't fully answer the question, say so clearly
5. Use simple, understandable language
6. Be precise and avoid speculation

Provide a comprehensive answer now: [/INST]"""
        
        logger.info(f"Prompt created with {contexts_used} contexts ({total_length} chars)")
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Generate answer using Mistral model"""
        try:
            logger.info("Generating answer...")

            # If Groq is configured, use remote inference
            if groq_generate is not None and os.getenv("GROQ_API_KEY"):
                logger.info("GROQ_API_KEY detected: using Groq remote inference for Mistral prompt")
                response = groq_generate(prompt, max_tokens=self.max_tokens, temperature=self.temperature, model=os.getenv("GROQ_MODEL"))
                return response.strip()

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
        # Also find standalone section numbers
        standalone = re.findall(r'\b(\d{2,3}[A-Z]?)\b', answer)
        all_sections = sections + standalone
        return list(set(all_sections))


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(
        self,
        pinecone_api_key: str,
        model_path: str = None,
        llama_model_path: str = None,
        index_name: str = "ipc-rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gpu_layers: int = 0
    ):
        logger.info("Initializing RAG Pipeline...")

        # Backwards-compatibility: allow callers to pass `llama_model_path`
        if model_path is None and llama_model_path is not None:
            model_path = llama_model_path

        if model_path is None:
            raise ValueError("model_path (or llama_model_path) must be provided to initialize RAGPipeline")
        
        # Initialize retriever
        try:
            self.retriever = VectorRetriever(
                pinecone_api_key=pinecone_api_key,
                index_name=index_name,
                embedding_model=embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
        
        # Initialize generator
        try:
            self.generator = MistralGenerator(
                model_path=model_path,
                gpu_layers=gpu_layers
            )
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise
        
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
            logger.warning("No contexts retrieved")
            return IPCAnswer(
                question=question,
                answer="I couldn't find relevant information in the IPC database to answer this question. Please try rephrasing or ask about a specific IPC section.",
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
            if ctx.section_number not in section_refs and ctx.section_number != 'Unknown':
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
            section_references=sorted(set(section_refs), key=lambda x: (len(x), x)),
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
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*80)
        print("IPC RAG Assistant - Interactive Mode (Powered by Mistral)")
        print("="*80)
        print("Ask questions about the Indian Penal Code. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using IPC RAG Assistant!")
                    break
                
                # Get answer
                answer = self.answer_question(question)
                
                # Display answer
                print(f"\n{'='*80}")
                print(f"Answer:")
                print(f"{'='*80}")
                print(f"\n{answer.answer}\n")
                print(f"Relevant Sections: {', '.join(answer.section_references)}")
                print(f"Confidence: {answer.confidence}")
                
            except KeyboardInterrupt:
                print("\n\nThank you for using IPC RAG Assistant!")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"\nError: {e}")


def main():
    """Test the RAG pipeline"""
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\AI_Class\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))  # Set to 50+ for GPU
    
    if not PINECONE_API_KEY:
        logger.error("Please set PINECONE_API_KEY environment variable")
        logger.error("Example: export PINECONE_API_KEY='your_api_key'")
        return
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at: {MODEL_PATH}")
        logger.error("Please check the model path in your .env file")
        return
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            model_path=MODEL_PATH,
            index_name="ipc-rag",
            gpu_layers=GPU_LAYERS
        )
        
        # Test questions
        test_questions = [
            "What is the punishment for theft under IPC?",
            "Explain Section 302 of IPC",
            "What constitutes murder according to Indian Penal Code?",
            "Is defamation a criminal offense in India?"
        ]
        
        print("\n" + "="*80)
        print("Testing RAG Pipeline with Sample Questions (Mistral Model)")
        print("="*80)
        
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
        
        # Start interactive mode
        print("\n" + "="*80)
        response = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            pipeline.interactive_mode()
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()