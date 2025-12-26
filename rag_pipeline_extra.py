"""
RAG Pipeline - Optimized for Phi-3-Mini Model
Fast CPU inference with excellent quality for IPC Q&A
"""

import os
import logging
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from ctransformers import AutoModelForCausalLM
from pathlib import Path
from dotenv import load_dotenv
import time
import threading

load_dotenv()

# Optional Groq integration (remote LLM). If `GROQ_API_KEY` is present in
# the environment, the pipeline will call Groq via `groq_client.generate()`.
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


class ProgressIndicator:
    """Show progress indicator for long-running operations"""
    
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.thread = None
    
    def _animate(self):
        """Animation loop"""
        chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        while self.running:
            sys.stdout.write(f'\r{chars[idx % len(chars)]} {self.message}...')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
    
    def start(self):
        """Start the progress indicator"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the progress indicator"""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()


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


class Phi3Generator:
    """Generate answers using Phi-3-Mini model (optimized for CPU)"""
    
    def __init__(
        self,
        model_path: str = r"D:\AI_Class\models\Phi-3-mini-4k-instruct-q4.gguf",
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
                f"Please download Phi-3-mini-4k-instruct-q4.gguf and place it at this path.\n"
                f"Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
            )
        
        logger.info(f"Loading Phi-3-Mini model from: {model_path}")
        logger.info(f"GPU layers: {gpu_layers} (0 = CPU only, increase for GPU)")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="phi3",  # Phi-3 model type
                max_new_tokens=max_tokens,
                temperature=temperature,
                context_length=4096,  # Phi-3-Mini supports 4K context
                gpu_layers=gpu_layers,
                threads=4,  # Adjust based on your CPU cores
                batch_size=1
            )
            logger.info("Phi-3-Mini model loaded successfully")
            logger.info("Expected response time: 5-15 seconds on CPU")
        except Exception as e:
            # Primary load failed — attempt a minimal fallback to improve diagnostics
            logger.error(f"Error loading Phi-3-Mini model (primary): {e}")
            logger.info("Attempting fallback load without explicit model_type/extra kwargs...")
            try:
                # Try a simpler load which lets ctransformers auto-detect model type
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    threads=2
                )
                logger.warning("Fallback load succeeded (auto-detected model type).")
            except Exception as e2:
                logger.error(f"Fallback load failed: {e2}")
                logger.error("Make sure you have ctransformers installed: pip install ctransformers")
                logger.error("Also ensure the GGUF file is compatible and not corrupted.")
                logger.error("You can run a quick validation by trying a minimal ctransformers load in a Python REPL:")
                logger.error("  from ctransformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('path/to/model.gguf')")
                # Re-raise the original exception for visibility
                raise RuntimeError(f"Failed to create LLM from '{model_path}': {e2}") from e2
    
    def create_prompt(self, question: str, contexts: List[RetrievedContext], max_context_length: int = 1500) -> str:
        """Create structured prompt for Phi-3 with ChatML format"""
        
        # Build context section with length limit
        context_text = ""
        total_length = 0
        contexts_used = 0
        
        for idx, ctx in enumerate(contexts, 1):
            ctx_block = f"\nContext {idx} - Section {ctx.section_number}: {ctx.section_title}\n"
            ctx_block += f"{ctx.text}\n"
            
            # Check if adding this context exceeds limit
            if total_length + len(ctx_block) > max_context_length:
                logger.warning(f"Context length limit reached. Using {contexts_used}/{len(contexts)} contexts")
                break
            
            context_text += ctx_block
            total_length += len(ctx_block)
            contexts_used += 1
        
        # Phi-3 uses ChatML format: <|user|>\n{prompt}<|end|>\n<|assistant|>
        prompt = f"""<|system|>
You are an expert legal assistant specializing in the Indian Penal Code (IPC). Provide accurate answers based only on the provided legal contexts.<|end|>
<|user|>
LEGAL CONTEXTS FROM IPC:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the provided contexts
- Cite specific IPC section numbers
- Include punishment details if mentioned
- Mention if offense is bailable/non-bailable, cognizable/non-cognizable (if stated)
- Use clear, simple language
- If contexts don't fully answer, state this clearly

Provide your answer:<|end|>
<|assistant|>"""
        
        logger.info(f"Prompt created with {contexts_used} contexts ({total_length} chars)")
        return prompt
    
    def generate(self, prompt: str, show_progress: bool = True) -> str:
        """Generate answer using Phi-3 model with progress indicator"""
        try:
            logger.info("Generating answer with Phi-3-Mini...")

            # If Groq is configured, prefer remote Groq inference
            if groq_generate is not None and os.getenv("GROQ_API_KEY"):
                logger.info("GROQ_API_KEY detected: using Groq remote inference")
                if show_progress:
                    print("\n⚡ Generating answer via Groq API...")
                    progress = ProgressIndicator("Calling Groq")
                    progress.start()

                try:
                    start_time = time.time()
                    response = groq_generate(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        model=os.getenv("GROQ_MODEL")
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"Groq answer generated in {elapsed:.1f}s")
                finally:
                    if show_progress:
                        progress.stop()

                # Groq client returns string
                return response.strip()

            # Fallback to local model inference
            if show_progress:
                print("\n⚡ Generating answer (Phi-3-Mini on CPU: ~5-15 seconds)...")
                progress = ProgressIndicator("Processing")
                progress.start()

            start_time = time.time()
            response = self.model(prompt)
            elapsed = time.time() - start_time

            if show_progress:
                progress.stop()
                print(f"✅ Answer generated in {elapsed:.1f} seconds")

            logger.info(f"Answer generated successfully in {elapsed:.1f}s")

            # Clean up response
            response = response.strip()

            # Remove any trailing special tokens
            if '<|end|>' in response:
                response = response.split('<|end|>')[0].strip()

            return response
            
        except Exception as e:
            if show_progress:
                progress.stop()
            logger.error(f"Error during generation: {e}")
            return f"Error generating answer: {str(e)}"
    
    def extract_section_references(self, answer: str) -> List[str]:
        """Extract section numbers mentioned in the answer"""
        import re
        
        # Find patterns like "Section 123", "Section 123A", etc.
        sections = re.findall(r'Section\s+(\d+[A-Z]*)', answer, re.IGNORECASE)
        
        # Find standalone section numbers (2-3 digits, possibly with letter)
        standalone = re.findall(r'\b(\d{2,3}[A-Z]?)\b', answer)
        
        # Combine and deduplicate
        all_sections = list(set(sections + standalone))
        
        return all_sections


class MistralGenerator:
    """Generate answers using Mistral model (fallback when Phi-3 unavailable)"""
    
    def __init__(
        self,
        model_path: str = r"D:\AI_Class\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
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
                f"Please ensure Mistral model is available at this path."
            )
        
        logger.info(f"Loading Mistral model from: {model_path}")
        logger.info(f"GPU layers: {gpu_layers} (0 = CPU only, increase for GPU)")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",
                max_new_tokens=max_tokens,
                temperature=temperature,
                context_length=4096,
                gpu_layers=gpu_layers,
                threads=4,
                batch_size=1
            )
            logger.info("Mistral model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            logger.info("Attempting fallback load without explicit model_type...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    threads=2
                )
                logger.warning("Fallback load succeeded (auto-detected model type).")
            except Exception as e2:
                logger.error(f"Fallback load failed: {e2}")
                raise RuntimeError(f"Failed to load Mistral model from '{model_path}': {e2}") from e2
    
    def create_prompt(self, question: str, contexts: List[RetrievedContext], max_context_length: int = 1500) -> str:
        """Create structured prompt for Mistral"""
        
        context_text = ""
        total_length = 0
        contexts_used = 0
        
        for idx, ctx in enumerate(contexts, 1):
            ctx_block = f"\nContext {idx} - Section {ctx.section_number}: {ctx.section_title}\n"
            ctx_block += f"{ctx.text}\n"
            
            if total_length + len(ctx_block) > max_context_length:
                logger.warning(f"Context length limit reached. Using {contexts_used}/{len(contexts)} contexts")
                break
            
            context_text += ctx_block
            total_length += len(ctx_block)
            contexts_used += 1
        
        prompt = f"""You are an expert legal assistant specializing in the Indian Penal Code (IPC). Provide accurate answers based only on the provided legal contexts.

LEGAL CONTEXTS FROM IPC:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the provided contexts
2. Structure clearly with explanation, key points, and punishments
3. Cite specific section numbers
4. Be precise and avoid speculation

ANSWER:"""
        
        logger.info(f"Prompt created with {contexts_used} contexts ({total_length} chars)")
        return prompt
    
    def generate(self, prompt: str, show_progress: bool = False) -> str:
        """Generate answer using Mistral model"""
        try:
            logger.info("Generating answer with Mistral...")

            # Use Groq remote inference if configured
            if groq_generate is not None and os.getenv("GROQ_API_KEY"):
                logger.info("GROQ_API_KEY detected: using Groq remote inference for Mistral prompt")
                if show_progress:
                    print("\n⚡ Generating answer via Groq API...")
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
        standalone = re.findall(r'\b(\d{2,3}[A-Z]?)\b', answer)
        all_sections = list(set(sections + standalone))
        return all_sections


class RAGPipeline:
    """Complete RAG pipeline optimized for Phi-3-Mini"""
    
    def __init__(
        self,
        pinecone_api_key: str,
        model_path: str,
        index_name: str = "ipc-rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gpu_layers: int = 0
    ):
        logger.info("Initializing RAG Pipeline with Phi-3-Mini...")
        
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
        
        # Initialize Phi-3 generator with fallback to Mistral
        self.generator = None
        try:
            self.generator = Phi3Generator(
                model_path=model_path,
                gpu_layers=gpu_layers
            )
            logger.info("RAG Pipeline initialized successfully")
            logger.info("🚀 Ready for fast inference with Phi-3-Mini!")
        except Exception as e:
            logger.error(f"Failed to initialize Phi-3 generator: {e}")
            logger.warning("Attempting fallback to Mistral model...")
            
            # Fallback: try to load Mistral instead (always from models/ directory)
            try:
                mistral_path = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
                if os.path.exists(mistral_path):
                    self.generator = MistralGenerator(
                        model_path=mistral_path,
                        gpu_layers=gpu_layers
                    )
                    logger.warning(f"✅ Fallback succeeded! Using Mistral model: {mistral_path}")
                    logger.warning("Note: Response times may be ~2-5x slower than Phi-3")
                else:
                    logger.error(f"Fallback model not found: {mistral_path}")
                    raise FileNotFoundError(f"Neither Phi-3 nor Mistral found. Tried:\n  {model_path}\n  {mistral_path}")
            except Exception as e2:
                logger.error(f"Fallback to Mistral also failed: {e2}")
                logger.error("Cannot initialize any generator. Please check your model files.")
                raise RuntimeError(f"Failed to initialize both Phi-3 and Mistral generators:\n  Phi-3: {e}\n  Mistral: {e2}") from e2
        
        logger.info("RAG Pipeline initialized successfully")
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        return_contexts: bool = True,
        show_progress: bool = True
    ) -> IPCAnswer:
        """Answer a question using RAG pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing question: {question}")
        logger.info(f"{'='*60}")
        
        # Step 1: Retrieve relevant contexts
        if show_progress:
            print("\n🔍 Retrieving relevant IPC sections...")
        
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
        
        if show_progress:
            print(f"✅ Retrieved {len(contexts)} relevant sections (top score: {contexts[0].score:.3f})")
        
        # Step 2: Create prompt
        prompt = self.generator.create_prompt(question, contexts)
        
        # Step 3: Generate answer
        answer = self.generator.generate(prompt, show_progress=show_progress)
        
        # Step 4: Extract section references
        section_refs = self.generator.extract_section_references(answer)
        
        # Add context sections that might not be mentioned in answer
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
                "top_score": float(contexts[0].score) if contexts else 0.0,
                "model": "Phi-3-Mini-4K-Instruct"
            }
        )
        
        logger.info(f"Answer generated with {len(section_refs)} section references")
        logger.info(f"Confidence: {confidence}")
        
        return ipc_answer
    
    def batch_answer(self, questions: List[str], top_k: int = 5) -> List[IPCAnswer]:
        """Answer multiple questions"""
        answers = []
        total = len(questions)
        
        for idx, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"Question {idx}/{total}")
            print(f"{'='*80}")
            
            answer = self.answer_question(question, top_k=top_k)
            answers.append(answer)
            
            # Brief pause between questions
            if idx < total:
                time.sleep(0.5)
        
        return answers
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*80)
        print("🚀 IPC RAG Assistant - Interactive Mode (Powered by Phi-3-Mini)")
        print("="*80)
        print("⚡ Fast CPU inference - Expect 5-15 second responses")
        print("Ask questions about the Indian Penal Code. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using IPC RAG Assistant!")
                    break
                
                # Get answer
                answer = self.answer_question(question)
                
                # Display answer
                print(f"\n{'='*80}")
                print(f"📋 ANSWER:")
                print(f"{'='*80}")
                print(f"\n{answer.answer}\n")
                print(f"{'─'*80}")
                print(f"📚 Relevant Sections: {', '.join(answer.section_references) if answer.section_references else 'None'}")
                print(f"🎯 Confidence: {answer.confidence.upper()}")
                print(f"📊 Retrieval Score: {answer.metadata['avg_retrieval_score']:.3f}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using IPC RAG Assistant!")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"\n❌ Error: {e}")


def main():
    """Test the RAG pipeline with Phi-3-Mini"""
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Try huggingface-hub cache location first, fall back to models/ directory
    phi3_cache_path = r"D:\AI_Class\models\models--microsoft--Phi-3-mini-4k-instruct-gguf\snapshots\a64113399c2f6b8ad3e11c394733a2ddadaa7f33\Phi-3-mini-4k-instruct-q4.gguf"
    phi3_fallback_path = r"D:\AI_Class\models\Phi-3-mini-4k-instruct-q4.gguf"
    
    if os.path.exists(phi3_cache_path):
        MODEL_PATH = phi3_cache_path
    elif os.path.exists(phi3_fallback_path):
        MODEL_PATH = phi3_fallback_path
    else:
        MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", phi3_fallback_path)
    
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Phi-3-Mini model not found at: {MODEL_PATH}")
        print("\n📥 Download instructions:")
        print("1. Go to: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
        print("2. Download: Phi-3-mini-4k-instruct-q4.gguf (2.4 GB)")
        print(f"3. Save to: {MODEL_PATH}")
        return
    
    try:
        print("\n" + "="*80)
        print("🚀 IPC RAG PIPELINE - PHI-3-MINI (FAST CPU INFERENCE)")
        print("="*80)
        print("\nInitializing pipeline...\n")
        
        # Initialize pipeline
        pipeline = RAGPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            model_path=MODEL_PATH,
            index_name="ipc-rag",
            gpu_layers=GPU_LAYERS
        )
        
        print("\n✅ Pipeline initialized successfully!")
        print("⚡ Phi-3-Mini is optimized for fast CPU inference")
        
        # Test questions
        test_questions = [
            "What is the punishment for theft under IPC?",
            "Explain Section 302 of IPC",
            "Is defamation a criminal offense in India?",
        ]
        
        print("\n" + "="*80)
        print("📝 TESTING WITH SAMPLE QUESTIONS")
        print("="*80)
        
        # Answer questions
        for question in test_questions:
            print(f"\n{'─'*80}")
            print(f"Q: {question}")
            print(f"{'─'*80}")
            
            answer = pipeline.answer_question(question, show_progress=True)
            
            print(f"\n📋 ANSWER:")
            print(f"{answer.answer}\n")
            print(f"📚 Sections: {', '.join(answer.section_references)}")
            print(f"🎯 Confidence: {answer.confidence}")
        
        # Start interactive mode
        print("\n" + "="*80)
        response = input("\n Would you like to enter interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            pipeline.interactive_mode()
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()