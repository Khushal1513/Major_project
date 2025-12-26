"""
User Testing Module for IPC RAG Pipeline
Provides comprehensive testing interface with various query scenarios
"""

import os
import time
import json
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import logging
from dotenv import load_dotenv

# Import the RAG pipeline
from rag_pipeline_extra import RAGPipeline, IPCAnswer

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTester:
    """User testing interface for RAG pipeline"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.test_results = []
        
    def print_separator(self, char="=", length=80):
        """Print a separator line"""
        print(char * length)
    
    def print_header(self, text: str):
        """Print a formatted header"""
        self.print_separator()
        print(f"  {text}")
        self.print_separator()
    
    def display_answer(self, answer: IPCAnswer, show_contexts: bool = False):
        """Display answer in a formatted way"""
        print(f"\n📋 QUESTION:")
        print(f"   {answer.question}\n")
        
        print(f"💡 ANSWER:")
        print(f"   {answer.answer}\n")
        
        print(f"📚 REFERENCED SECTIONS: {', '.join(answer.section_references) if answer.section_references else 'None'}")
        print(f"🎯 CONFIDENCE: {answer.confidence.upper()}")
        print(f"📊 METADATA:")
        print(f"   - Contexts retrieved: {answer.metadata.get('num_contexts', 0)}")
        print(f"   - Avg retrieval score: {answer.metadata.get('avg_retrieval_score', 0):.3f}")
        print(f"   - Top score: {answer.metadata.get('top_score', 0):.3f}")
        
        if show_contexts and answer.retrieved_contexts:
            print(f"\n🔍 TOP RETRIEVED CONTEXTS:")
            for idx, ctx in enumerate(answer.retrieved_contexts[:3], 1):
                print(f"\n   Context {idx}:")
                print(f"   Section: {ctx.section_number} - {ctx.section_title}")
                print(f"   Score: {ctx.score:.3f}")
                print(f"   Text preview: {ctx.text[:150]}...")
    
    def test_single_query(self, query: str, show_contexts: bool = True):
        """Test a single query"""
        self.print_header(f"TESTING QUERY")
        print(f"\n🔍 Query: {query}\n")
        
        start_time = time.time()
        answer = self.pipeline.answer_question(query)
        elapsed_time = time.time() - start_time
        
        self.display_answer(answer, show_contexts=show_contexts)
        print(f"\n⏱️  Response Time: {elapsed_time:.2f} seconds")
        
        # Save result
        self.test_results.append({
            "query": query,
            "answer": answer.answer,
            "sections": answer.section_references,
            "confidence": answer.confidence,
            "response_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        })
        
        return answer
    
    def test_predefined_queries(self):
        """Test with predefined queries covering different scenarios"""
        
        test_queries = {
            "Basic Legal Questions": [
                "What is the punishment for theft under IPC?",
                "What is murder according to IPC?",
                "Is defamation a criminal offense in India?",
                "What is the difference between theft and robbery?"
            ],
            "Specific Section Queries": [
                "Explain Section 302 of IPC",
                "What does Section 420 deal with?",
                "Tell me about Section 376 IPC",
                "What is Section 377 about?"
            ],
            "Procedural Questions": [
                "Is theft a bailable offense?",
                "What is a cognizable offense?",
                "Can I file an FIR for defamation?",
                "What are non-cognizable offenses?"
            ],
            "Comparative Questions": [
                "What is the difference between culpable homicide and murder?",
                "How is assault different from battery?",
                "What's the difference between kidnapping and abduction?",
                "Compare theft and extortion"
            ],
            "Punishment-Related Questions": [
                "What is the maximum penalty for dowry death?",
                "What are the punishments under Section 307?",
                "Is there a death penalty in IPC?",
                "What is life imprisonment in IPC?"
            ],
            "Complex Scenarios": [
                "If someone threatens me with a weapon, what IPC section applies?",
                "What law covers cybercrimes in IPC?",
                "Can a company be held criminally liable under IPC?",
                "What protections exist for women under IPC?"
            ]
        }
        
        print("\n")
        self.print_header("PREDEFINED QUERY TESTING")
        print("\nThis will test the RAG system with various types of queries.\n")
        
        total_queries = sum(len(queries) for queries in test_queries.values())
        current = 0
        
        for category, queries in test_queries.items():
            print(f"\n\n{'='*80}")
            print(f"📁 CATEGORY: {category}")
            print(f"{'='*80}\n")
            
            for query in queries:
                current += 1
                print(f"\n[Query {current}/{total_queries}]")
                self.test_single_query(query, show_contexts=False)
                print("\n" + "-"*80)
                time.sleep(0.5)  # Small delay for readability
    
    def interactive_testing(self):
        """Interactive testing mode"""
        self.print_header("INTERACTIVE TESTING MODE")
        print("\n🎯 Ask any question about the Indian Penal Code")
        print("💡 Commands:")
        print("   - Type your question normally")
        print("   - 'help' - Show example questions")
        print("   - 'stats' - Show testing statistics")
        print("   - 'save' - Save test results")
        print("   - 'quit' or 'exit' - Exit testing\n")
        
        while True:
            try:
                user_input = input("\n🔍 Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Exiting testing mode...")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_example_questions()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                
                elif user_input.lower() == 'save':
                    self.save_results()
                    continue
                
                # Process query
                print()
                answer = self.test_single_query(user_input, show_contexts=True)
                
                # Ask for feedback
                print("\n📝 Was this answer helpful? (y/n/skip): ", end="")
                feedback = input().strip().lower()
                if feedback in ['y', 'n']:
                    self.test_results[-1]['user_feedback'] = 'positive' if feedback == 'y' else 'negative'
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting testing mode...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {e}")
    
    def show_example_questions(self):
        """Show example questions"""
        print("\n" + "="*80)
        print("📚 EXAMPLE QUESTIONS:")
        print("="*80)
        
        examples = [
            "What is the punishment for theft under IPC?",
            "Explain Section 302 of IPC",
            "Is assault a cognizable offense?",
            "What's the difference between murder and culpable homicide?",
            "What are the punishments for dowry death?",
            "Can I file an FIR for defamation?",
            "What does Section 420 deal with?",
            "What protections exist for women under IPC?"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print()
    
    def show_statistics(self):
        """Show testing statistics"""
        if not self.test_results:
            print("\n⚠️  No test results yet!")
            return
        
        print("\n" + "="*80)
        print("📊 TESTING STATISTICS")
        print("="*80)
        
        total_tests = len(self.test_results)
        avg_response_time = sum(r['response_time'] for r in self.test_results) / total_tests
        
        confidence_counts = {}
        for result in self.test_results:
            conf = result['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        positive_feedback = sum(1 for r in self.test_results if r.get('user_feedback') == 'positive')
        negative_feedback = sum(1 for r in self.test_results if r.get('user_feedback') == 'negative')
        
        print(f"\n📈 Total Queries Tested: {total_tests}")
        print(f"⏱️  Average Response Time: {avg_response_time:.2f} seconds")
        print(f"\n🎯 Confidence Distribution:")
        for conf, count in sorted(confidence_counts.items()):
            percentage = (count / total_tests) * 100
            print(f"   {conf.capitalize()}: {count} ({percentage:.1f}%)")
        
        if positive_feedback + negative_feedback > 0:
            print(f"\n👍 User Feedback:")
            print(f"   Positive: {positive_feedback}")
            print(f"   Negative: {negative_feedback}")
            satisfaction_rate = (positive_feedback / (positive_feedback + negative_feedback)) * 100
            print(f"   Satisfaction Rate: {satisfaction_rate:.1f}%")
        
        print()
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not self.test_results:
            print("\n⚠️  No test results to save!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Test results saved to: {filepath}")
    
    def benchmark_performance(self, num_queries: int = 10):
        """Benchmark system performance"""
        self.print_header("PERFORMANCE BENCHMARK")
        
        print(f"\n🚀 Running {num_queries} queries to benchmark performance...\n")
        
        test_queries = [
            "What is theft under IPC?",
            "Explain murder laws",
            "Is defamation criminal?",
            "What is Section 420?",
            "Tell me about assault",
            "What are kidnapping laws?",
            "Explain dowry death",
            "What is culpable homicide?",
            "Is theft bailable?",
            "What is robbery?"
        ]
        
        queries_to_test = (test_queries * (num_queries // len(test_queries) + 1))[:num_queries]
        
        times = []
        for i, query in enumerate(queries_to_test, 1):
            print(f"Query {i}/{num_queries}...", end=" ")
            start = time.time()
            self.pipeline.answer_question(query, return_contexts=False)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"{elapsed:.2f}s")
        
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"   Total queries: {num_queries}")
        print(f"   Average time: {sum(times)/len(times):.2f}s")
        print(f"   Min time: {min(times):.2f}s")
        print(f"   Max time: {max(times):.2f}s")
        print(f"   Total time: {sum(times):.2f}s")
        print()


def main():
    """Main testing function"""
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\AI_Class\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", "0"))
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return
    
    if not os.path.exists(LLAMA_MODEL_PATH):
        print(f"❌ Error: LLaMA model not found at: {LLAMA_MODEL_PATH}")
        print("Please download a GGUF model and set LLAMA_MODEL_PATH")
        return
    
    print("\n" + "="*80)
    print("🚀 IPC RAG PIPELINE - USER TESTING SUITE")
    print("="*80)
    print("\nInitializing RAG Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(
            pinecone_api_key=PINECONE_API_KEY,
            llama_model_path=LLAMA_MODEL_PATH,
            index_name="ipc-rag",
            gpu_layers=GPU_LAYERS
        )
        
        # Create tester
        tester = RAGTester(pipeline)
        
        print("\n✅ Pipeline initialized successfully!")
        
        # Main menu
        while True:
            print("\n" + "="*80)
            print("📋 TESTING MENU")
            print("="*80)
            print("\n1. 🎯 Interactive Testing (Ask your own questions)")
            print("2. 📝 Run Predefined Test Suite")
            print("3. 🔍 Test Single Query")
            print("4. 🚀 Performance Benchmark")
            print("5. 📊 View Statistics")
            print("6. 💾 Save Results")
            print("7. 🚪 Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                tester.interactive_testing()
            
            elif choice == '2':
                tester.test_predefined_queries()
            
            elif choice == '3':
                query = input("\nEnter your question: ").strip()
                if query:
                    tester.test_single_query(query, show_contexts=True)
            
            elif choice == '4':
                num = input("\nNumber of queries for benchmark (default 10): ").strip()
                num = int(num) if num.isdigit() else 10
                tester.benchmark_performance(num)
            
            elif choice == '5':
                tester.show_statistics()
            
            elif choice == '6':
                tester.save_results()
            
            elif choice == '7':
                print("\n👋 Thank you for testing! Goodbye!")
                break
            
            else:
                print("\n⚠️  Invalid option. Please select 1-7.")
    
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()