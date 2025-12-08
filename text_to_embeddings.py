"""
Text to Chunks & Embeddings Pipeline
Processes IPC text files, chunks them intelligently, generates embeddings, and stores in Pinecone
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IPCChunk:
    """Data structure for IPC text chunk"""
    chunk_id: str
    section_number: str
    section_title: str
    text: str
    chunk_index: int
    metadata: dict
    
    def to_dict(self):
        return asdict(self)


class IPCTextChunker:
    """Intelligent chunking for IPC legal text"""
    
    def __init__(self, chunk_size: int = 30, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_sections(self, text: str) -> List[Dict]:
        """Extract individual IPC sections from text"""
        sections = []
        
        # Pattern to match section headers
        # Examples: "Section 123", "123.", "Section 123A", etc.
        section_pattern = r'(?:Section\s+)?(\d+[A-Z]*)[\.\s\-\:]+([^\n]+)'
        
        # Split text by section markers
        parts = re.split(r'\n(?=(?:Section\s+)?\d+[A-Z]*[\.\s\-\:])', text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Try to extract section number and title
            match = re.match(section_pattern, part, re.IGNORECASE)
            
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                
                # Extract the full text after the header
                section_text = part[match.end():].strip()
                
                sections.append({
                    'section_number': section_num,
                    'section_title': section_title,
                    'full_text': section_text,
                    'raw_text': part
                })
            else:
                # Handle sections without clear headers
                # Try to find section number anywhere in first line
                first_line = part.split('\n')[0]
                sec_num_match = re.search(r'\b(\d+[A-Z]*)\b', first_line)
                
                if sec_num_match:
                    sections.append({
                        'section_number': sec_num_match.group(1),
                        'section_title': first_line[:100],
                        'full_text': part,
                        'raw_text': part
                    })
        
        logger.info(f"Extracted {len(sections)} sections from text")
        return sections
    
    def chunk_text(self, text: str, section_info: dict) -> List[str]:
        """Chunk a single section's text intelligently"""
        chunks = []
        
        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by sentences/paragraphs for better context
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from end of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    # Paragraph itself is too long, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                        else:
                            current_chunk += " " + sent if current_chunk else sent
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks_from_sections(self, sections: List[Dict]) -> List[IPCChunk]:
        """Create chunks from extracted sections"""
        all_chunks = []
        chunk_id_counter = 0
        
        for section in sections:
            section_num = section['section_number']
            section_title = section['section_title']
            text = section['full_text']
            
            # Create chunks for this section
            text_chunks = self.chunk_text(text, section)
            
            for idx, chunk_text in enumerate(text_chunks):
                chunk = IPCChunk(
                    chunk_id=f"ipc_{section_num}_{idx}",
                    section_number=section_num,
                    section_title=section_title,
                    text=chunk_text,
                    chunk_index=idx,
                    metadata={
                        'total_chunks_in_section': len(text_chunks),
                        'char_count': len(chunk_text),
                        'is_first_chunk': idx == 0,
                        'is_last_chunk': idx == len(text_chunks) - 1
                    }
                )
                all_chunks.append(chunk)
                chunk_id_counter += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks: List[IPCChunk], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for all chunks"""
        texts = [chunk.text for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings


class PineconeManager:
    """Manage Pinecone vector database operations"""
    
    def __init__(self, api_key: str, index_name: str = "ipc-rag", dimension: int = 384):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
        logger.info(f"Pinecone initialized")
    
    def create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully")
            else:
                logger.info(f"Index '{self.index_name}' already exists")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating/connecting to index: {e}")
            raise
    
    def upsert_chunks(self, chunks: List[IPCChunk], embeddings: List[np.ndarray], batch_size: int = 100):
        """Upload chunks and embeddings to Pinecone"""
        if not self.index:
            raise ValueError("Index not initialized. Call create_index_if_not_exists() first")
        
        logger.info(f"Upserting {len(chunks)} vectors to Pinecone...")
        
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector_data = {
                'id': chunk.chunk_id,
                'values': embedding.tolist(),
                'metadata': {
                    'section_number': chunk.section_number,
                    'section_title': chunk.section_title,
                    'text': chunk.text[:1000],  # Pinecone metadata limit
                    'chunk_index': chunk.chunk_index,
                    'char_count': chunk.metadata['char_count']
                }
            }
            vectors.append(vector_data)
        
        # Upsert in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Successfully upserted {len(vectors)} vectors")
    
    def get_index_stats(self):
        """Get statistics about the index"""
        if self.index:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        return None


class TextToEmbeddingsPipeline:
    """Complete pipeline from text files to Pinecone embeddings"""
    
    def __init__(
        self,
        text_dir: str = "data/texts",
        pinecone_api_key: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: str = "ipc-rag"
    ):
        self.text_dir = Path(text_dir)
        self.chunker = IPCTextChunker()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("Pinecone API key not provided")
        
        self.pinecone = PineconeManager(
            api_key=pinecone_api_key,
            index_name=index_name,
            dimension=self.embedding_generator.embedding_dim
        )
    
    def process_all_texts(self):
        """Process all text files and upload to Pinecone"""
        if not self.text_dir.exists():
            logger.error(f"Text directory not found: {self.text_dir}")
            return
        
        text_files = list(self.text_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in {self.text_dir}")
            return
        
        logger.info(f"Found {len(text_files)} text files to process")
        
        all_chunks = []
        
        # Process each text file
        for text_file in tqdm(text_files, desc="Processing text files"):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Remove metadata comments
                text = re.sub(r'^#.*\n', '', text, flags=re.MULTILINE)
                
                # Extract sections
                sections = self.chunker.extract_sections(text)
                
                # Create chunks
                chunks = self.chunker.create_chunks_from_sections(sections)
                all_chunks.extend(chunks)
                
                logger.info(f"Processed {text_file.name}: {len(sections)} sections, {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {e}")
        
        if not all_chunks:
            logger.error("No chunks created from text files")
            return
        
        logger.info(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Create Pinecone index
        self.pinecone.create_index_if_not_exists()
        
        # Upload to Pinecone
        self.pinecone.upsert_chunks(all_chunks, embeddings)
        
        # Get final stats
        stats = self.pinecone.get_index_stats()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline Complete!")
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        logger.info(f"Vectors in Pinecone: {stats.get('total_vector_count', 'N/A')}")
        logger.info(f"{'='*60}")


def main():
    """Main execution function"""
    # Get Pinecone API key from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        logger.error("Please set PINECONE_API_KEY environment variable")
        logger.info("Example: export PINECONE_API_KEY='your-api-key-here'")
        return
    
    # Initialize pipeline
    pipeline = TextToEmbeddingsPipeline(
        text_dir="data/texts",
        pinecone_api_key=pinecone_api_key,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_name="ipc-rag"
    )
    
    # Process all texts
    pipeline.process_all_texts()


if __name__ == "__main__":
    main()