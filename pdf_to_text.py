"""
PDF to Text Converter for IPC Documents
Extracts clean text from IPC PDFs and saves to .txt format
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional
import PyPDF2
import pdfplumber
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFToTextConverter:
    """Converts PDF files to clean text format with IPC-specific preprocessing"""
    
    def __init__(self, input_dir: str = "data/pdfs", output_dir: str = "data/texts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (primary method, better quality)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR issues
        text = text.replace('|', 'I')
        text = text.replace('0', 'O').replace('O', '0')  # Context-dependent
        
        # Normalize section headers
        text = re.sub(r'Section\s+(\d+[A-Z]*)', r'Section \1', text)
        text = re.sub(r'§\s*(\d+)', r'Section \1', text)
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'THE INDIAN PENAL CODE.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'IPC\s+\d{4}.*?\n', '', text, flags=re.IGNORECASE)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_ipc_metadata(self, text: str) -> dict:
        """Extract metadata from IPC text"""
        metadata = {
            'sections_found': [],
            'chapters_found': [],
            'total_sections': 0
        }
        
        # Find all section numbers
        sections = re.findall(r'Section\s+(\d+[A-Z]*)', text, re.IGNORECASE)
        metadata['sections_found'] = list(set(sections))
        metadata['total_sections'] = len(metadata['sections_found'])
        
        # Find chapter references
        chapters = re.findall(r'CHAPTER\s+([IVXLCDM]+)', text, re.IGNORECASE)
        metadata['chapters_found'] = list(set(chapters))
        
        return metadata
    
    def convert_pdf_to_text(self, pdf_path: Path) -> Optional[str]:
        """Convert a single PDF to text using best available method"""
        logger.info(f"Processing: {pdf_path.name}")
        
        # Try pdfplumber first (better quality)
        text = self.extract_with_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails or returns empty
        if not text or len(text.strip()) < 100:
            logger.warning(f"pdfplumber failed, trying PyPDF2 for {pdf_path.name}")
            text = self.extract_with_pypdf2(pdf_path)
        
        # Clean the extracted text
        text = self.clean_text(text)
        
        if not text or len(text.strip()) < 50:
            logger.error(f"Failed to extract meaningful text from {pdf_path.name}")
            return None
        
        return text
    
    def save_text(self, text: str, output_path: Path, metadata: dict = None):
        """Save extracted text to file with metadata"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata as comments at the top
                if metadata:
                    f.write(f"# Sections Found: {metadata['total_sections']}\n")
                    f.write(f"# Chapters: {', '.join(metadata['chapters_found'][:5])}\n")
                    f.write(f"# ===== IPC TEXT CONTENT =====\n\n")
                
                f.write(text)
            
            logger.info(f"Saved text to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save text to {output_path}: {e}")
            return False
    
    def convert_all_pdfs(self):
        """Convert all PDFs in input directory to text files"""
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        successful = 0
        failed = 0
        
        for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
            try:
                # Convert PDF to text
                text = self.convert_pdf_to_text(pdf_path)
                
                if text:
                    # Extract metadata
                    metadata = self.extract_ipc_metadata(text)
                    
                    # Save to text file
                    output_path = self.output_dir / f"{pdf_path.stem}.txt"
                    if self.save_text(text, output_path, metadata):
                        successful += 1
                        logger.info(f"✓ Converted: {pdf_path.name} ({metadata['total_sections']} sections)")
                    else:
                        failed += 1
                else:
                    failed += 1
                    logger.error(f"✗ Failed: {pdf_path.name}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {pdf_path.name}: {e}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Conversion Complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"{'='*60}")


def main():
    """Main execution function"""
    # Initialize converter
    converter = PDFToTextConverter(
        input_dir="data/pdfs",
        output_dir="data/texts"
    )
    
    # Convert all PDFs
    converter.convert_all_pdfs()


if __name__ == "__main__":
    main()