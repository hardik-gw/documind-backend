import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the processor with smart splitting parameters.
        - chunk_size: Target characters per chunk.
        - chunk_overlap: Overlap to maintain context between split boundaries.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""] 
            # Splits paragraphs -> sentences -> words dynamically
        )

    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads a PDF and extracts text page by page to preserve metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        pages_data = []
        
        # Open the PDF using PyMuPDF
        with fitz.open(file_path) as doc:
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                text = page.get_text("text").strip()
                
                if text: # Skip blank pages
                    pages_data.append({
                        "page_num": page_num,
                        "text": text
                    })
                    
        return pages_data

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page and splits it into semantic chunks
        while perfectly retaining the source page metadata for citations.
        """
        filename = os.path.basename(file_path)
        pages_data = self.extract_text_by_page(file_path)
        final_chunks = []

        for page in pages_data:
            page_num = page["page_num"]
            raw_text = page["text"]

            # Split the text of this specific page
            chunks_of_page = self.text_splitter.split_text(raw_text)

            # Package each chunk with its critical citation metadata
            for chunk in chunks_of_page:
                final_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_num
                    }
                })

        return final_chunks

# ==========================================
# TEST DRIVE RUNNER
# ==========================================
if __name__ == "__main__":
    TEST_PDF = "sample.pdf" 
    
    if not os.path.exists(TEST_PDF):
        print(f"⚠️ To test this script, please drop a PDF named '{TEST_PDF}' into your root folder.")
    else:
        print("🚀 Processing PDF...")
        processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
        processed_chunks = processor.process_pdf(TEST_PDF)
        
        print(f"✅ Generated {len(processed_chunks)} chunks!\n")
        
        print("--- CHUNK SAMPLE VIEW ---")
        for i, chunk in enumerate(processed_chunks[:2]):
            print(f"Chunk #{i + 1}")
            print(f"Source: {chunk['metadata']['source']} | Page: {chunk['metadata']['page']}")
            print(f"Content Preview: {chunk['text'][:150]}...")
            print("-" * 40)