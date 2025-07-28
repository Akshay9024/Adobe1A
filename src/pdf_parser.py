import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LTFigure
import pytesseract
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class PageContent:
    """Stores extracted content from a PDF page"""
    page_num: int  # 0-indexed
    text_blocks: List[Dict[str, Any]]
    has_text: bool
    used_ocr: bool
    
class PDFParser:
    """Multi-library PDF parser with OCR fallback"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_pymupdf = None
        self.doc_pdfplumber = None
        self.metadata = {}
        self.pages_content = []
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open PDF with multiple libraries"""
        try:
            # PyMuPDF - best for text extraction and OCR
            self.doc_pymupdf = fitz.open(self.pdf_path)
            
            # Extract metadata
            self.metadata = self.doc_pymupdf.metadata
            
            # PDFPlumber - excellent for table/structure detection
            self.doc_pdfplumber = pdfplumber.open(self.pdf_path)
            
        except Exception as e:
            logging.error(f"Error opening PDF: {e}")
            raise
            
    def close(self):
        """Close all document handles"""
        if self.doc_pymupdf:
            self.doc_pymupdf.close()
        if self.doc_pdfplumber:
            self.doc_pdfplumber.close()
            
    def extract_title_from_metadata(self) -> Optional[str]:
        """Extract title from PDF metadata"""
        # Check multiple metadata fields
        title_candidates = [
            self.metadata.get('title', ''),
            self.metadata.get('Title', ''),
            self.metadata.get('/Title', '')
        ]
        
        for title in title_candidates:
            if title and title.strip():
                return title.strip()
        return None
        
    def check_page_has_text(self, page_num: int) -> bool:
        """Quick check if page has extractable text"""
        # Use PyMuPDF for fast text detection
        page = self.doc_pymupdf[page_num]
        text = page.get_text().strip()
        
        # Also check with pdfplumber for verification
        plumber_page = self.doc_pdfplumber.pages[page_num]
        plumber_text = plumber_page.extract_text() or ""
        
        return bool(text or plumber_text.strip())
        
    def extract_text_with_position(self, page_num: int) -> List[Dict[str, Any]]:
        """Extract text blocks with position info using multiple libraries"""
        text_blocks = []
        
        # 1. PyMuPDF extraction with position
        page = self.doc_pymupdf[page_num]
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_blocks.append({
                            'text': span.get("text", ""),
                            'bbox': span.get("bbox", []),
                            'font': span.get("font", ""),
                            'size': span.get("size", 0),
                            'flags': span.get("flags", 0),
                            'source': 'pymupdf'
                        })
        
        # 2. PDFPlumber extraction for additional structure info
        plumber_page = self.doc_pdfplumber.pages[page_num]
        chars = plumber_page.chars
        
        # Group chars into words/lines for better structure detection
        if chars:
            current_line = []
            current_y = None
            
            for char in sorted(chars, key=lambda x: (x['top'], x['x0'])):
                if current_y is None or abs(char['top'] - current_y) < 2:
                    current_line.append(char)
                    current_y = char['top']
                else:
                    if current_line:
                        text = ''.join(c['text'] for c in current_line)
                        text_blocks.append({
                            'text': text,
                            'bbox': [
                                min(c['x0'] for c in current_line),
                                min(c['top'] for c in current_line),
                                max(c['x1'] for c in current_line),
                                max(c['bottom'] for c in current_line)
                            ],
                            'font': current_line[0].get('fontname', ''),
                            'size': current_line[0].get('size', 0),
                            'source': 'pdfplumber'
                        })
                    current_line = [char]
                    current_y = char['top']
                    
        # 3. PDFMiner for layout analysis
        try:
            for page_layout in extract_pages(self.pdf_path, page_numbers=[page_num]):
                for element in page_layout:
                    if isinstance(element, (LTTextBox, LTTextLine)):
                        text_blocks.append({
                            'text': element.get_text().strip(),
                            'bbox': element.bbox,
                            'source': 'pdfminer'
                        })
        except:
            pass  # PDFMiner can be fragile, continue if it fails
            
        return text_blocks
        
    def perform_ocr(self, page_num: int) -> List[Dict[str, Any]]:
        """Perform OCR on page when text extraction fails"""
        logging.info(f"Performing OCR on page {page_num}")
        
        page = self.doc_pymupdf[page_num]
        
        # Render page to image at high DPI for better OCR
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        
        # Perform OCR with position data
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        text_blocks = []
        n_boxes = len(ocr_data['text'])
        
        # Scale factor to convert from image coords to PDF coords
        scale_x = page.rect.width / img.width
        scale_y = page.rect.height / img.height
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                # Convert image coordinates to PDF coordinates
                x = ocr_data['left'][i] * scale_x
                y = ocr_data['top'][i] * scale_y
                w = ocr_data['width'][i] * scale_x
                h = ocr_data['height'][i] * scale_y
                
                text_blocks.append({
                    'text': text,
                    'bbox': [x, y, x + w, y + h],
                    'conf': ocr_data['conf'][i],
                    'source': 'ocr'
                })
                
        return text_blocks
        
    def process_page(self, page_num: int) -> PageContent:
        """Process a single page with fallback to OCR if needed"""
        has_text = self.check_page_has_text(page_num)
        
        if has_text:
            # Extract text normally
            text_blocks = self.extract_text_with_position(page_num)
            used_ocr = False
        else:
            # Fallback to OCR
            text_blocks = self.perform_ocr(page_num)
            used_ocr = True
            
        return PageContent(
            page_num=page_num,
            text_blocks=text_blocks,
            has_text=has_text,
            used_ocr=used_ocr
        )
        
    def process_all_pages(self) -> List[PageContent]:
        """Process all pages in the PDF"""
        total_pages = len(self.doc_pymupdf)
        
        for page_num in range(total_pages):
            logging.info(f"Processing page {page_num + 1}/{total_pages}")
            page_content = self.process_page(page_num)
            self.pages_content.append(page_content)
            
        return self.pages_content
        
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed document"""
        return {
            'total_pages': len(self.pages_content),
            'pages_with_text': sum(1 for p in self.pages_content if p.has_text),
            'pages_with_ocr': sum(1 for p in self.pages_content if p.used_ocr),
            'metadata_title': self.extract_title_from_metadata()
        }


# Usage example
def parse_pdf(pdf_path: str) -> Tuple[List[PageContent], Dict[str, Any]]:
    """Main function to parse PDF"""
    with PDFParser(pdf_path) as parser:
        pages = parser.process_all_pages()
        stats = parser.get_document_stats()
        
        logging.info(f"Document stats: {stats}")
        
        return pages, stats