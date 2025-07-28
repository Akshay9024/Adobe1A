import fitz
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HeadingExtractionPipeline:
    """Complete pipeline for PDF heading extraction"""
    
    def __init__(self, 
                 use_multiprocessing: bool = False,
                 use_ml_classifier: bool = True,
                 ml_model_path: Optional[str] = "models/heading_classifier.joblib"):
        self.use_multiprocessing = use_multiprocessing
        self.use_ml_classifier = use_ml_classifier
        self.ml_model_path = ml_model_path
        
        # Import all components
        from src.pdf_parser import PDFParser
        from src.layout_extractor import EnhancedLayoutExtractor
        from src.title_identifier import DocumentTitleExtractor
        from src.heading_detector import HeadingDetector
        from src.heading_classifier import HeadingLevelClassifier
        from src.hybrid_classifier import HybridHeadingClassifier
        from src.complex_layout_handler import ComplexLayoutProcessor
        from src.json_postprocessor import PostProcessor
        from src.evaluation_system import MultilingualSupport, PerformanceOptimizer
        
        # Initialize components
        self.multilingual = MultilingualSupport()
        self.optimizer = PerformanceOptimizer() if use_multiprocessing else None
        
    def extract_headings(self, pdf_path: str) -> Dict:
        """Extract headings from PDF file"""
        start_time = time.time()
        
        try:
            # Store pdf_path for use in table detection
            self.pdf_path = pdf_path
            
            with fitz.open(pdf_path) as doc:
                logging.info(f"Processing PDF: {pdf_path}")
                logging.info(f"Pages: {len(doc)}")
                
                # Step 1: Parse PDF and extract text blocks
                all_blocks = self._extract_blocks(doc)
                
                # Step 2: Extract document title
                title = self._extract_title(doc, all_blocks)
                
                # Step 3: Detect heading candidates
                candidates = self._detect_candidates(all_blocks)
                
                # Step 4: Handle complex layouts
                candidates = self._handle_complex_layouts(doc, all_blocks, candidates)
                
                # Step 5: Apply ML classification if enabled
                if self.use_ml_classifier:
                    candidates = self._apply_ml_classification(candidates)
                
                # Step 6: Classify heading levels
                candidates = self._classify_levels(candidates)
                
                # Step 7: Generate JSON output
                json_output = self._generate_output(title, candidates)
                
                runtime = time.time() - start_time
                logging.info(f"Extraction completed in {runtime:.2f}s")
                
                return json.loads(json_output)
                
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise
            
    def _extract_blocks(self, doc) -> List:
        """Extract text blocks from all pages"""
        from src.pdf_parser import PDFParser
        from src.layout_extractor import EnhancedLayoutExtractor
        
        all_blocks = []
        
        if self.use_multiprocessing and len(doc) > 10:
            # Use multiprocessing for large documents
            blocks_by_page = self.optimizer.process_pages_parallel(
                doc.name, list(range(len(doc)))
            )
            for page_num in sorted(blocks_by_page.keys()):
                all_blocks.extend(blocks_by_page[page_num])
        else:
            # Sequential processing
            extractor = EnhancedLayoutExtractor()
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks, _ = extractor.process_page(page, page_num)
                all_blocks.extend(blocks)
                
        logging.info(f"Extracted {len(all_blocks)} text blocks")
        return all_blocks
        
    def _extract_title(self, doc, blocks) -> str:
        """Extract document title"""
        from src.title_identifier import DocumentTitleExtractor
        
        extractor = DocumentTitleExtractor()
        title = extractor.extract_title(doc.name, blocks)
        logging.info(f"Extracted title: '{title}'")
        return title
        
    def _detect_candidates(self, blocks) -> List:
        """Detect heading candidates"""
        from src.heading_detector import HeadingDetector
        
        detector = HeadingDetector()
        candidates = detector.detect_candidates(blocks, self.pdf_path)
        
        # Apply multilingual enhancements
        for candidate in candidates:
            script = self.multilingual.detect_script(candidate.text)
            
            # Check multilingual numbering
            has_num, pattern, depth = self.multilingual.detect_multilingual_numbering(
                candidate.text
            )
            if has_num and not candidate.has_numbering:
                candidate.has_numbering = True
                candidate.numbering_pattern = pattern
                candidate.numbering_depth = depth
                candidate.confidence_score += 20
                
        logging.info(f"Detected {len(candidates)} heading candidates")
        return candidates
        
    def _handle_complex_layouts(self, doc, blocks, candidates) -> List:
        """Handle complex PDF layouts"""
        from src.complex_layout_handler import ComplexLayoutProcessor
        
        processor = ComplexLayoutProcessor()
        candidates = processor.process_complex_document(doc, blocks, candidates)
        return candidates
        
    def _apply_ml_classification(self, candidates) -> List:
        """Apply ML-based classification"""
        from src.hybrid_classifier import HybridHeadingClassifier
        
        classifier = HybridHeadingClassifier(ml_model_path=self.ml_model_path)
        candidates = classifier.classify(candidates)
        return candidates
        
    def _classify_levels(self, candidates) -> List:
        """Classify heading levels (H1/H2/H3)"""
        from src.heading_classifier import HeadingLevelClassifier
        
        classifier = HeadingLevelClassifier()
        candidates = classifier.classify_heading_levels(candidates)
        return candidates
        
    def _generate_output(self, title: str, candidates) -> str:
        """Generate JSON output"""
        from src.json_postprocessor import PostProcessor
        
        processor = PostProcessor()
        json_output = processor.process(title, candidates)
        return json_output
        
    def process_batch(self, pdf_dir: str, output_dir: str):
        """Process multiple PDFs in a directory"""
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logging.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                # Extract headings
                result = self.extract_headings(str(pdf_path))
                
                # Save output
                output_path = output_dir / f"{pdf_path.stem}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
                logging.info(f"Saved output to {output_path}")
                
            except Exception as e:
                logging.error(f"Failed to process {pdf_path}: {e}")


def main():
    """Main entry point for Docker container"""
    # Configure for Docker environment
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Check directories exist
    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = HeadingExtractionPipeline(
        use_multiprocessing=False,  # Disable for Docker
        use_ml_classifier=True
    )
    
    # Process all PDFs
    pipeline.process_batch(input_dir, output_dir)
    
    logging.info("Processing complete")


if __name__ == "__main__":
    main()