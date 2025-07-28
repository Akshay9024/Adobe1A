#!/usr/bin/env python3
"""
PDF Heading Extraction Tool
Extracts structured headings (H1/H2/H3) from PDF documents
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
import os

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging with optional file output"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Import pipeline components
try:
    from src.main_pipeline import HeadingExtractionPipeline
except ImportError:
    # Fallback for different directory structures
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from main_pipeline import HeadingExtractionPipeline


class PDFHeadingExtractor:
    """Main class for PDF heading extraction"""
    
    def __init__(self, 
                 use_ml: bool = True,
                 ml_model_path: Optional[str] = None,
                 preserve_trailing_spaces: bool = True,
                 log_level: str = "INFO"):
        self.pipeline = HeadingExtractionPipeline(
            use_multiprocessing=False,  # Disabled in Docker
            use_ml_classifier=use_ml,
            ml_model_path=ml_model_path
        )
        self.preserve_trailing_spaces = preserve_trailing_spaces
        setup_logging(log_level)
        
    def process_single_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single PDF file"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logging.info(f"Processing: {pdf_path}")
        start_time = time.time()
        
        try:
            # Extract headings
            result = self.pipeline.extract_headings(str(pdf_path))
            
            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
                logging.info(f"Output saved to: {output_path}")
            
            elapsed = time.time() - start_time
            logging.info(f"Completed in {elapsed:.2f}s")
            
            # Validate runtime constraint
            if elapsed > 10.0:
                logging.warning(f"Runtime exceeded 10s limit: {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            raise
            
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDFs in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDFs
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {input_dir}")
            return
            
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        success_count = 0
        error_count = 0
        
        for pdf_file in pdf_files:
            try:
                output_file = output_path / f"{pdf_file.stem}.json"
                self.process_single_pdf(str(pdf_file), str(output_file))
                success_count += 1
                
            except Exception as e:
                logging.error(f"Failed to process {pdf_file}: {str(e)}")
                error_count += 1
                continue
                
        logging.info(f"Processing complete: {success_count} successful, {error_count} failed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Extract headings from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF
  python extract_headings.py input.pdf -o output.json
  
  # Process directory
  python extract_headings.py --input-dir ./pdfs --output-dir ./outputs
  
  # Disable ML classifier for faster processing
  python extract_headings.py input.pdf --no-ml
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'pdf_file',
        nargs='?',
        help='Path to input PDF file'
    )
    input_group.add_argument(
        '--input-dir',
        help='Directory containing PDF files'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (for single PDF)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for JSON files (for batch processing)'
    )
    
    # Processing options
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML classifier (faster but potentially less accurate)'
    )
    parser.add_argument(
        '--ml-model',
        help='Path to pre-trained ML model'
    )
    parser.add_argument(
        '--no-preserve-spaces',
        action='store_true',
        help='Remove trailing spaces from headings'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    parser.add_argument(
        '--log-file',
        help='Save logs to file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pdf_file and not args.output:
        # Default output name
        args.output = Path(args.pdf_file).stem + '.json'
        
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir required when using --input-dir")
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create extractor
    extractor = PDFHeadingExtractor(
        use_ml=not args.no_ml,
        ml_model_path=args.ml_model,
        preserve_trailing_spaces=not args.no_preserve_spaces,
        log_level=args.log_level
    )
    
    try:
        if args.pdf_file:
            # Process single file
            result = extractor.process_single_pdf(args.pdf_file, args.output)
            
            # Print to stdout if no output file specified
            if not args.output:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
        else:
            # Process directory
            extractor.process_directory(args.input_dir, args.output_dir)
            
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()