#!/usr/bin/env python3
"""
ML Model Training Script for PDF Heading Extraction

This script trains the hybrid classifier using your PDFs and ground truth JSON files.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import fitz

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from src.pdf_parser import PDFParser
from src.layout_extractor import EnhancedLayoutExtractor, TextBlock
from src.heading_detector import HeadingDetector, HeadingCandidate
from src.hybrid_classifier import HybridHeadingClassifier, LightweightMLClassifier
import re
from src.title_identifier import DocumentTitleExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataExtractor:
    """Extracts training data from PDFs and ground truth JSONs"""
    
    def __init__(self):
        self.extractor = EnhancedLayoutExtractor()
        self.detector = HeadingDetector()
        
    def extract_pdf_candidates(self, pdf_path: str) -> List[HeadingCandidate]:
        """Extract all text candidates from PDF"""
        candidates = []
        
        with fitz.open(pdf_path) as doc:
            all_blocks = []
            
            # Extract text blocks from all pages
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks, _ = self.extractor.process_page(page, page_num)
                all_blocks.extend(blocks)
            
            # Detect heading candidates
            candidates = self.detector.detect_candidates(all_blocks)
            
        logger.info(f"Extracted {len(candidates)} candidates from {pdf_path}")
        return candidates
        
    def load_ground_truth(self, json_path: str) -> List[Dict[str, Any]]:
        """Load ground truth headings from JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        headings = data.get('outline', [])
        logger.info(f"Loaded {len(headings)} ground truth headings from {json_path}")
        return headings
        
    def match_candidates_to_ground_truth(self, 
                                       candidates: List[HeadingCandidate],
                                       ground_truth: List[Dict[str, Any]]) -> List[Tuple[HeadingCandidate, bool, bool]]:
        """Match candidates to ground truth labels"""
        training_data = []
        
        # Create set of ground truth texts for fast lookup
        gt_texts = set()
        for gt in ground_truth:
            # Clean text for matching
            clean_text = gt['text'].strip().lower()
            gt_texts.add(clean_text)
        
        # Label each candidate
        page_firsts = {}  # Track first candidate on each page
        
        for candidate in candidates:
            # Check if this is first candidate on page
            page_num = candidate.page_num
            if page_num not in page_firsts:
                page_firsts[page_num] = candidate
            is_first_on_page = (candidate == page_firsts[page_num])
            
            # Clean candidate text for matching
            candidate_text = candidate.text.strip().lower()
            
            # Check if it matches ground truth
            is_heading = False
            for gt_text in gt_texts:
                # Fuzzy matching - check if ground truth is contained in candidate or vice versa
                if (gt_text in candidate_text or 
                    candidate_text in gt_text or 
                    self._fuzzy_match(candidate_text, gt_text)):
                    is_heading = True
                    break
            
            training_data.append((candidate, is_heading, is_first_on_page))
            
        # Log statistics
        positive_count = sum(1 for _, is_heading, _ in training_data if is_heading)
        logger.info(f"Matched {positive_count}/{len(training_data)} candidates as headings")
        
        return training_data
        
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching based on shared words"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= threshold


def train_model_with_data(training_files: List[Tuple[str, str]], 
                         model_save_path: str = "models/heading_classifier.joblib"):
    """Train model with PDF and JSON file pairs"""
    
    # Create models directory
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Initialize training components
    data_extractor = TrainingDataExtractor()
    all_training_data = []
    
    logger.info(f"Training model with {len(training_files)} file pairs...")
    
    # Process each PDF-JSON pair
    for pdf_path, json_path in training_files:
        logger.info(f"Processing {pdf_path} and {json_path}")
        
        try:
            # Extract candidates from PDF
            candidates = data_extractor.extract_pdf_candidates(pdf_path)
            
            # Load ground truth
            ground_truth = data_extractor.load_ground_truth(json_path)
            
            # Match candidates to labels
            training_data = data_extractor.match_candidates_to_ground_truth(
                candidates, ground_truth
            )
            
            all_training_data.extend(training_data)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    if not all_training_data:
        logger.error("No training data extracted!")
        return None
        
    logger.info(f"Total training examples: {len(all_training_data)}")
    
    # Train the ML model
    ml_classifier = LightweightMLClassifier()
    ml_classifier.train(all_training_data)
    
    # Save the trained model
    ml_classifier.save_model(model_save_path)
    
    # Test the hybrid classifier
    hybrid_classifier = HybridHeadingClassifier(ml_model_path=model_save_path)
    
    # Test on a sample
    test_candidates = [item[0] for item in all_training_data[:10]]
    classified = hybrid_classifier.classify(test_candidates)
    
    logger.info(f"Test classification: {len(classified)} headings detected from {len(test_candidates)} candidates")
    
    return model_save_path


def main():
    """Main training function"""
    
    # Define training file pairs (PDF, JSON)
    base_path = Path(__file__).parent
    sample_dir = base_path / "Challenge_1a" / "sample_dataset"
    
    training_files = [
        (
            str(sample_dir / "pdfs" / "file02.pdf"),
            str(sample_dir / "outputs" / "file02.json")
        ),
        (
            str(sample_dir / "pdfs" / "file03.pdf"), 
            str(sample_dir / "outputs" / "file03.json")
        ),
        (
            str(sample_dir / "pdfs" / "file04.pdf"),
            str(sample_dir / "outputs" / "file04.json")  
        ),
        (
            str(sample_dir / "pdfs" / "file01.pdf"),
            str(sample_dir / "outputs" / "file01.json")  
        ),
        (
            str(sample_dir / "pdfs" / "file05.pdf"),
            str(sample_dir / "outputs" / "file05.json")  
        )
    ]
    
    # Verify files exist
    for pdf_path, json_path in training_files:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return
        if not os.path.exists(json_path):
            logger.error(f"JSON file not found: {json_path}")
            return
    
    logger.info("Starting model training...")
    
    # Train the model
    model_path = train_model_with_data(training_files)
    
    if model_path:
        logger.info(f"Model training completed! Saved to: {model_path}")
        logger.info("You can now use this trained model in your pipeline by setting:")
        logger.info(f"  ml_model_path='{model_path}'")
    else:
        logger.error("Model training failed!")


if __name__ == "__main__":
    main()