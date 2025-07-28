import json
import time
import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
import re

@dataclass
class HeadingMatch:
    """Represents a heading for comparison"""
    level: str
    text: str
    page: int
    
    def __hash__(self):
        return hash((self.level, self.text.strip(), self.page))
    
    def __eq__(self, other):
        return (self.level == other.level and 
                self.text.strip() == other.text.strip() and 
                self.page == other.page)


@dataclass
class EvaluationMetrics:
    """Evaluation results"""
    precision: float
    recall: float
    f1_score: float
    exact_matches: int
    false_positives: List[HeadingMatch]
    false_negatives: List[HeadingMatch]
    runtime_seconds: float
    
    def __str__(self):
        return (f"Precision: {self.precision:.3f}\n"
                f"Recall: {self.recall:.3f}\n"
                f"F1 Score: {self.f1_score:.3f}\n"
                f"Exact Matches: {self.exact_matches}\n"
                f"Runtime: {self.runtime_seconds:.2f}s")


class Evaluator:
    """Evaluates heading extraction performance"""
    
    def __init__(self):
        self.debug_mode = True
        
    def load_ground_truth(self, json_path: str) -> Tuple[str, List[HeadingMatch]]:
        """Load ground truth from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        title = data['title']
        headings = []
        
        for entry in data['outline']:
            heading = HeadingMatch(
                level=entry['level'],
                text=entry['text'],
                page=entry['page']
            )
            headings.append(heading)
            
        return title, headings
        
    def load_predictions(self, json_path: str) -> Tuple[str, List[HeadingMatch]]:
        """Load predictions from JSON file"""
        return self.load_ground_truth(json_path)  # Same format
        
    def evaluate(self, predictions: List[HeadingMatch], 
                ground_truth: List[HeadingMatch],
                runtime: float) -> EvaluationMetrics:
        """Calculate evaluation metrics"""
        pred_set = set(predictions)
        truth_set = set(ground_truth)
        
        # Calculate metrics
        true_positives = pred_set.intersection(truth_set)
        false_positives = pred_set - truth_set
        false_negatives = truth_set - pred_set
        
        precision = len(true_positives) / len(pred_set) if pred_set else 0
        recall = len(true_positives) / len(truth_set) if truth_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            exact_matches=len(true_positives),
            false_positives=list(false_positives),
            false_negatives=list(false_negatives),
            runtime_seconds=runtime
        )
        
        if self.debug_mode:
            self._log_errors(metrics)
            
        return metrics
        
    def _log_errors(self, metrics: EvaluationMetrics):
        """Log detailed error analysis"""
        if metrics.false_positives:
            logging.info(f"\nFalse Positives ({len(metrics.false_positives)}):")
            for fp in metrics.false_positives[:5]:  # Show first 5
                logging.info(f"  - {fp.level}: '{fp.text[:50]}...' (page {fp.page})")
                
        if metrics.false_negatives:
            logging.info(f"\nFalse Negatives ({len(metrics.false_negatives)}):")
            for fn in metrics.false_negatives[:5]:  # Show first 5
                logging.info(f"  - {fn.level}: '{fn.text[:50]}...' (page {fn.page})")


class RuleTuner:
    """Automatically tunes detection rules based on errors"""
    
    def __init__(self):
        self.error_patterns = defaultdict(list)
        self.tuning_suggestions = []
        
    def analyze_errors(self, metrics: EvaluationMetrics, 
                      all_candidates: List['HeadingCandidate']):
        """Analyze errors to suggest rule adjustments"""
        
        # Analyze false negatives (missed headings)
        for fn in metrics.false_negatives:
            self._analyze_missed_heading(fn, all_candidates)
            
        # Analyze false positives (incorrect detections)
        for fp in metrics.false_positives:
            self._analyze_false_detection(fp, all_candidates)
            
        # Generate tuning suggestions
        self._generate_suggestions()
        
    def _analyze_missed_heading(self, missed: HeadingMatch, 
                               candidates: List['HeadingCandidate']):
        """Analyze why a heading was missed"""
        # Find if it exists in candidates but was filtered
        for candidate in candidates:
            if (candidate.page_num + 1 == missed.page and 
                self._text_matches(candidate.text, missed.text)):
                
                # Found but filtered - analyze why
                reasons = []
                
                if candidate.confidence_score < 40:
                    reasons.append(f"Low confidence: {candidate.confidence_score}")
                    
                if not candidate.is_larger_than_body:
                    reasons.append(f"Font size too small: {candidate.font_size}")
                    
                if candidate.word_count > 15:
                    reasons.append(f"Too many words: {candidate.word_count}")
                    
                if not candidate.has_numbering and not candidate.is_bold:
                    reasons.append("No numbering or bold")
                    
                self.error_patterns['missed'].append({
                    'heading': missed,
                    'candidate': candidate,
                    'reasons': reasons
                })
                break
                
    def _analyze_false_detection(self, false_positive: HeadingMatch, 
                                candidates: List['HeadingCandidate']):
        """Analyze why a non-heading was detected"""
        for candidate in candidates:
            if (candidate.page_num + 1 == false_positive.page and 
                self._text_matches(candidate.text, false_positive.text)):
                
                # Found false positive - analyze characteristics
                characteristics = []
                
                if candidate.is_larger_than_body:
                    characteristics.append(f"Large font: {candidate.font_size}")
                    
                if candidate.is_bold:
                    characteristics.append("Bold text")
                    
                if candidate.word_count < 5:
                    characteristics.append(f"Short text: {candidate.word_count} words")
                    
                self.error_patterns['false_positive'].append({
                    'heading': false_positive,
                    'candidate': candidate,
                    'characteristics': characteristics
                })
                break
                
    def _text_matches(self, text1: str, text2: str) -> bool:
        """Fuzzy text matching"""
        return text1.strip().lower() == text2.strip().lower()
        
    def _generate_suggestions(self):
        """Generate tuning suggestions based on error patterns"""
        suggestions = []
        
        # Analyze missed headings
        if self.error_patterns['missed']:
            font_issues = sum(1 for e in self.error_patterns['missed'] 
                            if 'Font size too small' in str(e['reasons']))
            if font_issues > 2:
                suggestions.append("Lower font size threshold from 1.15x to 1.10x")
                
            confidence_issues = sum(1 for e in self.error_patterns['missed']
                                  if 'Low confidence' in str(e['reasons']))
            if confidence_issues > 2:
                suggestions.append("Lower confidence threshold from 40 to 35")
                
        # Analyze false positives
        if self.error_patterns['false_positive']:
            short_text_issues = sum(1 for e in self.error_patterns['false_positive']
                                  if 'Short text' in str(e['characteristics']))
            if short_text_issues > 2:
                suggestions.append("Increase minimum word count from 1 to 2")
                
        self.tuning_suggestions = suggestions
        
        if suggestions:
            logging.info("\nTuning Suggestions:")
            for suggestion in suggestions:
                logging.info(f"  - {suggestion}")


class MultilingualSupport:
    """Handles multilingual text and numbering systems"""
    
    def __init__(self):
        # Unicode ranges for different scripts
        self.script_ranges = {
            'arabic': (0x0600, 0x06FF),
            'devanagari': (0x0900, 0x097F),
            'chinese': (0x4E00, 0x9FFF),
            'japanese': (0x3040, 0x309F),
            'cyrillic': (0x0400, 0x04FF),
            'hebrew': (0x0590, 0x05FF)
        }
        
        # Numbering patterns for different locales
        self.numbering_patterns = {
            'arabic_indic': re.compile(r'^([٠-٩]+(?:\.[٠-٩]+)*)\s*\.?\s+(.+)$'),
            'devanagari': re.compile(r'^([०-९]+(?:\.[०-९]+)*)\s*\.?\s+(.+)$'),
            'chinese': re.compile(r'^([一二三四五六七八九十]+)\s*[、.]\s*(.+)$'),
            'roman_universal': re.compile(r'^([IVXLCDM]+)\s*\.?\s+(.+)$', re.IGNORECASE)
        }
        
    def detect_script(self, text: str) -> str:
        """Detect the primary script of text"""
        if not text:
            return 'latin'
            
        script_counts = defaultdict(int)
        
        for char in text:
            code_point = ord(char)
            
            # Check each script range
            for script, (start, end) in self.script_ranges.items():
                if start <= code_point <= end:
                    script_counts[script] += 1
                    break
            else:
                if code_point < 0x0080:
                    script_counts['latin'] += 1
                    
        # Return most common script
        if script_counts:
            return max(script_counts.items(), key=lambda x: x[1])[0]
        return 'latin'
        
    def detect_multilingual_numbering(self, text: str) -> Tuple[bool, Optional[str], int]:
        """Detect numbering in various scripts"""
        # Try ASCII numbering first (most common)
        ascii_match = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$', text)
        if ascii_match:
            numbering = ascii_match.group(1)
            depth = numbering.count('.') + 1
            return True, numbering, depth
            
        # Try other numbering systems
        for name, pattern in self.numbering_patterns.items():
            match = pattern.match(text)
            if match:
                numbering = match.group(1)
                # Estimate depth for non-hierarchical systems
                depth = 1 if name in ['chinese', 'roman_universal'] else numbering.count('.') + 1
                return True, numbering, depth
                
        return False, None, 0
        
    def is_titlecase_multilingual(self, text: str, script: str) -> bool:
        """Check if text follows title case conventions for the script"""
        if script in ['chinese', 'japanese', 'arabic', 'hebrew']:
            # These scripts don't have case
            return False
            
        if script == 'latin':
            # Standard title case check
            words = text.split()
            significant_words = [w for w in words if len(w) > 3]
            if not significant_words:
                return False
            uppercase_count = sum(1 for w in significant_words if w[0].isupper())
            return uppercase_count >= len(significant_words) * 0.7
            
        # For other scripts with case (Cyrillic, etc.)
        return text[0].isupper() if text else False


class PerformanceOptimizer:
    """Optimizes performance for large PDFs"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        
    def process_pages_parallel(self, pdf_path: str, 
                             page_numbers: List[int]) -> Dict[int, List['TextBlock']]:
        """Process pages in parallel for faster extraction"""
        # Create chunks of pages
        chunk_size = max(1, len(page_numbers) // self.max_workers)
        chunks = [page_numbers[i:i + chunk_size] 
                 for i in range(0, len(page_numbers), chunk_size)]
        
        # Process chunks in parallel
        with mp.Pool(self.max_workers) as pool:
            process_func = partial(self._process_page_chunk, pdf_path)
            chunk_results = pool.map(process_func, chunks)
            
        # Merge results
        all_blocks = {}
        for chunk_result in chunk_results:
            all_blocks.update(chunk_result)
            
        return all_blocks
        
    def _process_page_chunk(self, pdf_path: str, 
                           page_numbers: List[int]) -> Dict[int, List['TextBlock']]:
        """Process a chunk of pages"""
        import fitz
        from src.layout_extractor import EnhancedLayoutExtractor
        
        blocks_by_page = {}
        
        with fitz.open(pdf_path) as doc:
            extractor = EnhancedLayoutExtractor()
            
            for page_num in page_numbers:
                if page_num < len(doc):
                    page = doc[page_num]
                    blocks, _ = extractor.process_page(page, page_num)
                    blocks_by_page[page_num] = blocks
                    
        return blocks_by_page


# Main evaluation pipeline
class EvaluationPipeline:
    """Complete evaluation and tuning pipeline"""
    
    def __init__(self, use_multiprocessing: bool = True):
        self.evaluator = Evaluator()
        self.tuner = RuleTuner()
        self.multilingual = MultilingualSupport()
        self.optimizer = PerformanceOptimizer() if use_multiprocessing else None
        
    def evaluate_system(self, pdf_path: str, ground_truth_path: str) -> EvaluationMetrics:
        """Evaluate the complete system on a PDF"""
        start_time = time.time()
        
        # Extract headings (with timing)
        predictions = self._extract_headings(pdf_path)
        
        runtime = time.time() - start_time
        
        # Load ground truth
        _, truth_headings = self.evaluator.load_ground_truth(ground_truth_path)
        
        # Convert predictions to HeadingMatch format
        pred_headings = []
        for entry in predictions['outline']:
            pred_headings.append(HeadingMatch(
                level=entry['level'],
                text=entry['text'],
                page=entry['page']
            ))
            
        # Evaluate
        metrics = self.evaluator.evaluate(pred_headings, truth_headings, runtime)
        
        # Log results
        logging.info(f"\nEvaluation Results for {pdf_path}:")
        logging.info(str(metrics))
        
        # Check runtime constraint
        if runtime > 10.0:
            logging.warning(f"Runtime exceeds 10s limit: {runtime:.2f}s")
            
        return metrics
        
    def _extract_headings(self, pdf_path: str) -> Dict:
        """Extract headings using the complete pipeline"""
        # This would call your complete extraction pipeline
        # Placeholder for integration
        return {"title": "Test", "outline": []}