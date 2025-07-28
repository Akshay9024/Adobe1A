import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import re
from sklearn.cluster import DBSCAN

@dataclass
class TOCEntry:
    """Represents a Table of Contents entry from PDF bookmarks"""
    level: int  # 1, 2, 3...
    title: str
    page: int  # 0-indexed
    
    def to_heading_candidate(self) -> 'HeadingCandidate':
        """Convert TOC entry to heading candidate format"""
        # Create a synthetic heading candidate
        from src.heading_detector import HeadingCandidate
        from src.layout_extractor import TextBlock
        
        # Create minimal text block
        block = TextBlock(
            text=self.title,
            bbox=(0, 0, 0, 0),  # No position info from TOC
            font_name="",
            font_size=0,
            font_flags=0,
            block_type="toc",
            block_no=0,
            page_num=self.page
        )
        
        candidate = HeadingCandidate(
            text=self.title,
            block=block,
            page_num=self.page,
            font_size=0,
            confidence_score=90.0  # High confidence for TOC entries
        )
        
        # Set heading level based on TOC level
        candidate.heading_level = min(self.level, 3)  # Cap at H3
        
        return candidate


class ColumnDetector:
    """Advanced column detection and handling"""
    
    def __init__(self, min_column_gap: float = 30.0):
        self.min_column_gap = min_column_gap
        
    def detect_columns_advanced(self, blocks: List['TextBlock']) -> Dict[int, List['TextBlock']]:
        """Detect columns using clustering and gap analysis"""
        if not blocks:
            return {0: blocks}
            
        # Extract x-coordinates
        x_coords = np.array([[b.x0, b.x1] for b in blocks])
        
        # Find potential column boundaries
        column_ranges = self._find_column_ranges(x_coords)
        
        if len(column_ranges) == 1:
            # Single column
            return {0: blocks}
            
        # Assign blocks to columns
        columns = defaultdict(list)
        for block in blocks:
            col_id = self._assign_to_column(block, column_ranges)
            columns[col_id].append(block)
            
        return dict(columns)
        
    def _find_column_ranges(self, x_coords: np.ndarray) -> List[Tuple[float, float]]:
        """Find distinct column ranges based on x-coordinates"""
        # Flatten and sort x-coordinates
        all_x = np.sort(x_coords.flatten())
        
        # Find gaps between text regions
        gaps = []
        for i in range(1, len(all_x)):
            gap_size = all_x[i] - all_x[i-1]
            if gap_size > self.min_column_gap:
                gaps.append((all_x[i-1], all_x[i], gap_size))
                
        if not gaps:
            # No significant gaps, single column
            return [(all_x.min(), all_x.max())]
            
        # Find the most significant gap(s)
        gaps.sort(key=lambda x: x[2], reverse=True)
        
        # For 2-column layout, use the largest gap
        if len(gaps) >= 1:
            main_gap = gaps[0]
            return [(all_x.min(), main_gap[0]), (main_gap[1], all_x.max())]
            
        return [(all_x.min(), all_x.max())]
        
    def _assign_to_column(self, block: 'TextBlock', 
                         column_ranges: List[Tuple[float, float]]) -> int:
        """Assign a block to a column based on its position"""
        block_center = (block.x0 + block.x1) / 2
        
        for col_id, (x_min, x_max) in enumerate(column_ranges):
            if x_min <= block_center <= x_max:
                return col_id
                
        # Default to first column if no match
        return 0
        
    def sort_multi_column_blocks(self, blocks: List['TextBlock']) -> List['TextBlock']:
        """Sort blocks respecting multi-column layout"""
        columns = self.detect_columns_advanced(blocks)
        
        if len(columns) == 1:
            # Single column, simple y-sort
            return sorted(blocks, key=lambda b: b.y0)
            
        # Multi-column: sort within columns, then concatenate
        sorted_blocks = []
        
        for col_id in sorted(columns.keys()):
            # Sort blocks within column by y-position
            col_blocks = sorted(columns[col_id], key=lambda b: b.y0)
            sorted_blocks.extend(col_blocks)
            
        return sorted_blocks


class HeaderFooterDetector:
    """Detects and filters headers/footers"""
    
    def __init__(self, position_threshold: float = 50.0,
                 repetition_threshold: float = 0.8):
        self.position_threshold = position_threshold
        self.repetition_threshold = repetition_threshold
        self.header_patterns = []
        self.footer_patterns = []
        
    def detect_headers_footers(self, pages_blocks: Dict[int, List['TextBlock']]) -> Set[str]:
        """Detect repeated headers/footers across pages"""
        if len(pages_blocks) < 3:
            return set()  # Need multiple pages to detect patterns
            
        # Collect potential headers/footers by position
        top_texts = []
        bottom_texts = []
        
        page_heights = {}
        
        for page_num, blocks in pages_blocks.items():
            if not blocks:
                continue
                
            # Get page height from blocks
            page_height = max(b.y1 for b in blocks)
            page_heights[page_num] = page_height
            
            # Find top and bottom blocks
            sorted_blocks = sorted(blocks, key=lambda b: b.y0)
            
            # Top blocks (potential headers)
            for block in sorted_blocks[:3]:
                if block.y0 < self.position_threshold:
                    top_texts.append(block.text.strip())
                    
            # Bottom blocks (potential footers)
            for block in sorted_blocks[-3:]:
                if block.y1 > page_height - self.position_threshold:
                    bottom_texts.append(block.text.strip())
                    
        # Find repeated patterns
        repeated_texts = set()
        
        # Check headers
        header_counter = Counter(top_texts)
        for text, count in header_counter.items():
            if count >= len(pages_blocks) * self.repetition_threshold:
                repeated_texts.add(text)
                self.header_patterns.append(text)
                
        # Check footers
        footer_counter = Counter(bottom_texts)
        for text, count in footer_counter.items():
            if count >= len(pages_blocks) * self.repetition_threshold:
                repeated_texts.add(text)
                self.footer_patterns.append(text)
                
        # Also check for page number patterns
        page_num_pattern = re.compile(r'^\s*\d+\s*$|^page\s+\d+|^\d+\s*/\s*\d+', re.IGNORECASE)
        
        for blocks in pages_blocks.values():
            for block in blocks:
                if page_num_pattern.match(block.text.strip()):
                    repeated_texts.add(block.text.strip())
                    
        return repeated_texts
        
    def is_header_footer(self, block: 'TextBlock', page_height: float) -> bool:
        """Check if a block is likely a header or footer"""
        text = block.text.strip()
        
        # Check position
        is_top = block.y0 < self.position_threshold
        is_bottom = block.y1 > page_height - self.position_threshold
        
        if not (is_top or is_bottom):
            return False
            
        # Check against known patterns
        if text in self.header_patterns or text in self.footer_patterns:
            return True
            
        # Check for page numbers
        if re.match(r'^\s*\d+\s*$|^page\s+\d+|^\d+\s*/\s*\d+', text, re.IGNORECASE):
            return True
            
        return False


class TOCExtractor:
    """Extracts and processes PDF bookmarks/TOC"""
    
    def __init__(self):
        self.toc_entries = []
        
    def extract_toc(self, doc) -> List[TOCEntry]:
        """Extract table of contents from PDF bookmarks"""
        try:
            toc = doc.get_toc()
            
            for entry in toc:
                if len(entry) >= 3:
                    level, title, page = entry[0], entry[1], entry[2] - 1  # Convert to 0-indexed
                    
                    # Clean title
                    title = title.strip()
                    if title:
                        self.toc_entries.append(TOCEntry(level, title, page))
                        
            logging.info(f"Extracted {len(self.toc_entries)} TOC entries")
            return self.toc_entries
            
        except Exception as e:
            logging.warning(f"Failed to extract TOC: {e}")
            return []
            
    def merge_with_detected_headings(self, 
                                   detected_candidates: List['HeadingCandidate'],
                                   toc_entries: List[TOCEntry]) -> List['HeadingCandidate']:
        """Merge TOC entries with detected headings"""
        if not toc_entries:
            return detected_candidates
            
        # Convert TOC entries to candidates
        toc_candidates = [entry.to_heading_candidate() for entry in toc_entries]
        
        # Create a map of detected headings by page and approximate text
        detected_map = defaultdict(list)
        for candidate in detected_candidates:
            key = (candidate.page_num, self._normalize_for_matching(candidate.text))
            detected_map[key].append(candidate)
            
        # Merge or add TOC entries
        merged = list(detected_candidates)  # Start with detected
        
        for toc_candidate in toc_candidates:
            key = (toc_candidate.page_num, self._normalize_for_matching(toc_candidate.text))
            
            if key in detected_map:
                # Already detected, possibly update level
                for detected in detected_map[key]:
                    if detected.heading_level != toc_candidate.heading_level:
                        logging.debug(f"TOC level override: '{detected.text}' "
                                    f"H{detected.heading_level} -> H{toc_candidate.heading_level}")
                        detected.heading_level = toc_candidate.heading_level
            else:
                # Not detected, add from TOC
                logging.debug(f"Adding from TOC: '{toc_candidate.text}'")
                merged.append(toc_candidate)
                
        return merged
        
    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        # Remove extra whitespace, punctuation, and lowercase
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.lower().split())
        return text


class OCRTextHandler:
    """Special handling for OCR-extracted text"""
    
    def __init__(self):
        self.ocr_confidence_threshold = 60  # Tesseract confidence
        
    def process_ocr_candidates(self, candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
        """Apply special rules for OCR-extracted candidates"""
        ocr_candidates = [c for c in candidates if hasattr(c.block, 'source') and c.block.source == 'ocr']
        
        if not ocr_candidates:
            return candidates
            
        logging.info(f"Processing {len(ocr_candidates)} OCR candidates")
        
        for candidate in ocr_candidates:
            # OCR text lacks font info, rely more on other features
            
            # Boost confidence if has strong numbering
            if candidate.has_numbering:
                candidate.confidence_score += 20
                
            # Check for all-caps (strong heading indicator in OCR)
            if candidate.is_uppercase and candidate.word_count <= 10:
                candidate.confidence_score += 15
                
            # Position-based boost (top of page more likely heading)
            if hasattr(candidate.block, 'y0'):
                relative_pos = candidate.block.y0 / 792  # Assume letter size
                if relative_pos < 0.2:  # Top 20% of page
                    candidate.confidence_score += 10
                    
        return candidates


class ComplexLayoutProcessor:
    """Main processor for complex layouts"""
    
    def __init__(self):
        self.column_detector = ColumnDetector()
        self.header_footer_detector = HeaderFooterDetector()
        self.toc_extractor = TOCExtractor()
        self.ocr_handler = OCRTextHandler()
        
    def process_complex_document(self, 
                               doc,
                               all_blocks: List['TextBlock'],
                               candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
        """Process document with complex layout handling"""
        
        # Group blocks by page
        blocks_by_page = defaultdict(list)
        for block in all_blocks:
            blocks_by_page[block.page_num].append(block)
            
        # 1. Detect and filter headers/footers
        repeated_texts = self.header_footer_detector.detect_headers_footers(blocks_by_page)
        
        # Filter candidates
        filtered_candidates = []
        for candidate in candidates:
            if candidate.text.strip() not in repeated_texts:
                filtered_candidates.append(candidate)
            else:
                logging.debug(f"Filtered header/footer: '{candidate.text}'")
                
        # 2. Handle multi-column layouts
        for page_num, page_blocks in blocks_by_page.items():
            # Sort blocks considering columns
            sorted_blocks = self.column_detector.sort_multi_column_blocks(page_blocks)
            
            # Update reading order in candidates
            for i, block in enumerate(sorted_blocks):
                for candidate in filtered_candidates:
                    if candidate.block == block:
                        candidate.reading_order = i
                        
        # 3. Extract and merge TOC
        toc_entries = self.toc_extractor.extract_toc(doc)
        if toc_entries:
            filtered_candidates = self.toc_extractor.merge_with_detected_headings(
                filtered_candidates, toc_entries
            )
            
        # 4. Handle OCR text specially
        filtered_candidates = self.ocr_handler.process_ocr_candidates(filtered_candidates)
        
        # 5. Final sorting by page and reading order
        filtered_candidates.sort(
            key=lambda c: (c.page_num, getattr(c, 'reading_order', c.block.y0))
        )
        
        logging.info(f"Complex layout processing complete: "
                    f"{len(candidates)} -> {len(filtered_candidates)} candidates")
        
        return filtered_candidates


# Integration function
def handle_complex_layouts(doc, blocks: List['TextBlock'], 
                         candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
    """Main function to handle complex PDF layouts"""
    processor = ComplexLayoutProcessor()
    return processor.process_complex_document(doc, blocks, candidates)