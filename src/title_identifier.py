import re
from typing import Optional, List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class TitleCandidate:
    """Represents a potential title with scoring metrics"""
    text: str
    source: str  # 'metadata' or 'content'
    confidence: float
    font_size: float = 0
    position_y: float = 0
    is_centered: bool = False
    is_uppercase: bool = False
    word_count: int = 0
    page_num: int = 0
    
    def calculate_score(self, page_height: float) -> float:
        """Calculate title likelihood score"""
        score = 0.0
        
        # Source bonus
        if self.source == 'metadata':
            score += 50  # Strong preference for metadata
            
        # Font size score (normalized)
        if self.font_size > 0:
            score += min(self.font_size / 2, 30)  # Cap at 30 points
            
        # Position score (prefer top of page)
        if self.position_y > 0 and page_height > 0:
            relative_position = self.position_y / page_height
            if relative_position < 0.3:  # Top 30% of page
                score += (1 - relative_position) * 20
                
        # Centering bonus
        if self.is_centered:
            score += 15
            
        # Length penalty (titles are usually concise)
        if 1 <= self.word_count <= 10:
            score += 10
        elif self.word_count > 15:
            score -= (self.word_count - 15) * 2
            
        # Uppercase bonus (many titles are uppercase)
        if self.is_uppercase and self.word_count > 1:
            score += 5
            
        return score * self.confidence


class TitleIdentifier:
    """Identifies document title using multiple strategies"""
    
    def __init__(self):
        self.metadata_fields = ['title', 'Title', '/Title', 'dc:title', 'Subject']
        self.noise_patterns = [
            r'^untitled\d*$',
            r'^document\d*$',
            r'^microsoft word',
            r'^\.pdf$',
            r'^\d+$',  # Just numbers
            r'^page \d+$'
        ]
        
    def clean_title(self, title: str) -> str:
        """Clean title while preserving case and important formatting"""
        if not title:
            return ""
            
        # Strip whitespace and control characters
        title = title.strip()
        title = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', title)
        
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title)
        
        # Remove trailing dots or commas
        title = title.rstrip('.,')
        
        return title
        
    def is_valid_title(self, text: str) -> bool:
        """Check if text is a valid title candidate"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Check against noise patterns
        text_lower = text.lower().strip()
        for pattern in self.noise_patterns:
            if re.match(pattern, text_lower):
                return False
                
        # Must contain at least one letter
        if not any(c.isalpha() for c in text):
            return False
            
        return True
        
    def extract_metadata_title(self, metadata: Dict) -> Optional[TitleCandidate]:
        """Extract title from PDF metadata"""
        for field in self.metadata_fields:
            if field in metadata and metadata[field]:
                title_text = str(metadata[field]).strip()
                
                if self.is_valid_title(title_text):
                    cleaned = self.clean_title(title_text)
                    if cleaned:
                        return TitleCandidate(
                            text=cleaned,
                            source='metadata',
                            confidence=0.95,
                            word_count=len(cleaned.split())
                        )
                        
        return None
        
    def analyze_text_block_as_title(self, block, page_width: float) -> TitleCandidate:
        """Analyze a text block to determine if it's a title"""
        text = self.clean_title(block.text)
        
        if not self.is_valid_title(text):
            return None
            
        # Calculate centering
        block_center = (block.x0 + block.x1) / 2
        page_center = page_width / 2
        center_deviation = abs(block_center - page_center)
        is_centered = center_deviation < page_width * 0.15  # Within 15% of center
        
        # Check if uppercase
        is_uppercase = text.isupper() and len(text) > 3
        
        # Calculate confidence based on multiple factors
        confidence = 1.0
        
        # Reduce confidence for very long text
        word_count = len(text.split())
        if word_count > 20:
            confidence *= 0.7
        elif word_count > 30:
            confidence *= 0.5
            
        # Reduce confidence if not at top of page
        if block.y0 > 200:  # More than ~2.8 inches from top
            confidence *= 0.8
            
        return TitleCandidate(
            text=text,
            source='content',
            confidence=confidence,
            font_size=block.font_size,
            position_y=block.y0,
            is_centered=is_centered,
            is_uppercase=is_uppercase,
            word_count=word_count,
            page_num=block.page_num
        )
        
    def find_title_from_first_page(self, blocks: List, page_width: float, 
                                  page_height: float) -> Optional[TitleCandidate]:
        """Find title from first page content"""
        if not blocks:
            return None
            
        # Filter blocks from first page (page_num = 0)
        first_page_blocks = [b for b in blocks if b.page_num == 0]
        
        if not first_page_blocks:
            return None
            
        # Find blocks in top portion of page
        top_threshold = page_height * 0.4  # Top 40% of page
        top_blocks = [b for b in first_page_blocks if b.y0 < top_threshold]
        
        if not top_blocks:
            top_blocks = first_page_blocks[:10]  # Fallback to first 10 blocks
            
        # Generate candidates
        candidates = []
        for block in top_blocks:
            candidate = self.analyze_text_block_as_title(block, page_width)
            if candidate:
                candidates.append(candidate)
                
        if not candidates:
            return None
            
        # Score and rank candidates
        scored_candidates = [
            (candidate, candidate.calculate_score(page_height))
            for candidate in candidates
        ]
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Log top candidates for debugging
        for i, (candidate, score) in enumerate(scored_candidates[:3]):
            logging.debug(f"Title candidate {i+1}: '{candidate.text}' (score: {score:.2f})")
            
        return scored_candidates[0][0] if scored_candidates else None
        
    def identify_title(self, metadata: Dict, blocks: List, 
                      page_dimensions: Tuple[float, float]) -> str:
        """Main method to identify document title"""
        page_width, page_height = page_dimensions
        
        # Strategy 1: Try metadata first
        metadata_candidate = self.extract_metadata_title(metadata)
        
        # Strategy 2: Extract from content
        content_candidate = self.find_title_from_first_page(
            blocks, page_width, page_height
        )
        
        # Choose best candidate
        candidates = [c for c in [metadata_candidate, content_candidate] if c]
        
        if not candidates:
            # Fallback: Use first significant text block
            for block in blocks:
                if block.page_num == 0 and len(block.text.strip()) > 5:
                    return self.clean_title(block.text)
            return "Untitled Document"
            
        # If we have both, prefer metadata unless content is significantly better
        if len(candidates) == 2:
            meta_score = metadata_candidate.calculate_score(page_height)
            content_score = content_candidate.calculate_score(page_height)
            
            # Only override metadata if content is significantly better
            if content_score > meta_score * 1.5:
                return content_candidate.text
                
        return candidates[0].text


# Integration helper
class DocumentTitleExtractor:
    """High-level title extraction interface"""
    
    def __init__(self):
        self.identifier = TitleIdentifier()
        
    def extract_title(self, pdf_path: str, blocks: Optional[List] = None) -> str:
        """Extract title from PDF file"""
        import fitz
        
        try:
            with fitz.open(pdf_path) as doc:
                # Get metadata
                metadata = doc.metadata or {}
                
                # Get first page dimensions
                if len(doc) > 0:
                    first_page = doc[0]
                    page_rect = first_page.rect
                    page_dimensions = (page_rect.width, page_rect.height)
                    
                    # If blocks not provided, extract them
                    if blocks is None:
                        from src.layout_extractor import EnhancedLayoutExtractor
                        extractor = EnhancedLayoutExtractor()
                        blocks, _ = extractor.process_page(first_page, 0)
                else:
                    page_dimensions = (612, 792)  # Default US Letter
                    
                # Identify title
                title = self.identifier.identify_title(
                    metadata, blocks or [], page_dimensions
                )
                
                logging.info(f"Identified title: '{title}'")
                return title
                
        except Exception as e:
            logging.error(f"Error extracting title: {e}")
            return "Untitled Document"


# Example usage
def get_document_title(pdf_path: str, page_blocks: Optional[List] = None) -> str:
    """Simple function to get document title"""
    extractor = DocumentTitleExtractor()
    return extractor.extract_title(pdf_path, page_blocks)