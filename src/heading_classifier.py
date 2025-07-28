import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import re

@dataclass
class HeadingHierarchy:
    """Tracks document heading hierarchy state"""
    current_h1: Optional[str] = None
    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    last_level: int = 0
    h1_count: int = 0
    h2_count: int = 0
    h3_count: int = 0
    
    def update(self, level: int, text: str):
        """Update hierarchy state"""
        if level == 1:
            self.current_h1 = text
            self.current_h2 = None
            self.current_h3 = None
            self.h1_count += 1
        elif level == 2:
            self.current_h2 = text
            self.current_h3 = None
            self.h2_count += 1
        elif level == 3:
            self.current_h3 = text
            self.h3_count += 1
        self.last_level = level


class NumberingAnalyzer:
    """Analyzes numbering patterns to determine heading levels"""
    
    def __init__(self):
        # Regex for different numbering styles
        self.hierarchical_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$')
        self.simple_pattern = re.compile(r'^(\d+)\s*\.\s+(.+)$')
        self.letter_pattern = re.compile(r'^([A-Z])\s*\.\s+(.+)$', re.IGNORECASE)
        self.roman_pattern = re.compile(r'^([IVXLCDM]+)\s*\.\s+(.+)$', re.IGNORECASE)
        
        # Track numbering sequences
        self.number_sequences = defaultdict(list)
        
    def get_level_from_numbering(self, text: str) -> Tuple[int, str]:
        """Extract heading level from numbering pattern"""
        text = text.strip()
        
        # Check hierarchical numbering first (most reliable)
        match = self.hierarchical_pattern.match(text)
        if match:
            numbering = match.group(1)
            segments = numbering.split('.')
            
            # Level = number of segments (1 = H1, 1.1 = H2, 1.1.1 = H3)
            level = min(len(segments), 3)  # Cap at H3
            
            # Track the sequence
            self.number_sequences[level].append(numbering)
            
            return level, numbering
            
        # Check simple numbering (assume H1)
        if self.simple_pattern.match(text):
            return 1, text.split()[0]
            
        # Letter or roman numerals (context-dependent, default to H2)
        if self.letter_pattern.match(text) or self.roman_pattern.match(text):
            # Could be H1 or H2 depending on context
            return 2, text.split()[0]
            
        return 0, ""
    
    def validate_numbering_sequence(self, candidates: List['HeadingCandidate']):
        """Validate and adjust levels based on numbering sequence"""
        # Group by detected level
        by_level = defaultdict(list)
        for c in candidates:
            if c.numbering_pattern:
                by_level[c.heading_level].append(c)
                
        # Check for inconsistencies
        for level, items in by_level.items():
            if level == 1:
                # Check if these look like subsections
                if all('.' in c.numbering_pattern for c in items):
                    # Might be misclassified H2/H3
                    depths = [c.numbering_pattern.count('.') + 1 for c in items]
                    if min(depths) > 1:
                        # Adjust levels
                        for c in items:
                            c.heading_level = min(depths, 3)


class FontSizeClassifier:
    """Classifies heading levels based on font sizes"""
    
    def __init__(self, min_size_difference: float = 2.0):  # Increased for better separation
        self.min_size_difference = min_size_difference
        
    def classify_by_font_size(self, candidates: List['HeadingCandidate']) -> Dict[int, float]:
        """Group candidates into levels based on font size tiers"""
        if not candidates:
            return {}
            
        # Get unique font sizes
        font_sizes = sorted(set(c.font_size for c in candidates), reverse=True)
        
        # Create size tiers with improved algorithm
        size_tiers = self._create_improved_size_tiers(font_sizes)
        
        # Assign levels with strict hierarchy
        size_to_level = self._assign_levels_strictly(size_tiers, font_sizes)
                
        # Apply to candidates with consistency checks
        for candidate in candidates:
            if not candidate.has_numbering:  # Only for unnumbered headings
                candidate.heading_level = size_to_level.get(candidate.font_size, 3)
                
        return size_to_level
        
    def _create_improved_size_tiers(self, font_sizes: List[float]) -> List[List[float]]:
        """Create font size tiers with improved clustering"""
        if not font_sizes:
            return []
            
        if len(font_sizes) == 1:
            return [[font_sizes[0]]]
            
        # Calculate relative size differences
        size_diffs = []
        for i in range(1, len(font_sizes)):
            diff = font_sizes[i-1] - font_sizes[i]
            relative_diff = diff / font_sizes[i-1] if font_sizes[i-1] > 0 else 0
            size_diffs.append((diff, relative_diff, i))
        
        # Find significant breaks (both absolute and relative)
        significant_breaks = []
        for diff, rel_diff, idx in size_diffs:
            # Use both absolute difference (2pt+) and relative difference (20%+)
            if diff >= self.min_size_difference or rel_diff >= 0.2:
                significant_breaks.append(idx)
        
        # Create tiers based on breaks
        tiers = []
        start_idx = 0
        
        for break_idx in significant_breaks:
            if start_idx < break_idx:
                tiers.append(font_sizes[start_idx:break_idx])
            start_idx = break_idx
            
        # Add final tier
        if start_idx < len(font_sizes):
            tiers.append(font_sizes[start_idx:])
            
        # Ensure we have at least one tier
        if not tiers:
            tiers = [font_sizes]
            
        # Limit to 3 tiers but merge smaller tiers into H3 if needed
        if len(tiers) > 3:
            # Keep top 2 tiers as H1 and H2, merge rest into H3
            merged_h3 = []
            for tier in tiers[2:]:
                merged_h3.extend(tier)
            tiers = tiers[:2] + [merged_h3]
            
        return tiers
    
    def _assign_levels_strictly(self, tiers: List[List[float]], all_sizes: List[float]) -> Dict[float, int]:
        """Assign levels with strict font size hierarchy"""
        size_to_level = {}
        
        # Assign levels ensuring larger fonts get lower level numbers
        for tier_idx, tier_sizes in enumerate(tiers):
            level = min(tier_idx + 1, 3)  # H1, H2, or H3
            
            for size in tier_sizes:
                size_to_level[size] = level
        
        # Validate assignments - ensure no smaller font has lower level number
        for i, size1 in enumerate(all_sizes):
            for j, size2 in enumerate(all_sizes):
                if i < j:  # size1 > size2 (since sorted descending)
                    level1 = size_to_level.get(size1, 3)
                    level2 = size_to_level.get(size2, 3)
                    
                    # Fix hierarchy violations
                    if level1 > level2:  # Larger font has higher level number (wrong)
                        # Promote the larger font to match smaller font's level
                        size_to_level[size1] = level2
                        logging.debug(f"Fixed hierarchy: {size1}pt promoted to H{level2}")
        
        return size_to_level


class HierarchyValidator:
    """Validates and adjusts heading hierarchy"""
    
    def __init__(self):
        self.hierarchy = HeadingHierarchy()
        
    def validate_document_hierarchy(self, candidates: List['HeadingCandidate']):
        """Ensure consistent hierarchy throughout document"""
        # Sort by page and position
        sorted_candidates = sorted(candidates, 
                                 key=lambda c: (c.page_num, c.block.y0))
        
        # Track level transitions
        for i, candidate in enumerate(sorted_candidates):
            if candidate.heading_level is None:
                continue
                
            # Check for invalid jumps (e.g., H1 -> H3 without H2)
            if i > 0:
                prev_level = sorted_candidates[i-1].heading_level
                curr_level = candidate.heading_level
                
                if prev_level and curr_level - prev_level > 1:
                    # Invalid jump, adjust if not numbered
                    if not candidate.has_numbering:
                        candidate.heading_level = prev_level + 1
                        logging.debug(f"Adjusted heading level: '{candidate.text}' -> H{candidate.heading_level}")
                        
            # Update hierarchy state
            self.hierarchy.update(candidate.heading_level, candidate.text)
            
    def promote_orphan_headings(self, candidates: List['HeadingCandidate']):
        """Promote headings that appear without proper parents"""
        pages_with_headings = defaultdict(list)
        
        for c in candidates:
            if c.heading_level:
                pages_with_headings[c.page_num].append(c)
                
        for page_num, page_candidates in pages_with_headings.items():
            # Sort by position
            page_candidates.sort(key=lambda c: c.block.y0)
            
            # Check first heading on page
            if page_candidates and page_candidates[0].heading_level > 1:
                # First heading is H2/H3, might need promotion
                first = page_candidates[0]
                
                # Only promote if it's significantly larger or has strong indicators
                if first.size_ratio > 1.5 or (first.is_bold and first.is_uppercase):
                    first.heading_level = 1
                    logging.debug(f"Promoted first heading on page {page_num}: '{first.text}'")


class HeadingLevelClassifier:
    """Main classifier combining all strategies"""
    
    def __init__(self):
        self.numbering_analyzer = NumberingAnalyzer()
        self.font_classifier = FontSizeClassifier()
        self.hierarchy_validator = HierarchyValidator()
        
    def classify_heading_levels(self, candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
        """Assign heading levels to all candidates"""
        if not candidates:
            return []
            
        # Step 1: Classify numbered headings
        self._classify_numbered_headings(candidates)
        
        # Step 2: Classify unnumbered headings by font size
        self._classify_by_font_size(candidates)
        
        # Step 3: Validate numbering sequences
        self.numbering_analyzer.validate_numbering_sequence(candidates)
        
        # Step 4: Validate document hierarchy
        self.hierarchy_validator.validate_document_hierarchy(candidates)
        
        # Step 5: Handle orphan headings
        self.hierarchy_validator.promote_orphan_headings(candidates)
        
        # Step 6: Final adjustments
        self._final_adjustments(candidates)
        
        # Log statistics
        self._log_classification_stats(candidates)
        
        return candidates
        
    def _classify_numbered_headings(self, candidates: List['HeadingCandidate']):
        """Classify headings with numbering patterns"""
        for candidate in candidates:
            if candidate.has_numbering:
                level, numbering = self.numbering_analyzer.get_level_from_numbering(
                    candidate.text
                )
                if level > 0:
                    candidate.heading_level = level
                    candidate.numbering_pattern = numbering
                    
    def _classify_by_font_size(self, candidates: List['HeadingCandidate']):
        """Classify unnumbered headings by font size"""
        # Get unnumbered candidates
        unnumbered = [c for c in candidates if not c.has_numbering and c.heading_level is None]
        
        if unnumbered:
            self.font_classifier.classify_by_font_size(unnumbered)
            
    def _final_adjustments(self, candidates: List['HeadingCandidate']):
        """Make final adjustments based on context"""
        # Sort candidates by document order
        sorted_candidates = sorted(candidates, key=lambda c: (c.page_num, c.block.y0))
        
        # Global consistency pass
        self._ensure_global_font_hierarchy(sorted_candidates)
        
        # Page-level adjustments
        by_page = defaultdict(list)
        for c in candidates:
            if c.heading_level:
                by_page[c.page_num].append(c)
                
        for page_candidates in by_page.values():
            self._adjust_page_level_conflicts(page_candidates)
            
        # Document flow validation
        self._validate_document_flow(sorted_candidates)
    
    def _ensure_global_font_hierarchy(self, candidates: List['HeadingCandidate']):
        """Ensure font size hierarchy is respected globally"""
        if not candidates:
            return
            
        # Get unique size-level combinations and sort by font size
        size_level_map = {}
        for c in candidates:
            if c.heading_level:
                size = round(c.font_size, 1)  # Round to avoid floating point issues
                if size not in size_level_map:
                    size_level_map[size] = set()
                size_level_map[size].add(c.heading_level)
        
        # Convert to sorted list for processing
        size_levels = []
        for size in sorted(size_level_map.keys(), reverse=True):
            levels = size_level_map[size]
            # If multiple levels for same size, take the lowest (best) level
            best_level = min(levels)
            size_levels.append((size, best_level))
        
        # Create corrected mapping ensuring hierarchy
        corrected_mapping = {}
        min_level_so_far = 1
        
        for size, original_level in size_levels:
            # Ensure this size gets at least min_level_so_far
            corrected_level = max(original_level, min_level_so_far)
            corrected_mapping[size] = corrected_level
            
            # Update minimum level for smaller fonts
            min_level_so_far = corrected_level
        
        # Apply corrections
        for c in candidates:
            if c.heading_level:
                size = round(c.font_size, 1)
                new_level = corrected_mapping.get(size, c.heading_level)
                if new_level != c.heading_level:
                    logging.debug(f"Hierarchy fix: {size}pt changed from H{c.heading_level} to H{new_level}")
                    c.heading_level = new_level
    
    def _adjust_page_level_conflicts(self, page_candidates: List['HeadingCandidate']):
        """Resolve conflicts within a single page"""
        if len(page_candidates) <= 1:
            return
            
        # Sort by position
        page_candidates.sort(key=lambda c: c.block.y0)
        
        # Check for same-font-different-level conflicts
        font_groups = defaultdict(list)
        for c in page_candidates:
            font_groups[c.font_size].append(c)
            
        for font_size, candidates_with_size in font_groups.items():
            if len(candidates_with_size) > 1:
                levels = [c.heading_level for c in candidates_with_size]
                if len(set(levels)) > 1:  # Multiple levels for same font size
                    # Use context to decide the correct level
                    correct_level = self._determine_correct_level_by_context(candidates_with_size)
                    for c in candidates_with_size:
                        if c.heading_level != correct_level:
                            logging.debug(f"Same-font fix: '{c.text[:30]}...' changed from H{c.heading_level} to H{correct_level}")
                            c.heading_level = correct_level
        
        # Handle all-same-level pages
        levels = [c.heading_level for c in page_candidates]
        level_counts = Counter(levels)
        
        if len(level_counts) == 1 and len(page_candidates) > 2:
            # All headings same level - differentiate by context
            self._differentiate_by_enhanced_context(page_candidates)
    
    def _determine_correct_level_by_context(self, candidates: List['HeadingCandidate']) -> int:
        """Determine correct level based on context clues"""
        # Prefer level from numbered headings (most reliable)
        numbered_levels = [c.heading_level for c in candidates if c.has_numbering]
        if numbered_levels:
            return max(set(numbered_levels), key=numbered_levels.count)  # Most common
            
        # Consider confidence scores
        level_confidences = defaultdict(list)
        for c in candidates:
            level_confidences[c.heading_level].append(c.confidence_score)
            
        # Return level with highest average confidence
        best_level = 3
        best_avg_confidence = 0
        for level, confidences in level_confidences.items():
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf > best_avg_confidence:
                best_avg_confidence = avg_conf
                best_level = level
                
        return best_level
    
    def _differentiate_by_enhanced_context(self, page_candidates: List['HeadingCandidate']):
        """Differentiate same-level headings using enhanced context"""
        # Sort by confidence and other factors
        scored_candidates = []
        for c in page_candidates:
            score = 0
            
            # Factors that suggest higher-level heading
            if c.has_numbering:
                score += 20
            if c.is_uppercase:
                score += 15
            if c.is_bold and c.has_distinct_font:
                score += 10
            if c.size_ratio > 1.3:
                score += 10
            if c.vertical_gap_before > 15:
                score += 5
                
            # Position factors
            if c.block.y0 < 100:  # Top of page
                score += 8
                
            scored_candidates.append((score, c))
        
        # Sort by score descending
        scored_candidates.sort(reverse=True)
        
        # Assign levels based on scores (top 1/3 get current level, rest get level+1)
        num_candidates = len(scored_candidates)
        current_level = page_candidates[0].heading_level
        
        for i, (score, candidate) in enumerate(scored_candidates):
            if i < num_candidates // 3:  # Top third
                candidate.heading_level = current_level
            elif current_level < 3:  # Can demote
                candidate.heading_level = current_level + 1
            # else keep current level (already H3)
    
    def _validate_document_flow(self, sorted_candidates: List['HeadingCandidate']):
        """Validate overall document flow and structure"""
        if len(sorted_candidates) < 2:
            return
            
        # Check for unreasonable level jumps
        for i in range(1, len(sorted_candidates)):
            prev_level = sorted_candidates[i-1].heading_level
            curr_level = sorted_candidates[i].heading_level
            
            # Skip if either is None
            if not prev_level or not curr_level:
                continue
                
            # Check for inappropriate jumps (more than 1 level down)
            if curr_level - prev_level > 1:
                # Only adjust if current heading isn't numbered (numbered are reliable)
                if not sorted_candidates[i].has_numbering:
                    new_level = prev_level + 1
                    logging.debug(f"Flow fix: '{sorted_candidates[i].text[:30]}...' adjusted from H{curr_level} to H{new_level}")
                    sorted_candidates[i].heading_level = new_level
                        
    def _log_classification_stats(self, candidates: List['HeadingCandidate']):
        """Log classification statistics"""
        level_counts = Counter(c.heading_level for c in candidates if c.heading_level)
        
        logging.info("Heading classification complete:")
        for level in sorted(level_counts.keys()):
            logging.info(f"  H{level}: {level_counts[level]} headings")
            
        # Log some examples
        for level in [1, 2, 3]:
            examples = [c for c in candidates if c.heading_level == level][:3]
            if examples:
                logging.debug(f"H{level} examples: {[e.text[:50] for e in examples]}")


# Integration function
def classify_heading_levels(candidates: List['HeadingCandidate']) -> List['HeadingCandidate']:
    """Main function to classify heading levels"""
    classifier = HeadingLevelClassifier()
    return classifier.classify_heading_levels(candidates)


# Helper function to get final headings
def get_classified_headings(candidates: List['HeadingCandidate']) -> List[Dict[str, any]]:
    """Convert classified candidates to output format"""
    headings = []
    
    for candidate in candidates:
        if candidate.heading_level:
            headings.append({
                'level': f'H{candidate.heading_level}',
                'text': candidate.text,
                'page': candidate.page_num,  # 0-indexed as required
                'confidence': candidate.confidence_score,
                'font_size': candidate.font_size,
                'has_numbering': candidate.has_numbering
            })
            
    # Sort by page and position
    headings.sort(key=lambda h: (h['page'], candidates[headings.index(h)].block.y0))
    
    return headings