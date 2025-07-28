import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from collections import defaultdict

@dataclass
class TextBlock:
    """Enhanced text block with all formatting information"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_name: str
    font_size: float
    font_flags: int  # bold, italic etc.
    block_type: str
    block_no: int
    line_no: int = 0
    span_no: int = 0
    page_num: int = 0
    column_id: Optional[int] = None
    reading_order: Optional[int] = None
    
    @property
    def x0(self) -> float:
        return self.bbox[0]
    
    @property
    def y0(self) -> float:
        return self.bbox[1]
    
    @property
    def x1(self) -> float:
        return self.bbox[2]
    
    @property
    def y1(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def is_bold(self) -> bool:
        return bool(self.font_flags & 2**4)  # 16
    
    @property
    def is_italic(self) -> bool:
        return bool(self.font_flags & 2**1)  # 2


class LayoutAnalyzer:
    """Analyzes page layout and determines reading order"""
    
    def __init__(self, tolerance_y: float = 5.0, tolerance_x: float = 50.0):
        self.tolerance_y = tolerance_y  # Vertical tolerance for same line
        self.tolerance_x = tolerance_x  # Horizontal tolerance for columns
        
    def extract_text_blocks(self, page, page_num: int) -> List[TextBlock]:
        """Extract all text blocks with detailed formatting info"""
        blocks = []
        
        # Get detailed text information
        page_dict = page.get_text("dict")
        
        for block_no, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:  # Text block
                block_bbox = block["bbox"]
                
                for line_no, line in enumerate(block["lines"]):
                    for span_no, span in enumerate(line["spans"]):
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        text_block = TextBlock(
                            text=text,
                            bbox=tuple(span["bbox"]),
                            font_name=span.get("font", ""),
                            font_size=span.get("size", 0),
                            font_flags=span.get("flags", 0),
                            block_type="text",
                            block_no=block_no,
                            line_no=line_no,
                            span_no=span_no,
                            page_num=page_num
                        )
                        blocks.append(text_block)
                        
        return blocks
    
    def detect_columns(self, blocks: List[TextBlock]) -> Dict[int, List[TextBlock]]:
        """Detect columns using DBSCAN clustering on x-coordinates"""
        if not blocks:
            return {0: []}
            
        # Extract x-coordinates of block centers
        x_coords = np.array([[block.center_x] for block in blocks])
        
        # Use DBSCAN to cluster blocks by x-coordinate
        clustering = DBSCAN(eps=self.tolerance_x, min_samples=3)
        labels = clustering.fit_predict(x_coords)
        
        # Group blocks by column
        columns = defaultdict(list)
        for block, label in zip(blocks, labels):
            # Noise points (-1) are assigned to their own column
            if label == -1:
                label = max(columns.keys(), default=-1) + 1
            block.column_id = label
            columns[label].append(block)
            
        # Sort columns by average x-coordinate (left to right)
        sorted_columns = dict(sorted(
            columns.items(),
            key=lambda x: np.mean([b.center_x for b in x[1]])
        ))
        
        # Reassign column IDs to be sequential
        final_columns = {}
        for new_id, (old_id, blocks_list) in enumerate(sorted_columns.items()):
            for block in blocks_list:
                block.column_id = new_id
            final_columns[new_id] = blocks_list
            
        return final_columns
    
    def sort_blocks_in_column(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort blocks within a column by vertical position"""
        # Group blocks that are on the same line (within tolerance)
        lines = []
        sorted_by_y = sorted(blocks, key=lambda b: b.y0)
        
        current_line = [sorted_by_y[0]]
        current_y = sorted_by_y[0].y0
        
        for block in sorted_by_y[1:]:
            if abs(block.y0 - current_y) <= self.tolerance_y:
                current_line.append(block)
            else:
                # Sort current line by x-coordinate
                lines.append(sorted(current_line, key=lambda b: b.x0))
                current_line = [block]
                current_y = block.y0
                
        # Don't forget the last line
        if current_line:
            lines.append(sorted(current_line, key=lambda b: b.x0))
            
        # Flatten the sorted lines
        return [block for line in lines for block in line]
    
    def determine_reading_order(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Determine global reading order considering columns"""
        if not blocks:
            return []
            
        # Detect columns
        columns = self.detect_columns(blocks)
        
        # Sort blocks within each column
        sorted_blocks = []
        for col_id in sorted(columns.keys()):
            column_blocks = self.sort_blocks_in_column(columns[col_id])
            sorted_blocks.extend(column_blocks)
            
        # Assign reading order
        for order, block in enumerate(sorted_blocks):
            block.reading_order = order
            
        return sorted_blocks
    
    def merge_adjacent_blocks(self, blocks: List[TextBlock], 
                            horizontal_threshold: float = 10.0) -> List[TextBlock]:
        """Merge blocks that are likely part of the same logical unit"""
        if not blocks:
            return []
            
        merged = []
        blocks_by_line = defaultdict(list)
        
        # Group blocks by approximate y-coordinate
        for block in blocks:
            y_key = round(block.y0 / self.tolerance_y) * self.tolerance_y
            blocks_by_line[y_key].append(block)
            
        # Process each line
        for y_key in sorted(blocks_by_line.keys()):
            line_blocks = sorted(blocks_by_line[y_key], key=lambda b: b.x0)
            
            if not line_blocks:
                continue
                
            current_merged = line_blocks[0]
            
            for block in line_blocks[1:]:
                # Check if blocks should be merged
                gap = block.x0 - current_merged.x1
                same_font = (block.font_name == current_merged.font_name and 
                           abs(block.font_size - current_merged.font_size) < 0.5)
                
                if gap <= horizontal_threshold and same_font:
                    # Merge blocks
                    current_merged = TextBlock(
                        text=current_merged.text + " " + block.text,
                        bbox=(current_merged.x0, 
                              min(current_merged.y0, block.y0),
                              block.x1,
                              max(current_merged.y1, block.y1)),
                        font_name=current_merged.font_name,
                        font_size=current_merged.font_size,
                        font_flags=current_merged.font_flags,
                        block_type=current_merged.block_type,
                        block_no=current_merged.block_no,
                        line_no=current_merged.line_no,
                        span_no=current_merged.span_no,
                        page_num=current_merged.page_num
                    )
                else:
                    # Save current merged block and start new one
                    merged.append(current_merged)
                    current_merged = block
                    
            # Don't forget the last block
            merged.append(current_merged)
            
        return merged
    
    def merge_wrapped_headings(self, blocks: List[TextBlock], 
                              vertical_threshold: float = 15.0,
                              max_heading_words: int = 15) -> List[TextBlock]:
        """Merge lines that appear to be wrapped headings"""
        if not blocks:
            return []
            
        merged = []
        i = 0
        
        while i < len(blocks):
            current_block = blocks[i]
            
            # Check if this could be a heading that wraps to next line
            if (i + 1 < len(blocks) and 
                len(current_block.text.split()) <= max_heading_words):
                
                next_block = blocks[i + 1]
                
                # Check for wrapping indicators
                should_merge = self._should_merge_wrapped_lines(
                    current_block, next_block, vertical_threshold
                )
                
                if should_merge:
                    # Merge the blocks
                    merged_text = self._merge_wrapped_text(
                        current_block.text, next_block.text
                    )
                    
                    merged_block = TextBlock(
                        text=merged_text,
                        bbox=(current_block.x0,
                              current_block.y0,
                              max(current_block.x1, next_block.x1),
                              next_block.y1),
                        font_name=current_block.font_name,
                        font_size=current_block.font_size,
                        font_flags=current_block.font_flags,
                        block_type=current_block.block_type,
                        block_no=current_block.block_no,
                        line_no=current_block.line_no,
                        span_no=current_block.span_no,
                        page_num=current_block.page_num
                    )
                    
                    merged.append(merged_block)
                    i += 2  # Skip next block as it's been merged
                    continue
            
            # No merging, add block as-is
            merged.append(current_block)
            i += 1
            
        return merged
    
    def _should_merge_wrapped_lines(self, block1: TextBlock, block2: TextBlock, 
                                   vertical_threshold: float) -> bool:
        """Determine if two blocks should be merged as wrapped lines"""
        # Must be on consecutive lines (close vertically)
        vertical_gap = abs(block2.y0 - block1.y1)
        if vertical_gap > vertical_threshold:
            return False
            
        # Must have similar or compatible formatting
        if (block1.font_name != block2.font_name or 
            abs(block1.font_size - block2.font_size) > 1.0):
            return False
            
        # Check for wrapping indicators
        text1 = block1.text.strip()
        text2 = block2.text.strip()
        
        # Case 1: First line ends with hyphen (word break)
        if text1.endswith('-'):
            return True
            
        # Case 2: First line ends mid-sentence (no punctuation)
        if (not text1.endswith(('.', '!', '?', ':', ';')) and 
            not text1[-1].isdigit() and
            len(text1.split()) < 10):  # Likely heading length
            
            # Second line looks like continuation (starts lowercase or continues phrase)
            if (text2 and 
                (text2[0].islower() or 
                 text2.startswith(('and', 'or', 'of', 'for', 'with', 'to', 'in')))):
                return True
                
        # Case 3: Both lines have similar formatting and short length (likely heading)
        if (block1.font_size > 12 and  # Larger than typical body text
            len(text1.split()) <= 6 and len(text2.split()) <= 6 and
            block1.is_bold == block2.is_bold):
            return True
            
        return False
    
    def _merge_wrapped_text(self, text1: str, text2: str) -> str:
        """Merge wrapped text, handling hyphens and spacing"""
        text1 = text1.strip()
        text2 = text2.strip()
        
        # Handle hyphenated word breaks
        if text1.endswith('-'):
            # Remove hyphen and join directly (word continuation)
            return text1[:-1] + text2
        else:
            # Regular space join
            return text1 + " " + text2
    
    def analyze_page_layout(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze overall page layout characteristics"""
        if not blocks:
            return {"columns": 0, "dominant_font": None, "avg_font_size": 0}
            
        columns = self.detect_columns(blocks)
        
        # Font statistics
        font_counts = defaultdict(int)
        font_sizes = []
        
        for block in blocks:
            font_counts[block.font_name] += 1
            font_sizes.append(block.font_size)
            
        dominant_font = max(font_counts.items(), key=lambda x: x[1])[0] if font_counts else None
        avg_font_size = np.mean(font_sizes) if font_sizes else 0
        
        return {
            "columns": len(columns),
            "dominant_font": dominant_font,
            "avg_font_size": avg_font_size,
            "font_distribution": dict(font_counts),
            "column_widths": {
                col_id: max(b.x1 for b in blocks) - min(b.x0 for b in blocks)
                for col_id, blocks in columns.items()
            }
        }


class EnhancedLayoutExtractor:
    """Main class combining PDF parsing with layout analysis"""
    
    def __init__(self, merge_blocks: bool = True):
        self.analyzer = LayoutAnalyzer()
        self.merge_blocks = merge_blocks
        
    def process_page(self, page, page_num: int) -> Tuple[List[TextBlock], Dict[str, Any]]:
        """Process a single page and extract layout information"""
        # Extract text blocks
        blocks = self.analyzer.extract_text_blocks(page, page_num)
        
        if not blocks:
            return [], {"error": "No text blocks found"}
            
        # Optionally merge adjacent blocks
        if self.merge_blocks:
            blocks = self.analyzer.merge_adjacent_blocks(blocks)
            
        # Merge wrapped headings before ordering
        blocks = self.analyzer.merge_wrapped_headings(blocks)
            
        # Determine reading order
        ordered_blocks = self.analyzer.determine_reading_order(blocks)
        
        # Analyze layout
        layout_info = self.analyzer.analyze_page_layout(ordered_blocks)
        
        return ordered_blocks, layout_info
    
    def extract_structured_text(self, blocks: List[TextBlock]) -> str:
        """Extract text in reading order with column markers"""
        if not blocks:
            return ""
            
        text_parts = []
        current_column = None
        
        for block in sorted(blocks, key=lambda b: b.reading_order):
            if block.column_id != current_column:
                if current_column is not None:
                    text_parts.append("\n[Column Break]\n")
                current_column = block.column_id
                
            text_parts.append(block.text)
            
        return " ".join(text_parts)


# Integration function
def extract_layout_from_pdf(pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
    """Extract layout information from specified pages"""
    import fitz
    
    results = {}
    extractor = EnhancedLayoutExtractor(merge_blocks=True)
    
    with fitz.open(pdf_path) as doc:
        pages_to_process = page_numbers if page_numbers else range(len(doc))
        
        for page_num in pages_to_process:
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            blocks, layout_info = extractor.process_page(page, page_num)
            
            results[page_num] = {
                "blocks": blocks,
                "layout": layout_info,
                "structured_text": extractor.extract_structured_text(blocks)
            }
            
            logging.info(f"Page {page_num}: Found {len(blocks)} blocks in {layout_info['columns']} columns")
            
    return results