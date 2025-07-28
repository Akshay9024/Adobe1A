import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import logging
import fitz  # PyMuPDF for table detection

@dataclass
class HeadingCandidate:
    """Represents a potential heading with detection features"""
    text: str
    block: 'TextBlock'  # Reference to original block
    page_num: int
    
    # Detection features
    font_size: float
    is_larger_than_body: bool = False
    size_ratio: float = 1.0  # ratio to body text size
    
    is_bold: bool = False
    is_italic: bool = False
    has_distinct_font: bool = False
    
    has_numbering: bool = False
    numbering_pattern: Optional[str] = None
    numbering_depth: int = 0  # based on dots in numbering
    
    is_uppercase: bool = False
    is_titlecase: bool = False
    word_count: int = 0
    
    vertical_gap_before: float = 0.0
    has_spacing_cue: bool = False
    
    confidence_score: float = 0.0
    heading_level: Optional[int] = None  # Will be assigned later
    
    def __hash__(self):
        return hash((self.text, self.page_num, self.block.y0))
    
    def __lt__(self, other):
        """Enable sorting by page number and y-coordinate"""
        if self.page_num != other.page_num:
            return self.page_num < other.page_num
        return self.block.y0 < other.block.y0


class HeadingPatterns:
    """Regex patterns for various heading formats"""
    
    # Hierarchical numbering: 1, 1.1, 1.1.1, etc.
    DECIMAL_HIERARCHICAL = re.compile(r'^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$')
    
    # Simple numbering: 1. Title
    SIMPLE_DECIMAL = re.compile(r'^(\d+)\s*\.\s+(.+)$')
    
    # Letter numbering: A. Title, B. Title
    LETTER_NUMBERING = re.compile(r'^([A-Z])\s*\.\s+(.+)$', re.IGNORECASE)
    
    # Roman numerals: I., II., III., etc.
    ROMAN_NUMERALS = re.compile(r'^([IVXLCDM]+)\s*\.\s+(.+)$', re.IGNORECASE)
    
    # Section indicators (multilingual friendly)
    SECTION_PATTERN = re.compile(r'^(Section|Chapter|Part|Appendix)\s+(\d+|[A-Z]|[IVXLCDM]+)', re.IGNORECASE)
    
    # Parenthetical numbering: (1), (a), (i)
    PARENTHETICAL = re.compile(r'^\((\d+|[a-z]|[ivx]+)\)\s+(.+)$', re.IGNORECASE)
    
    # Dash or bullet patterns
    BULLET_PATTERN = re.compile(r'^[\-•‣⁃]\s+(.+)$')
    
    @classmethod
    def detect_numbering(cls, text: str) -> Tuple[bool, Optional[str], int]:
        """Detect numbering pattern and depth"""
        text = text.strip()
        
        # Check hierarchical decimal first (most specific)
        match = cls.DECIMAL_HIERARCHICAL.match(text)
        if match:
            numbering = match.group(1)
            depth = numbering.count('.') + 1
            return True, 'hierarchical_decimal', depth
            
        # Check other patterns
        patterns = [
            (cls.SIMPLE_DECIMAL, 'simple_decimal', 1),
            (cls.LETTER_NUMBERING, 'letter', 1),
            (cls.ROMAN_NUMERALS, 'roman', 1),
            (cls.SECTION_PATTERN, 'section', 1),
            (cls.PARENTHETICAL, 'parenthetical', 1),
            (cls.BULLET_PATTERN, 'bullet', 1)
        ]
        
        for pattern, name, depth in patterns:
            if pattern.match(text):
                return True, name, depth
                
        return False, None, 0


class FontAnalyzer:
    """Analyzes font characteristics across document"""
    
    def __init__(self):
        self.font_sizes = []
        self.font_names = []
        self.body_text_size = None
        self.font_size_distribution = None
        
    def analyze_document_fonts(self, blocks: List['TextBlock']) -> Dict[str, any]:
        """Analyze font usage across entire document"""
        font_size_counter = Counter()
        font_name_counter = Counter()
        
        for block in blocks:
            if block.font_size > 0:
                # Round font sizes to nearest 0.5 for better grouping
                rounded_size = round(block.font_size * 2) / 2
                font_size_counter[rounded_size] += len(block.text)
                font_name_counter[block.font_name] += 1
                
        # Find mode (most common) font size by character count
        if font_size_counter:
            self.body_text_size = font_size_counter.most_common(1)[0][0]
        else:
            self.body_text_size = 12.0  # Default fallback
            
        # Calculate font size distribution
        sizes = list(font_size_counter.keys())
        if sizes:
            self.font_size_distribution = {
                'min': min(sizes),
                'max': max(sizes),
                'mode': self.body_text_size,
                'mean': np.mean(sizes),
                'std': np.std(sizes)
            }
        
        return {
            'body_text_size': self.body_text_size,
            'font_distribution': self.font_size_distribution,
            'common_fonts': font_name_counter.most_common(5)
        }
        
    def is_larger_than_body(self, font_size: float, threshold_ratio: float = 1.1) -> bool:
        """Check if font size is larger than body text"""
        if self.body_text_size is None:
            return False
        return font_size > self.body_text_size * threshold_ratio
        
    def get_size_ratio(self, font_size: float) -> float:
        """Get ratio of font size to body text size"""
        if self.body_text_size is None or self.body_text_size == 0:
            return 1.0
        return font_size / self.body_text_size
        
    def is_distinct_font(self, font_name: str, common_fonts: List[str]) -> bool:
        """Check if font is distinct from common body fonts"""
        if not common_fonts:
            return False
            
        # Check for bold/italic indicators in font name
        font_lower = font_name.lower()
        if any(indicator in font_lower for indicator in ['bold', 'heavy', 'black', 'semibold']):
            return True
            
        # Check if it's different from most common fonts
        return font_name not in common_fonts[:2]


@dataclass
class ColumnLayout:
    """Represents a detected column layout on a page"""
    page_num: int
    num_columns: int
    column_regions: List[Tuple[float, float, float, float]] # [(x0, y0, x1, y1), ...]
    column_boundaries: List[float]  # X-coordinates of column separations
    gutter_width: float = 0.0
    confidence: float = 0.0
    layout_type: str = "single"  # "single", "double", "triple", "multi"
    
    def get_column_for_block(self, block: 'TextBlock') -> int:
        """Determine which column a text block belongs to"""
        block_center_x = (block.x0 + block.x1) / 2
        
        for i, (x0, y0, x1, y1) in enumerate(self.column_regions):
            if x0 <= block_center_x <= x1:
                return i
        
        # Fallback: find closest column
        min_distance = float('inf')
        closest_column = 0
        for i, (x0, y0, x1, y1) in enumerate(self.column_regions):
            column_center = (x0 + x1) / 2
            distance = abs(block_center_x - column_center)
            if distance < min_distance:
                min_distance = distance
                closest_column = i
                
        return closest_column


class MultiColumnDetector:
    """Detects and handles multi-column page layouts"""
    
    def __init__(self, 
                 column_density_threshold: float = 1.4,
                 min_gutter_width: float = 20.0,
                 min_column_width: float = 100.0):
        self.column_density_threshold = column_density_threshold
        self.min_gutter_width = min_gutter_width
        self.min_column_width = min_column_width
        
    def detect_column_layouts(self, blocks: List['TextBlock']) -> List[ColumnLayout]:
        """Detect column layouts across all pages"""
        layouts = []
        
        # Group blocks by page
        blocks_by_page = defaultdict(list)
        for block in blocks:
            blocks_by_page[block.page_num].append(block)
            
        for page_num, page_blocks in blocks_by_page.items():
            if not page_blocks:
                continue
                
            layout = self._analyze_page_layout(page_num, page_blocks)
            if layout:
                layouts.append(layout)
                
        return layouts
    
    def _analyze_page_layout(self, page_num: int, blocks: List['TextBlock']) -> Optional[ColumnLayout]:
        """Analyze layout for a single page"""
        if len(blocks) < 10:  # Not enough blocks for meaningful analysis
            return self._create_single_column_layout(page_num, blocks)
            
        # Step 1: Detect column density using Y-scanline analysis
        column_density = self._calculate_column_density(blocks)
        
        if column_density < self.column_density_threshold:
            return self._create_single_column_layout(page_num, blocks)
            
        # Step 2: Find column boundaries using X-position clustering
        column_boundaries = self._find_column_boundaries(blocks)
        
        if len(column_boundaries) < 1:  # Need at least one boundary for multi-column
            return self._create_single_column_layout(page_num, blocks)
            
        # Step 3: Create column regions
        column_regions = self._create_column_regions(blocks, column_boundaries)
        
        # Step 4: Validate column layout
        if not self._validate_column_layout(blocks, column_regions):
            return self._create_single_column_layout(page_num, blocks)
            
        # Step 5: Calculate layout properties
        num_columns = len(column_regions)
        gutter_width = self._calculate_gutter_width(column_boundaries, column_regions)
        confidence = self._calculate_layout_confidence(blocks, column_regions, column_density)
        
        layout_type = {1: "single", 2: "double", 3: "triple"}.get(num_columns, "multi")
        
        return ColumnLayout(
            page_num=page_num,
            num_columns=num_columns,
            column_regions=column_regions,
            column_boundaries=column_boundaries,
            gutter_width=gutter_width,
            confidence=confidence,
            layout_type=layout_type
        )
    
    def _calculate_column_density(self, blocks: List['TextBlock']) -> float:
        """Calculate average number of blocks per Y-scanline"""
        if not blocks:
            return 0.0
            
        # Group blocks by Y-position (with tolerance for line height)
        y_groups = defaultdict(list)
        for block in blocks:
            # Round Y coordinate to group blocks on similar lines
            rounded_y = round(block.y0 / 10) * 10  # 10-point tolerance
            y_groups[rounded_y].append(block)
            
        if not y_groups:
            return 0.0
            
        # Calculate average blocks per Y-level
        total_blocks = sum(len(group) for group in y_groups.values())
        num_y_levels = len(y_groups)
        
        return total_blocks / num_y_levels
    
    def _find_column_boundaries(self, blocks: List['TextBlock']) -> List[float]:
        """Find column boundaries by analyzing X-position gaps"""
        if not blocks:
            return []
            
        # Get all left and right edges
        left_edges = [block.x0 for block in blocks]
        right_edges = [block.x1 for block in blocks]
        
        # Find potential gutters (large gaps between text)
        all_x_positions = sorted(set(left_edges + right_edges))
        
        # Look for large gaps that could be column gutters
        potential_boundaries = []
        
        for i in range(len(all_x_positions) - 1):
            gap_start = all_x_positions[i]
            gap_end = all_x_positions[i + 1]
            gap_width = gap_end - gap_start
            
            if gap_width >= self.min_gutter_width:
                # Check if this gap consistently appears across multiple Y-levels
                if self._is_consistent_gutter(blocks, gap_start, gap_end):
                    boundary_x = (gap_start + gap_end) / 2
                    potential_boundaries.append(boundary_x)
                    
        return potential_boundaries
    
    def _is_consistent_gutter(self, blocks: List['TextBlock'], gap_start: float, gap_end: float) -> bool:
        """Check if a gap consistently appears as a gutter across Y-levels"""
        # Group blocks by Y-level
        y_groups = defaultdict(list)
        for block in blocks:
            rounded_y = round(block.y0 / 20) * 20  # 20-point tolerance
            y_groups[rounded_y].append(block)
            
        gutter_count = 0
        total_levels = len(y_groups)
        
        for y_level, level_blocks in y_groups.items():
            # Check if this Y-level has text on both sides of the gap
            has_left_text = any(block.x1 <= gap_start + 5 for block in level_blocks)
            has_right_text = any(block.x0 >= gap_end - 5 for block in level_blocks)
            
            if has_left_text and has_right_text:
                gutter_count += 1
                
        # Require gutter to appear in at least 60% of Y-levels
        return (gutter_count / total_levels) >= 0.6 if total_levels > 0 else False
    
    def _create_column_regions(self, blocks: List['TextBlock'], 
                              boundaries: List[float]) -> List[Tuple[float, float, float, float]]:
        """Create column regions based on boundaries"""
        if not blocks:
            return []
            
        # Get page bounds
        min_x = min(block.x0 for block in blocks)
        max_x = max(block.x1 for block in blocks)
        min_y = min(block.y0 for block in blocks)
        max_y = max(block.y1 for block in blocks)
        
        regions = []
        
        # Sort boundaries
        sorted_boundaries = sorted(boundaries)
        
        # Create regions between boundaries
        current_x = min_x
        
        for boundary in sorted_boundaries:
            if boundary - current_x >= self.min_column_width:
                regions.append((current_x, min_y, boundary, max_y))
                current_x = boundary
                
        # Add final region
        if max_x - current_x >= self.min_column_width:
            regions.append((current_x, min_y, max_x, max_y))
            
        return regions
    
    def _validate_column_layout(self, blocks: List['TextBlock'], 
                               regions: List[Tuple[float, float, float, float]]) -> bool:
        """Validate that the detected column layout makes sense"""
        if len(regions) < 2:
            return False
            
        # Check that each column has reasonable content
        for i, (x0, y0, x1, y1) in enumerate(regions):
            column_blocks = [
                block for block in blocks
                if x0 <= (block.x0 + block.x1) / 2 <= x1
            ]
            
            # Each column should have at least a few blocks
            if len(column_blocks) < 3:
                return False
                
            # Column should have reasonable width
            if x1 - x0 < self.min_column_width:
                return False
                
        return True
    
    def _calculate_gutter_width(self, boundaries: List[float], 
                               regions: List[Tuple[float, float, float, float]]) -> float:
        """Calculate average gutter width"""
        if len(regions) < 2:
            return 0.0
            
        gutter_widths = []
        for i in range(len(regions) - 1):
            left_region = regions[i]
            right_region = regions[i + 1]
            gutter_width = right_region[0] - left_region[2]  # x0_right - x1_left
            gutter_widths.append(gutter_width)
            
        return np.mean(gutter_widths) if gutter_widths else 0.0
    
    def _calculate_layout_confidence(self, blocks: List['TextBlock'], 
                                   regions: List[Tuple[float, float, float, float]],
                                   density: float) -> float:
        """Calculate confidence in the detected layout"""
        if len(regions) < 2:
            return 0.0
            
        confidence = 0.0
        
        # Factor 1: Column density (higher = more likely multi-column)
        density_score = min((density - 1.0) * 0.4, 0.4)  # Max 40 points
        confidence += density_score
        
        # Factor 2: Balance between columns
        column_block_counts = []
        for x0, y0, x1, y1 in regions:
            column_blocks = [
                block for block in blocks
                if x0 <= (block.x0 + block.x1) / 2 <= x1
            ]
            column_block_counts.append(len(column_blocks))
            
        if column_block_counts:
            balance_variance = np.var(column_block_counts)
            balance_score = max(0.3 - balance_variance / 100, 0)  # Max 30 points
            confidence += balance_score
            
        # Factor 3: Consistent column widths
        column_widths = [x1 - x0 for x0, y0, x1, y1 in regions]
        if len(column_widths) > 1:
            width_variance = np.var(column_widths)
            width_score = max(0.3 - width_variance / 10000, 0)  # Max 30 points
            confidence += width_score
            
        return min(confidence, 1.0)
    
    def _create_single_column_layout(self, page_num: int, blocks: List['TextBlock']) -> ColumnLayout:
        """Create a single-column layout as fallback"""
        if not blocks:
            return ColumnLayout(
                page_num=page_num,
                num_columns=1,
                column_regions=[(0, 0, 100, 100)],
                column_boundaries=[],
                layout_type="single"
            )
            
        min_x = min(block.x0 for block in blocks)
        max_x = max(block.x1 for block in blocks)
        min_y = min(block.y0 for block in blocks)
        max_y = max(block.y1 for block in blocks)
        
        return ColumnLayout(
            page_num=page_num,
            num_columns=1,
            column_regions=[(min_x, min_y, max_x, max_y)],
            column_boundaries=[],
            confidence=1.0,
            layout_type="single"
        )


@dataclass
class TableRegion:
    """Represents a detected table region"""
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    blocks: List['TextBlock']
    columns: List[float]  # X-coordinates of column boundaries
    confidence: float = 0.0
    table_type: str = "grid"  # "grid", "aligned", "bordered"
    
    def contains_block(self, block: 'TextBlock') -> bool:
        """Check if a text block is within this table region"""
        return (self.page_num == block.page_num and
                self.x0 <= block.x0 <= self.x1 and
                self.y0 <= block.y0 <= self.y1)


class TableDetector:
    """Detects table regions to avoid misclassifying table content as headings"""
    
    def __init__(self, alignment_threshold: float = 5.0,
                 min_column_blocks: int = 4,
                 min_row_blocks: int = 3):
        self.alignment_threshold = alignment_threshold
        self.min_column_blocks = min_column_blocks
        self.min_row_blocks = min_row_blocks
        
    def detect_table_regions(self, blocks: List['TextBlock'], 
                           pdf_path: Optional[str] = None) -> List[TableRegion]:
        """Detect table regions using multiple methods"""
        table_regions = []
        
        # Group blocks by page
        blocks_by_page = defaultdict(list)
        for block in blocks:
            blocks_by_page[block.page_num].append(block)
            
        for page_num, page_blocks in blocks_by_page.items():
            if not page_blocks:
                continue
                
            # Method 1: Column alignment clustering
            aligned_regions = self._detect_column_alignment(page_blocks, page_num)
            table_regions.extend(aligned_regions)
            
            # Method 2: Row pattern detection
            row_regions = self._detect_row_patterns(page_blocks, page_num)
            table_regions.extend(row_regions)
            
            # Method 3: PyMuPDF table detection (if PDF path available)
            if pdf_path:
                pymupdf_regions = self._detect_pymupdf_tables(pdf_path, page_num, page_blocks)
                table_regions.extend(pymupdf_regions)
                
        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions(table_regions)
        
        logging.info(f"Detected {len(merged_regions)} table regions")
        return merged_regions
    
    def _detect_column_alignment(self, blocks: List['TextBlock'], page_num: int) -> List[TableRegion]:
        """Detect tables by column alignment clustering"""
        if len(blocks) < self.min_column_blocks:
            return []
            
        # Group blocks by their X-coordinates (left alignment)
        x_groups = defaultdict(list)
        for block in blocks:
            # Round X coordinate to handle minor alignment variations
            rounded_x = round(block.x0 / self.alignment_threshold) * self.alignment_threshold
            x_groups[rounded_x].append(block)
            
        # Find groups with multiple blocks (potential columns)
        column_candidates = []
        for x_coord, group_blocks in x_groups.items():
            if len(group_blocks) >= self.min_column_blocks:
                # Sort by Y coordinate
                group_blocks.sort(key=lambda b: b.y0)
                column_candidates.append((x_coord, group_blocks))
                
        if len(column_candidates) < 2:
            return []  # Need at least 2 columns for a table
            
        # Look for aligned columns that form tables
        table_regions = []
        column_candidates.sort(key=lambda x: x[0])  # Sort by X coordinate
        
        # Check for groups of aligned columns
        for i in range(len(column_candidates) - 1):
            current_cols = [column_candidates[i]]
            
            # Find adjacent columns that align vertically
            for j in range(i + 1, len(column_candidates)):
                x_coord, blocks_list = column_candidates[j]
                
                # Check if this column aligns with current group
                if self._columns_are_aligned(current_cols, (x_coord, blocks_list)):
                    current_cols.append((x_coord, blocks_list))
                else:
                    break
                    
            # If we found multiple aligned columns, it's likely a table
            if len(current_cols) >= 2:
                table_region = self._create_table_region_from_columns(current_cols, page_num, "aligned")
                if table_region:
                    table_regions.append(table_region)
                    
        return table_regions
    
    def _detect_row_patterns(self, blocks: List['TextBlock'], page_num: int) -> List[TableRegion]:
        """Detect tables by row pattern recognition"""
        regions = []
        
        # Sort blocks by Y coordinate
        sorted_blocks = sorted(blocks, key=lambda b: b.y0)
        
        # Group blocks into potential rows
        rows = []
        current_row = [sorted_blocks[0]] if sorted_blocks else []
        
        for i in range(1, len(sorted_blocks)):
            prev_block = sorted_blocks[i-1]
            curr_block = sorted_blocks[i]
            
            # If blocks are on similar Y level, they're in the same row
            if abs(curr_block.y0 - prev_block.y0) <= self.alignment_threshold:
                current_row.append(curr_block)
            else:
                if len(current_row) >= 2:  # Row with multiple columns
                    rows.append(current_row)
                current_row = [curr_block]
                
        # Add last row
        if len(current_row) >= 2:
            rows.append(current_row)
            
        # Look for sequences of rows that form tables
        if len(rows) >= self.min_row_blocks:
            # Check for consistent column structure
            table_rows = []
            for i, row in enumerate(rows):
                if i == 0:
                    table_rows.append(row)
                    continue
                    
                # Check if this row has similar column structure
                if self._rows_have_similar_structure(table_rows[-1], row):
                    table_rows.append(row)
                elif len(table_rows) >= self.min_row_blocks:
                    # Found a complete table
                    table_region = self._create_table_region_from_rows(table_rows, page_num, "grid")
                    if table_region:
                        regions.append(table_region)
                    table_rows = [row]
                else:
                    table_rows = [row]
                    
            # Check for final table
            if len(table_rows) >= self.min_row_blocks:
                table_region = self._create_table_region_from_rows(table_rows, page_num, "grid")
                if table_region:
                    regions.append(table_region)
                    
        return regions
    
    def _detect_pymupdf_tables(self, pdf_path: str, page_num: int, 
                              blocks: List['TextBlock']) -> List[TableRegion]:
        """Use PyMuPDF's built-in table detection"""
        regions = []
        
        try:
            doc = fitz.open(pdf_path)
            if page_num < len(doc):
                page = doc[page_num]
                
                # Find tables using PyMuPDF
                tables = page.find_tables()
                
                for table in tables:
                    # Get table bounding box
                    bbox = table.bbox
                    
                    # Find blocks within this table region
                    table_blocks = []
                    for block in blocks:
                        if (bbox[0] <= block.x0 <= bbox[2] and 
                            bbox[1] <= block.y0 <= bbox[3]):
                            table_blocks.append(block)
                            
                    if len(table_blocks) >= self.min_column_blocks:
                        region = TableRegion(
                            page_num=page_num,
                            x0=bbox[0], y0=bbox[1],
                            x1=bbox[2], y1=bbox[3],
                            blocks=table_blocks,
                            columns=self._extract_column_positions(table_blocks),
                            confidence=0.9,  # High confidence for PyMuPDF detection
                            table_type="bordered"
                        )
                        regions.append(region)
                        
            doc.close()
            
        except Exception as e:
            logging.warning(f"PyMuPDF table detection failed: {e}")
            
        return regions
    
    def _columns_are_aligned(self, current_cols: List[Tuple[float, List]], 
                            new_col: Tuple[float, List]) -> bool:
        """Check if columns are vertically aligned (same Y ranges)"""
        new_x, new_blocks = new_col
        
        # Get Y range of new column
        new_y_min = min(b.y0 for b in new_blocks)
        new_y_max = max(b.y1 for b in new_blocks)
        
        # Check overlap with existing columns
        for x_coord, col_blocks in current_cols:
            col_y_min = min(b.y0 for b in col_blocks)
            col_y_max = max(b.y1 for b in col_blocks)
            
            # Check for significant Y overlap
            overlap = min(new_y_max, col_y_max) - max(new_y_min, col_y_min)
            total_range = max(new_y_max, col_y_max) - min(new_y_min, col_y_min)
            
            if total_range > 0 and overlap / total_range >= 0.5:  # 50% overlap
                return True
                
        return False
    
    def _rows_have_similar_structure(self, row1: List['TextBlock'], 
                                   row2: List['TextBlock']) -> bool:
        """Check if two rows have similar column structure"""
        if abs(len(row1) - len(row2)) > 1:  # Allow 1 column difference
            return False
            
        # Sort rows by X coordinate
        row1_sorted = sorted(row1, key=lambda b: b.x0)
        row2_sorted = sorted(row2, key=lambda b: b.x0)
        
        # Check if X positions are similar
        for i in range(min(len(row1_sorted), len(row2_sorted))):
            x_diff = abs(row1_sorted[i].x0 - row2_sorted[i].x0)
            if x_diff > self.alignment_threshold * 2:
                return False
                
        return True
    
    def _create_table_region_from_columns(self, columns: List[Tuple[float, List]], 
                                        page_num: int, table_type: str) -> Optional[TableRegion]:
        """Create table region from column data"""
        all_blocks = []
        column_positions = []
        
        for x_coord, blocks_list in columns:
            all_blocks.extend(blocks_list)
            column_positions.append(x_coord)
            
        if not all_blocks:
            return None
            
        # Calculate bounding box
        x0 = min(b.x0 for b in all_blocks)
        y0 = min(b.y0 for b in all_blocks)
        x1 = max(b.x1 for b in all_blocks)
        y1 = max(b.y1 for b in all_blocks)
        
        # Calculate confidence based on alignment quality
        confidence = self._calculate_alignment_confidence(all_blocks, column_positions)
        
        return TableRegion(
            page_num=page_num,
            x0=x0, y0=y0, x1=x1, y1=y1,
            blocks=all_blocks,
            columns=sorted(column_positions),
            confidence=confidence,
            table_type=table_type
        )
    
    def _create_table_region_from_rows(self, rows: List[List['TextBlock']], 
                                     page_num: int, table_type: str) -> Optional[TableRegion]:
        """Create table region from row data"""
        all_blocks = []
        for row in rows:
            all_blocks.extend(row)
            
        if not all_blocks:
            return None
            
        # Calculate bounding box
        x0 = min(b.x0 for b in all_blocks)
        y0 = min(b.y0 for b in all_blocks)
        x1 = max(b.x1 for b in all_blocks)
        y1 = max(b.y1 for b in all_blocks)
        
        # Extract column positions
        column_positions = self._extract_column_positions(all_blocks)
        
        # Calculate confidence based on row structure consistency
        confidence = self._calculate_row_confidence(rows)
        
        return TableRegion(
            page_num=page_num,
            x0=x0, y0=y0, x1=x1, y1=y1,
            blocks=all_blocks,
            columns=column_positions,
            confidence=confidence,
            table_type=table_type
        )
    
    def _extract_column_positions(self, blocks: List['TextBlock']) -> List[float]:
        """Extract column positions from blocks"""
        x_positions = []
        for block in blocks:
            x_positions.append(block.x0)
            
        # Cluster similar X positions
        x_positions.sort()
        clusters = []
        current_cluster = [x_positions[0]] if x_positions else []
        
        for x in x_positions[1:]:
            if abs(x - current_cluster[-1]) <= self.alignment_threshold:
                current_cluster.append(x)
            else:
                if current_cluster:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [x]
                
        if current_cluster:
            clusters.append(np.mean(current_cluster))
            
        return clusters
    
    def _calculate_alignment_confidence(self, blocks: List['TextBlock'], 
                                      columns: List[float]) -> float:
        """Calculate confidence score for column alignment"""
        if not blocks or not columns:
            return 0.0
            
        aligned_count = 0
        for block in blocks:
            for col_x in columns:
                if abs(block.x0 - col_x) <= self.alignment_threshold:
                    aligned_count += 1
                    break
                    
        return aligned_count / len(blocks)
    
    def _calculate_row_confidence(self, rows: List[List['TextBlock']]) -> float:
        """Calculate confidence score for row structure"""
        if len(rows) < 2:
            return 0.0
            
        # Check consistency of row structure
        row_lengths = [len(row) for row in rows]
        length_variance = np.var(row_lengths) if row_lengths else 1.0
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + length_variance)
        return min(confidence, 1.0)
    
    def _merge_overlapping_regions(self, regions: List[TableRegion]) -> List[TableRegion]:
        """Merge overlapping table regions"""
        if not regions:
            return []
            
        merged = []
        regions_sorted = sorted(regions, key=lambda r: (r.page_num, r.y0, r.x0))
        
        current_region = regions_sorted[0]
        
        for region in regions_sorted[1:]:
            if self._regions_overlap(current_region, region):
                # Merge regions
                current_region = self._merge_regions(current_region, region)
            else:
                merged.append(current_region)
                current_region = region
                
        merged.append(current_region)
        return merged
    
    def _regions_overlap(self, region1: TableRegion, region2: TableRegion) -> bool:
        """Check if two table regions overlap"""
        if region1.page_num != region2.page_num:
            return False
            
        # Check bounding box overlap
        x_overlap = (min(region1.x1, region2.x1) - max(region1.x0, region2.x0)) > 0
        y_overlap = (min(region1.y1, region2.y1) - max(region1.y0, region2.y0)) > 0
        
        return x_overlap and y_overlap
    
    def _merge_regions(self, region1: TableRegion, region2: TableRegion) -> TableRegion:
        """Merge two overlapping table regions"""
        # Create a list of unique blocks (can't use set with TextBlock objects)
        merged_blocks = []
        all_blocks = region1.blocks + region2.blocks
        seen_blocks = []
        for block in all_blocks:
            # Check if block already exists (by comparing position and text)
            is_duplicate = False
            for seen_block in seen_blocks:
                if (abs(block.x0 - seen_block.x0) < 1.0 and 
                    abs(block.y0 - seen_block.y0) < 1.0 and 
                    block.text == seen_block.text):
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged_blocks.append(block)
                seen_blocks.append(block)
                
        merged_columns = sorted(set(region1.columns + region2.columns))
        
        return TableRegion(
            page_num=region1.page_num,
            x0=min(region1.x0, region2.x0),
            y0=min(region1.y0, region2.y0),
            x1=max(region1.x1, region2.x1),
            y1=max(region1.y1, region2.y1),
            blocks=merged_blocks,
            columns=merged_columns,
            confidence=max(region1.confidence, region2.confidence),
            table_type=region1.table_type if region1.confidence >= region2.confidence else region2.table_type
        )


class MetadataDetector:
    """Detects metadata strings (dates, emails, URLs) using layout context"""
    
    def __init__(self, page_edge_threshold: float = 30.0, 
                 font_size_threshold: float = 8.0):
        self.page_edge_threshold = page_edge_threshold
        self.font_size_threshold = font_size_threshold
        
        # Metadata patterns
        self.date_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'\b(?:https?://|www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?\b', re.IGNORECASE)
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        
        # Additional metadata patterns
        self.version_pattern = re.compile(r'\b(?:version|ver?\.?|v\.?)\s*\d+(?:\.\d+)*\b', re.IGNORECASE)
        self.toc_pattern = re.compile(r'^[^.]*\.{3,}\s*\d+\s*$')  # Table of contents lines with dots and page numbers
        self.page_ref_pattern = re.compile(r'\b(?:page|p\.?)\s*\d+\b', re.IGNORECASE)
        self.copyright_pattern = re.compile(r'©|\bcopyright\b|\ball rights reserved\b', re.IGNORECASE)
        self.timestamp_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b', re.IGNORECASE)
        
    def is_metadata_candidate(self, candidate: HeadingCandidate, page_blocks: List['TextBlock']) -> bool:
        """Check if candidate is likely metadata based on layout context"""
        text = candidate.text.strip()
        block = candidate.block
        
        # Check for metadata patterns
        has_metadata_pattern = any([
            self.date_pattern.search(text),
            self.email_pattern.search(text),
            self.url_pattern.search(text),
            self.phone_pattern.search(text),
            self.version_pattern.search(text),
            self.toc_pattern.search(text),
            self.page_ref_pattern.search(text),
            self.copyright_pattern.search(text),
            self.timestamp_pattern.search(text)
        ])
        
        if not has_metadata_pattern:
            return False
            
        # Analyze layout context
        page_bounds = self._get_page_bounds(page_blocks)
        if not page_bounds:
            return False
            
        min_x, min_y, max_x, max_y = page_bounds
        page_width = max_x - min_x
        page_height = max_y - min_y
        
        # Check if block is at page edges (header/footer)
        is_at_top = block.y0 <= (min_y + self.page_edge_threshold)
        is_at_bottom = block.y1 >= (max_y - self.page_edge_threshold)
        is_at_left = block.x0 <= (min_x + self.page_edge_threshold)
        is_at_right = block.x1 >= (max_x - self.page_edge_threshold)
        
        # Check font size (metadata typically uses smaller fonts)
        is_small_font = block.font_size <= self.font_size_threshold
        
        # Calculate distance from center
        block_center_x = (block.x0 + block.x1) / 2
        block_center_y = (block.y0 + block.y1) / 2
        page_center_x = (min_x + max_x) / 2
        page_center_y = (min_y + max_y) / 2
        
        center_distance_x = abs(block_center_x - page_center_x) / (page_width / 2)
        center_distance_y = abs(block_center_y - page_center_y) / (page_height / 2)
        
        # Enhanced layout-based metadata indicators
        metadata_indicators = [
            is_at_top or is_at_bottom,  # At page top/bottom
            is_at_left or is_at_right,  # At page edges
            is_small_font,              # Small font size
            center_distance_x > 0.6,    # Far from horizontal center
            center_distance_y > 0.6,    # Far from vertical center
            self._is_isolated_block(block, page_blocks),  # Isolated from main content
            self._has_metadata_neighbors(block, page_blocks),  # Near other metadata
            self._is_in_header_footer_region(block, page_blocks),  # Clearly in header/footer
            self._has_metadata_formatting(text),  # Looks like metadata formatting
            self._is_short_metadata_string(text, is_small_font)  # Short metadata-like strings
        ]
        
        # If 3+ indicators suggest metadata, it's likely not a heading
        # But for very obvious metadata patterns, be more aggressive
        metadata_score = sum(metadata_indicators)
        
        # Lower threshold for obvious metadata patterns
        obvious_metadata_patterns = [
            self.version_pattern.search(text),
            self.toc_pattern.search(text),
            self.page_ref_pattern.search(text),
            bool(re.match(r'^\d+\.\d+$', text.strip())),  # Pure version numbers
            bool(re.match(r'^[A-Z][a-z]+ \d{1,2}, \d{4}$', text.strip()))  # Date format
        ]
        
        if any(obvious_metadata_patterns):
            return metadata_score >= 2  # More aggressive for obvious metadata
        else:
            return metadata_score >= 4  # Conservative for other patterns
        
    def _get_page_bounds(self, blocks: List['TextBlock']) -> Optional[Tuple[float, float, float, float]]:
        """Get the bounding box of all blocks on the page"""
        if not blocks:
            return None
            
        x_coords = [block.x0 for block in blocks] + [block.x1 for block in blocks]
        y_coords = [block.y0 for block in blocks] + [block.y1 for block in blocks]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
    def _is_isolated_block(self, target_block: 'TextBlock', page_blocks: List['TextBlock']) -> bool:
        """Check if block is isolated from main content flow"""
        nearby_blocks = []
        isolation_threshold = 30.0  # Points
        
        for block in page_blocks:
            if block == target_block:
                continue
                
            # Calculate minimum distance to target block
            min_distance = min([
                abs(block.x1 - target_block.x0),  # Left
                abs(target_block.x1 - block.x0),  # Right
                abs(block.y1 - target_block.y0),  # Above
                abs(target_block.y1 - block.y0)   # Below
            ])
            
            if min_distance <= isolation_threshold:
                nearby_blocks.append(block)
                
        # If few nearby blocks, it's isolated
        return len(nearby_blocks) <= 2
        
    def _has_metadata_neighbors(self, target_block: 'TextBlock', page_blocks: List['TextBlock']) -> bool:
        """Check if block has other metadata-like blocks nearby"""
        neighbor_threshold = 40.0  # Points
        metadata_neighbors = 0
        
        for block in page_blocks:
            if block == target_block:
                continue
                
            # Check if blocks are close
            horizontal_distance = min(abs(block.x1 - target_block.x0), abs(target_block.x1 - block.x0))
            vertical_distance = min(abs(block.y1 - target_block.y0), abs(target_block.y1 - block.y0))
            
            if horizontal_distance <= neighbor_threshold or vertical_distance <= neighbor_threshold:
                # Check if neighbor also looks like metadata
                neighbor_text = block.text.strip()
                if any([
                    self.date_pattern.search(neighbor_text),
                    self.email_pattern.search(neighbor_text),
                    self.url_pattern.search(neighbor_text),
                    self.phone_pattern.search(neighbor_text),
                    self.version_pattern.search(neighbor_text),
                    self.toc_pattern.search(neighbor_text),
                    self.page_ref_pattern.search(neighbor_text),
                    self.copyright_pattern.search(neighbor_text),
                    self.timestamp_pattern.search(neighbor_text),
                    len(neighbor_text) < 20 and any(char.isdigit() for char in neighbor_text)
                ]):
                    metadata_neighbors += 1
                    
        return metadata_neighbors >= 1
    
    def _is_in_header_footer_region(self, target_block: 'TextBlock', page_blocks: List['TextBlock']) -> bool:
        """Check if block is in a clear header or footer region"""
        if not page_blocks:
            return False
            
        # Get Y boundaries
        y_coords = [block.y0 for block in page_blocks] + [block.y1 for block in page_blocks]
        min_y, max_y = min(y_coords), max(y_coords)
        page_height = max_y - min_y
        
        # Define header/footer regions (top/bottom 10% of page)
        header_region = min_y + (page_height * 0.1)
        footer_region = max_y - (page_height * 0.1)
        
        # Check if block is in these regions AND has few nearby blocks
        in_header = target_block.y0 <= header_region
        in_footer = target_block.y1 >= footer_region
        
        if in_header or in_footer:
            # Count blocks in same region
            region_blocks = []
            for block in page_blocks:
                if in_header and block.y0 <= header_region:
                    region_blocks.append(block)
                elif in_footer and block.y1 >= footer_region:
                    region_blocks.append(block)
                    
            # If few blocks in region, likely header/footer
            return len(region_blocks) <= 3
            
        return False
    
    def _has_metadata_formatting(self, text: str) -> bool:
        """Check if text has formatting typical of metadata"""
        # Common metadata formatting patterns
        metadata_patterns = [
            r'^\d+\.\d+$',  # Version numbers like "1.0"
            r'^[A-Z][a-z]+ \d{1,2}, \d{4}$',  # Date format "March 21, 2003"
            r'.*\.{3,}.*\d+\s*$',  # TOC pattern with dots
            r'^(Rev|Version|Ver\.?|V\.?)\s*\d+',  # Revision/version prefixes
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Date format
            r'^Page \d+',  # Page references
            r'^\d+$'  # Just numbers
        ]
        
        return any(re.match(pattern, text.strip(), re.IGNORECASE) for pattern in metadata_patterns)
    
    def _is_short_metadata_string(self, text: str, is_small_font: bool) -> bool:
        """Check if text is a short string that looks like metadata"""
        text = text.strip()
        
        # Very short strings with small fonts are often metadata
        if len(text) <= 10 and is_small_font:
            # Common short metadata patterns
            short_metadata_indicators = [
                text.isdigit(),  # Just numbers
                bool(re.match(r'^v?\d+(\.\d+)*$', text, re.IGNORECASE)),  # Version numbers
                bool(re.match(r'^\d{1,2}/\d{1,2}(/\d{2,4})?$', text)),  # Dates
                text.lower() in ['draft', 'final', 'revised', 'updated', 'confidential'],
                bool(re.match(r'^[A-Z]{2,}$', text)) and len(text) <= 5  # Short all-caps codes
            ]
            return any(short_metadata_indicators)
            
        return False


class ParagraphAnalyzer:
    """Analyzes paragraph context to filter inline bold text"""
    
    def __init__(self, inline_proximity_threshold: float = 8.0,
                 line_height_threshold: float = 25.0):
        self.inline_proximity_threshold = inline_proximity_threshold
        self.line_height_threshold = line_height_threshold
        
    def group_blocks_into_paragraphs(self, blocks: List['TextBlock']) -> List[List['TextBlock']]:
        """Group text blocks into paragraph-level chunks"""
        if not blocks:
            return []
            
        # Sort blocks by page, then by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_num, b.y0, b.x0))
        
        paragraphs = []
        current_paragraph = [sorted_blocks[0]]
        
        for i in range(1, len(sorted_blocks)):
            prev_block = sorted_blocks[i-1]
            curr_block = sorted_blocks[i]
            
            # Check if blocks belong to same paragraph
            if self._are_in_same_paragraph(prev_block, curr_block):
                current_paragraph.append(curr_block)
            else:
                # Start new paragraph
                paragraphs.append(current_paragraph)
                current_paragraph = [curr_block]
                
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        return paragraphs
    
    def _are_in_same_paragraph(self, block1: 'TextBlock', block2: 'TextBlock') -> bool:
        """Check if two blocks belong to the same paragraph"""
        # Different pages = different paragraphs
        if block1.page_num != block2.page_num:
            return False
            
        # Calculate vertical gap
        vertical_gap = abs(block2.y0 - block1.y1)
        
        # If blocks are very close vertically, they're likely same paragraph
        if vertical_gap <= self.inline_proximity_threshold:
            return True
            
        # Check if they're on the same line (similar Y positions)
        line_gap = abs(block2.y0 - block1.y0)
        if line_gap <= self.line_height_threshold:
            return True
            
        return False
    
    def is_inline_bold_text(self, candidate: HeadingCandidate, 
                           paragraph_blocks: List['TextBlock']) -> bool:
        """Check if a bold candidate is inline emphasis rather than a heading"""
        if not candidate.is_bold:
            return False
            
        candidate_block = candidate.block
        
        # Find blocks that are adjacent to this candidate
        adjacent_blocks = self._find_adjacent_blocks(candidate_block, paragraph_blocks)
        
        if not adjacent_blocks:
            return False  # No adjacent blocks, likely a standalone heading
            
        # Check various inline indicators
        inline_indicators = [
            self._has_adjacent_non_bold_text(candidate_block, adjacent_blocks),
            self._has_punctuation_continuity(candidate, adjacent_blocks),
            self._has_same_line_continuation(candidate_block, adjacent_blocks),
            self._is_similar_font_size(candidate_block, adjacent_blocks),
            self._lacks_heading_spacing(candidate_block, adjacent_blocks)
        ]
        
        # If multiple indicators suggest inline text, filter it out
        inline_count = sum(inline_indicators)
        return inline_count >= 3  # At least 3 indicators needed (more conservative)
    
    def _find_adjacent_blocks(self, target_block: 'TextBlock', 
                             paragraph_blocks: List['TextBlock']) -> List['TextBlock']:
        """Find blocks that are horizontally or vertically adjacent"""
        adjacent = []
        
        for block in paragraph_blocks:
            if block == target_block:
                continue
                
            # Check horizontal adjacency (same line)
            if abs(block.y0 - target_block.y0) <= self.line_height_threshold / 2:
                horizontal_gap = min(abs(block.x0 - target_block.x1), 
                                   abs(target_block.x0 - block.x1))
                if horizontal_gap <= self.inline_proximity_threshold * 2:
                    adjacent.append(block)
                    
            # Check vertical adjacency (consecutive lines)
            elif abs(block.y0 - target_block.y1) <= self.line_height_threshold:
                adjacent.append(block)
                
        return adjacent
    
    def _has_adjacent_non_bold_text(self, target_block: 'TextBlock', 
                                   adjacent_blocks: List['TextBlock']) -> bool:
        """Check if there's non-bold text immediately adjacent"""
        for block in adjacent_blocks:
            if not block.is_bold:
                return True
        return False
    
    def _has_punctuation_continuity(self, candidate: HeadingCandidate, 
                                   adjacent_blocks: List['TextBlock']) -> bool:
        """Check if text suggests continuation (punctuation, flow)"""
        text = candidate.text.strip()
        
        # Check if candidate ends with continuation punctuation
        if text.endswith((',', ':', ';', '-')):
            return True
            
        # Check if adjacent blocks start with continuation words
        continuation_words = ['and', 'or', 'but', 'then', 'also', 'however', 'therefore']
        for block in adjacent_blocks:
            block_text = block.text.strip().lower()
            if any(block_text.startswith(word) for word in continuation_words):
                return True
                
        return False
    
    def _has_same_line_continuation(self, target_block: 'TextBlock', 
                                   adjacent_blocks: List['TextBlock']) -> bool:
        """Check if blocks are on the same line with text flow"""
        same_line_blocks = [
            block for block in adjacent_blocks 
            if abs(block.y0 - target_block.y0) <= self.line_height_threshold / 4
        ]
        
        return len(same_line_blocks) > 0
    
    def _is_similar_font_size(self, target_block: 'TextBlock', 
                             adjacent_blocks: List['TextBlock']) -> bool:
        """Check if font size is similar to surrounding text"""
        for block in adjacent_blocks:
            size_ratio = target_block.font_size / block.font_size if block.font_size > 0 else 1.0
            # If sizes are very similar (within 10%), likely inline emphasis
            if 0.9 <= size_ratio <= 1.1:
                return True
        return False
    
    def _lacks_heading_spacing(self, target_block: 'TextBlock', 
                              adjacent_blocks: List['TextBlock']) -> bool:
        """Check if block lacks typical heading spacing"""
        # True headings usually have more spacing above/below
        min_gaps = []
        for block in adjacent_blocks:
            vertical_gap = min(abs(block.y0 - target_block.y1), 
                             abs(target_block.y0 - block.y1))
            min_gaps.append(vertical_gap)
            
        if min_gaps:
            min_gap = min(min_gaps)
            # If very close to other text, likely inline
            return min_gap <= self.inline_proximity_threshold
            
        return False


class HeadingDetector:
    """Main heading detection engine"""
    
    def __init__(self, 
                 size_threshold_ratio: float = 1.05,
                 max_heading_words: int = 25,
                 min_vertical_gap: float = 5.0):
        self.size_threshold_ratio = size_threshold_ratio
        self.max_heading_words = max_heading_words
        self.min_vertical_gap = min_vertical_gap
        self.font_analyzer = FontAnalyzer()
        self.paragraph_analyzer = ParagraphAnalyzer()
        self.table_detector = TableDetector()
        self.column_detector = MultiColumnDetector()
        self.metadata_detector = MetadataDetector()
        
    def detect_candidates(self, blocks: List['TextBlock'], pdf_path: Optional[str] = None) -> List[HeadingCandidate]:
        """Detect all heading candidates from text blocks"""
        if not blocks:
            return []
            
        # Detect table regions first to avoid misclassifying table content
        table_regions = self.table_detector.detect_table_regions(blocks, pdf_path)
        
        # Filter out blocks that are inside table regions
        non_table_blocks = self._filter_table_blocks(blocks, table_regions)
        logging.info(f"Filtered {len(blocks) - len(non_table_blocks)} blocks from {len(table_regions)} table regions")
        
        # Detect column layouts to handle multi-column pages
        column_layouts = self.column_detector.detect_column_layouts(non_table_blocks)
        multi_column_pages = [layout for layout in column_layouts if layout.num_columns > 1]
        if multi_column_pages:
            logging.info(f"Detected multi-column layouts on {len(multi_column_pages)} pages")
        
        # Analyze fonts across document (using non-table blocks)
        font_stats = self.font_analyzer.analyze_document_fonts(non_table_blocks)
        common_fonts = [f[0] for f in font_stats['common_fonts'][:3]]
        
        logging.info(f"Body text size detected: {font_stats['body_text_size']}")
        
        # Group blocks into paragraphs for context analysis
        paragraphs = self.paragraph_analyzer.group_blocks_into_paragraphs(non_table_blocks)
        logging.info(f"Grouped {len(non_table_blocks)} blocks into {len(paragraphs)} paragraphs")
        
        # Calculate vertical gaps (column-aware)
        blocks_by_page = defaultdict(list)
        for block in non_table_blocks:
            blocks_by_page[block.page_num].append(block)
            
        vertical_gaps = self._calculate_vertical_gaps_column_aware(blocks_by_page, non_table_blocks, column_layouts)
        
        # Process each block (column-aware)
        candidates = []
        for i, block in enumerate(non_table_blocks):
            candidate = self._analyze_block_column_aware(block, common_fonts, vertical_gaps.get(i, 0), column_layouts)
            if candidate and self._is_valid_candidate(candidate):
                candidates.append(candidate)
                
        # Apply column-aware filtering
        column_filtered_candidates = self._filter_column_break_false_positives(candidates, column_layouts)
        logging.info(f"Filtered {len(candidates) - len(column_filtered_candidates)} column break false positives")
        
        # Apply inline bold text filtering
        filtered_candidates = self._filter_inline_bold_text(column_filtered_candidates, paragraphs)
        logging.info(f"Filtered {len(column_filtered_candidates) - len(filtered_candidates)} inline bold candidates")
        
        # Apply additional table-aware filtering
        table_filtered_candidates = self._filter_table_headers(filtered_candidates, table_regions)
        logging.info(f"Filtered {len(filtered_candidates) - len(table_filtered_candidates)} table header candidates")
        
        # Apply metadata filtering
        metadata_filtered_candidates = self._filter_metadata_candidates(table_filtered_candidates, blocks_by_page)
        logging.info(f"Filtered {len(table_filtered_candidates) - len(metadata_filtered_candidates)} metadata candidates")
        
        # Calculate confidence scores (column-aware)
        self._calculate_confidence_scores_column_aware(metadata_filtered_candidates, column_layouts)
        
        return metadata_filtered_candidates
    
    def _calculate_vertical_gaps_column_aware(self, blocks_by_page: Dict[int, List], 
                                            blocks: List['TextBlock'],
                                            column_layouts: List[ColumnLayout]) -> Dict[int, float]:
        """Calculate vertical gaps with column-awareness"""
        gaps = {}
        
        # Create lookup for column layouts
        layout_by_page = {layout.page_num: layout for layout in column_layouts}
        
        for page_num, page_blocks in blocks_by_page.items():
            layout = layout_by_page.get(page_num)
            
            if layout and layout.num_columns > 1:
                # Multi-column: calculate gaps within each column
                gaps.update(self._calculate_gaps_within_columns(page_blocks, layout, blocks))
            else:
                # Single column: use regular gap calculation
                sorted_blocks = sorted(page_blocks, key=lambda b: (b.y0, b.x0))
                
                for i in range(1, len(sorted_blocks)):
                    prev_block = sorted_blocks[i-1]
                    curr_block = sorted_blocks[i]
                    
                    gap = curr_block.y0 - prev_block.y1
                    
                    # Find index in original list
                    for j, block in enumerate(blocks):
                        if block == curr_block:
                            gaps[j] = gap
                            break
                            
        return gaps
    
    def _calculate_gaps_within_columns(self, page_blocks: List['TextBlock'], 
                                     layout: ColumnLayout,
                                     all_blocks: List['TextBlock']) -> Dict[int, float]:
        """Calculate vertical gaps within individual columns"""
        gaps = {}
        
        # Group blocks by column
        blocks_by_column = defaultdict(list)
        for block in page_blocks:
            column_idx = layout.get_column_for_block(block)
            blocks_by_column[column_idx].append(block)
            
        # Calculate gaps within each column
        for column_idx, column_blocks in blocks_by_column.items():
            sorted_blocks = sorted(column_blocks, key=lambda b: b.y0)
            
            for i in range(1, len(sorted_blocks)):
                prev_block = sorted_blocks[i-1]
                curr_block = sorted_blocks[i]
                
                gap = curr_block.y0 - prev_block.y1
                
                # Find index in original list
                for j, block in enumerate(all_blocks):
                    if block == curr_block:
                        gaps[j] = gap
                        break
                        
        return gaps
    
    def _analyze_block_column_aware(self, block: 'TextBlock', common_fonts: List[str], 
                                   vertical_gap: float, column_layouts: List[ColumnLayout]) -> Optional[HeadingCandidate]:
        """Analyze a block with column layout awareness"""
        # Start with regular analysis
        candidate = self._analyze_block(block, common_fonts, vertical_gap)
        
        if not candidate:
            return None
            
        # Find column layout for this block's page
        layout = None
        for col_layout in column_layouts:
            if col_layout.page_num == block.page_num:
                layout = col_layout
                break
                
        if layout and layout.num_columns > 1:
            # Add column-specific analysis
            candidate = self._enhance_candidate_with_column_info(candidate, layout)
            
        return candidate
    
    def _enhance_candidate_with_column_info(self, candidate: HeadingCandidate, 
                                          layout: ColumnLayout) -> HeadingCandidate:
        """Enhance candidate with column-specific information"""
        block = candidate.block
        
        # Determine which column this candidate belongs to
        column_idx = layout.get_column_for_block(block)
        
        # Check if heading spans multiple columns (full-width heading)
        block_width = block.x1 - block.x0
        total_page_width = layout.column_regions[-1][2] - layout.column_regions[0][0]
        
        spans_columns = block_width / total_page_width >= 0.7  # Spans at least 70% of page width
        
        # Check if heading is centered across columns
        block_center = (block.x0 + block.x1) / 2
        page_center = (layout.column_regions[0][0] + layout.column_regions[-1][2]) / 2
        is_centered = abs(block_center - page_center) <= 20  # Within 20 points of center
        
        # Adjust confidence based on column characteristics
        if spans_columns and is_centered:
            # Full-width centered headings are very likely to be section headings
            candidate.confidence_score += 15
        elif spans_columns:
            # Full-width headings are likely section headings
            candidate.confidence_score += 10
        else:
            # Single-column headings might be subsection headings
            candidate.confidence_score += 5
            
        return candidate
    
    def _filter_column_break_false_positives(self, candidates: List[HeadingCandidate], 
                                           column_layouts: List[ColumnLayout]) -> List[HeadingCandidate]:
        """Filter out false positives caused by column breaks"""
        if not column_layouts:
            return candidates
            
        filtered = []
        layout_by_page = {layout.page_num: layout for layout in column_layouts}
        
        for candidate in candidates:
            layout = layout_by_page.get(candidate.page_num)
            
            if layout and layout.num_columns > 1:
                if not self._is_column_break_false_positive(candidate, layout):
                    filtered.append(candidate)
                else:
                    logging.debug(f"Filtered column break false positive: '{candidate.text[:30]}...'")
            else:
                # Single column page, keep candidate
                filtered.append(candidate)
                
        return filtered
    
    def _is_column_break_false_positive(self, candidate: HeadingCandidate, 
                                      layout: ColumnLayout) -> bool:
        """Check if candidate is a false positive due to column break"""
        block = candidate.block
        column_idx = layout.get_column_for_block(block)
        
        if column_idx >= len(layout.column_regions):
            return False
            
        column_region = layout.column_regions[column_idx]
        
        # Check if block is at the top of a column (common false positive)
        column_top_threshold = column_region[1] + 30  # Within 30 points of column top (reduced)
        
        if block.y0 <= column_top_threshold:
            # This might be the start of column content, not a section heading
            
            # Additional checks:
            # 1. Does the text look like continuation of previous column?
            if self._looks_like_continuation_text(candidate):
                return True
                
            # 2. Is the text width similar to single column width?
            block_width = block.x1 - block.x0
            column_width = column_region[2] - column_region[0]
            
            if 0.7 <= block_width / column_width <= 1.3:  # Similar to column width (more lenient)
                # Check if there's no extra spacing around it
                if candidate.vertical_gap_before <= 10:  # Normal line spacing (more lenient)
                    return True
                    
        return False
    
    def _looks_like_continuation_text(self, candidate: HeadingCandidate) -> bool:
        """Check if text looks like continuation from previous column"""
        text = candidate.text.lower().strip()
        
        # Check for continuation indicators
        continuation_patterns = [
            r'^[a-z]',  # Starts with lowercase (middle of sentence)
            r'^\d+\.',  # Numbered list continuation
            r'^[a-z]\)',  # Lettered list continuation
            r'^and\b', r'^or\b', r'^but\b',  # Conjunction starters
            r'^however\b', r'^therefore\b', r'^furthermore\b',  # Transition words
        ]
        
        for pattern in continuation_patterns:
            if re.match(pattern, text):
                return True
                
        return False
    
    def _calculate_confidence_scores_column_aware(self, candidates: List[HeadingCandidate],
                                                column_layouts: List[ColumnLayout]):
        """Calculate confidence scores with column layout awareness"""
        # Start with regular confidence calculation
        self._calculate_confidence_scores(candidates)
        
        # Apply column-specific adjustments
        layout_by_page = {layout.page_num: layout for layout in column_layouts}
        
        for candidate in candidates:
            layout = layout_by_page.get(candidate.page_num)
            
            if layout and layout.num_columns > 1:
                # Apply column-specific confidence adjustments
                self._apply_column_confidence_adjustments(candidate, layout)
    
    def _apply_column_confidence_adjustments(self, candidate: HeadingCandidate, 
                                           layout: ColumnLayout):
        """Apply column-specific confidence score adjustments"""
        block = candidate.block
        
        # Check heading span and position
        block_width = block.x1 - block.x0
        total_page_width = layout.column_regions[-1][2] - layout.column_regions[0][0]
        span_ratio = block_width / total_page_width
        
        # Full-width headings get bonus
        if span_ratio >= 0.7:
            candidate.confidence_score += 12
            
        # Centered headings get bonus
        block_center = (block.x0 + block.x1) / 2
        page_center = (layout.column_regions[0][0] + layout.column_regions[-1][2]) / 2
        
        if abs(block_center - page_center) <= 20:
            candidate.confidence_score += 8
            
        # Check for column gutter alignment (headings often align with gutters)
        for boundary in layout.column_boundaries:
            if abs(block.x0 - boundary) <= 10 or abs(block.x1 - boundary) <= 10:
                candidate.confidence_score += 5
                break
                
        # Ensure score doesn't exceed reasonable bounds
        candidate.confidence_score = min(candidate.confidence_score, 100.0)
    
    def _filter_table_blocks(self, blocks: List['TextBlock'], 
                           table_regions: List[TableRegion]) -> List['TextBlock']:
        """Filter out blocks that are inside table regions"""
        if not table_regions:
            return blocks
            
        filtered_blocks = []
        for block in blocks:
            is_in_table = False
            for table_region in table_regions:
                if table_region.contains_block(block):
                    is_in_table = True
                    break
                    
            if not is_in_table:
                filtered_blocks.append(block)
                
        return filtered_blocks
    
    def _filter_table_headers(self, candidates: List[HeadingCandidate], 
                            table_regions: List[TableRegion]) -> List[HeadingCandidate]:
        """Filter out candidates that are likely table headers"""
        if not table_regions:
            return candidates
            
        filtered = []
        for candidate in candidates:
            is_table_header = False
            
            # Check if candidate is near a table region (potential header)
            for table_region in table_regions:
                if self._is_likely_table_header(candidate, table_region):
                    is_table_header = True
                    logging.debug(f"Filtered table header: '{candidate.text[:30]}...'")
                    break
                    
            if not is_table_header:
                filtered.append(candidate)
                
        return filtered
    
    def _is_likely_table_header(self, candidate: HeadingCandidate, 
                              table_region: TableRegion) -> bool:
        """Check if a candidate is likely a table header"""
        block = candidate.block
        
        # Check if block is just above the table region
        vertical_distance = table_region.y0 - block.y1
        if 0 <= vertical_distance <= 30:  # Within 30 points above table
            # Check if block spans across table columns
            block_width = block.x1 - block.x0
            table_width = table_region.x1 - table_region.x0
            
            # If block width is similar to table width, likely a table title
            if block_width / table_width >= 0.7:
                return True
                
            # Check if block aligns with table columns
            for col_x in table_region.columns:
                if abs(block.x0 - col_x) <= 10:  # Within 10 points of column
                    # Check if there are multiple rows below with similar structure
                    if self._has_tabular_rows_below(candidate, table_region):
                        return True
                        
        return False
    
    def _has_tabular_rows_below(self, candidate: HeadingCandidate, 
                              table_region: TableRegion) -> bool:
        """Check if there are tabular rows below the candidate"""
        # Look for blocks in the table region that are below the candidate
        candidate_y = candidate.block.y1
        rows_below = []
        
        for block in table_region.blocks:
            if block.y0 > candidate_y:
                rows_below.append(block)
                
        # If there are multiple blocks below in a grid-like pattern
        if len(rows_below) >= 4:  # At least 2 rows with 2 columns each
            # Group by Y coordinate to find rows
            rows_by_y = defaultdict(list)
            for block in rows_below:
                rounded_y = round(block.y0 / 5) * 5  # Group by 5-point intervals
                rows_by_y[rounded_y].append(block)
                
            # If we have multiple rows with similar structure
            multi_column_rows = [row for row in rows_by_y.values() if len(row) >= 2]
            return len(multi_column_rows) >= 2
            
        return False
    
    def _filter_metadata_candidates(self, candidates: List[HeadingCandidate], 
                                   blocks_by_page: Dict[int, List['TextBlock']]) -> List[HeadingCandidate]:
        """Filter out candidates that are likely metadata (dates, emails, URLs) based on layout"""
        filtered = []
        
        for candidate in candidates:
            page_blocks = blocks_by_page.get(candidate.page_num, [])
            
            if self.metadata_detector.is_metadata_candidate(candidate, page_blocks):
                logging.debug(f"Filtered metadata candidate: '{candidate.text[:30]}...'")
                continue
                
            filtered.append(candidate)
            
        return filtered
    
    def _filter_inline_bold_text(self, candidates: List[HeadingCandidate], 
                                paragraphs: List[List['TextBlock']]) -> List[HeadingCandidate]:
        """Filter out inline bold text that should not be treated as headings"""
        filtered = []
        
        for candidate in candidates:
            # Find which paragraph this candidate belongs to
            candidate_paragraph = None
            for paragraph in paragraphs:
                if candidate.block in paragraph:
                    candidate_paragraph = paragraph
                    break
                    
            if candidate_paragraph is None:
                # If we can't find the paragraph, keep the candidate
                filtered.append(candidate)
                continue
                
            # Check if this is inline bold text
            if self.paragraph_analyzer.is_inline_bold_text(candidate, candidate_paragraph):
                logging.debug(f"Filtered inline bold text: '{candidate.text[:30]}...'")
                continue
                
            # Additional checks for text fragmentation
            if self._is_likely_fragmented_text(candidate, candidate_paragraph):
                logging.debug(f"Filtered fragmented text: '{candidate.text[:30]}...'")
                continue
                
            filtered.append(candidate)
            
        return filtered
    
    def _is_likely_fragmented_text(self, candidate: HeadingCandidate, 
                                  paragraph_blocks: List['TextBlock']) -> bool:
        """Check if candidate is part of fragmented text that should be merged"""
        # Only filter extremely short text (1 character) as fragmentation
        if len(candidate.text.strip()) <= 1:
            # Check if there are similar blocks nearby that could be part of same heading
            similar_blocks = []
            for block in paragraph_blocks:
                if block == candidate.block:
                    continue
                    
                # Same line and similar styling
                if (abs(block.y0 - candidate.block.y0) <= 5 and
                    block.is_bold == candidate.is_bold and
                    abs(block.font_size - candidate.font_size) <= 1):
                    similar_blocks.append(block)
                    
            # If there are multiple similar blocks, likely fragmentation
            return len(similar_blocks) >= 3  # Require more evidence
            
        return False
        
    def _analyze_block(self, block: 'TextBlock', common_fonts: List[str], 
                      vertical_gap: float) -> Optional[HeadingCandidate]:
        """Analyze a single block for heading characteristics"""
        text = block.text.strip()
        if not text:
            return None
            
        # Create candidate
        candidate = HeadingCandidate(
            text=text,
            block=block,
            page_num=block.page_num,
            font_size=block.font_size,
            word_count=len(text.split())
        )
        
        # Font size analysis
        candidate.is_larger_than_body = self.font_analyzer.is_larger_than_body(
            block.font_size, self.size_threshold_ratio
        )
        candidate.size_ratio = self.font_analyzer.get_size_ratio(block.font_size)
        
        # Font style analysis
        candidate.is_bold = block.is_bold
        candidate.is_italic = block.is_italic
        candidate.has_distinct_font = self.font_analyzer.is_distinct_font(
            block.font_name, common_fonts
        )
        
        # Numbering detection
        has_num, pattern, depth = HeadingPatterns.detect_numbering(text)
        candidate.has_numbering = has_num
        candidate.numbering_pattern = pattern
        candidate.numbering_depth = depth
        
        # Text case analysis
        candidate.is_uppercase = text.isupper() and len(text) > 2
        candidate.is_titlecase = self._is_titlecase(text)
        
        # Spacing analysis
        candidate.vertical_gap_before = vertical_gap
        candidate.has_spacing_cue = vertical_gap > self.min_vertical_gap
        
        return candidate
        
    def _calculate_vertical_gaps(self, blocks_by_page: Dict[int, List], blocks: List) -> Dict[int, float]:
        """Calculate vertical gaps between consecutive blocks"""
        gaps = {}
        
        for page_num, page_blocks in blocks_by_page.items():
            sorted_blocks = sorted(page_blocks, key=lambda b: (b.y0, b.x0))
            
            for i in range(1, len(sorted_blocks)):
                prev_block = sorted_blocks[i-1]
                curr_block = sorted_blocks[i]
                
                # Calculate gap between bottom of previous and top of current
                gap = curr_block.y0 - prev_block.y1
                
                # Find index in original list
                for j, block in enumerate(blocks):
                    if block == curr_block:
                        gaps[j] = gap
                        break
                        
        return gaps
        
    def _is_titlecase(self, text: str) -> bool:
        """Check if text is in title case"""
        words = text.split()
        if not words:
            return False
            
        # Check if significant words start with uppercase
        significant_words = [w for w in words if len(w) > 3]
        if not significant_words:
            significant_words = words
            
        uppercase_count = sum(1 for w in significant_words if w[0].isupper())
        return uppercase_count >= len(significant_words) * 0.7
        
    def _is_valid_candidate(self, candidate: HeadingCandidate) -> bool:
        """Apply filtering rules to validate candidates"""
        # Expanded criteria - must have at least one indicator
        strong_indicators = [
            candidate.has_numbering,
            candidate.is_larger_than_body and candidate.size_ratio > 1.1,  # Lowered from 1.2
            candidate.is_bold and candidate.has_distinct_font,
            candidate.is_uppercase and candidate.word_count <= 15,  # Increased from 10
            candidate.has_spacing_cue and candidate.is_larger_than_body,
            candidate.is_bold and candidate.size_ratio >= 0.95,  # Added: bold text at normal size
            candidate.is_titlecase and candidate.word_count <= 12,  # Added: title case
            candidate.has_spacing_cue and candidate.word_count <= 8,  # Added: well-spaced short text
            candidate.is_larger_than_body  # Added: any larger text
        ]
        
        # More lenient - need fewer strong indicators
        strong_count = sum(strong_indicators)
        if strong_count == 0:
            return False
            
        # Relaxed filters
        if candidate.word_count > self.max_heading_words and not candidate.has_numbering and strong_count < 2:
            return False
            
        # Very small text is unlikely to be heading (lowered threshold)
        if candidate.font_size < 6:
            return False
            
        return True
        
    def _calculate_confidence_scores(self, candidates: List[HeadingCandidate]):
        """Calculate confidence scores for each candidate"""
        # Get document-level statistics for adaptive scoring
        doc_stats = self._analyze_document_characteristics(candidates)
        
        for candidate in candidates:
            score = 0.0
            
            # Font size contribution (0-35 points) - more adaptive
            if candidate.is_larger_than_body:
                # Scale based on how much larger it is
                size_factor = min((candidate.size_ratio - 1) * 30, 35)
                score += size_factor
            elif candidate.size_ratio >= 0.95:  # Even slightly smaller can be heading
                score += 8  # Better boost for near-body size
            elif candidate.size_ratio >= 0.90:  # Slightly smaller text can still be heading
                score += 4
                
            # Style contribution (0-25 points) - contextual
            if candidate.is_bold:
                # Bold is stronger indicator if document has mixed formatting
                bold_bonus = 15 if doc_stats['has_mixed_formatting'] else 10
                score += bold_bonus
                
            if candidate.has_distinct_font:
                score += 10
                
            # Numbering contribution (0-30 points) - unchanged as it's reliable
            if candidate.has_numbering:
                score += 30
                
            # Case contribution (0-15 points) - more nuanced
            if candidate.is_uppercase:
                # Uppercase less reliable if everything is uppercase
                if doc_stats['uppercase_ratio'] < 0.3:
                    score += 15
                else:
                    score += 5  # Reduced if common
                    
            elif candidate.is_titlecase:
                score += 10
                
            # Position and spacing (0-20 points) - enhanced
            if candidate.has_spacing_cue:
                score += 12
                
            # First on page bonus (common heading pattern)
            if candidate.block.y0 < doc_stats['avg_first_quarter_y']:
                score += 8
                
            # Length scoring - more forgiving
            word_count = candidate.word_count
            if word_count <= 2:
                score += 8  # Short phrases often headings
            elif word_count <= 6:
                score += 12  # Ideal heading length
            elif word_count <= 10:
                score += 8   # Still reasonable
            elif word_count <= 15:
                score += 3   # Still acceptable
            elif word_count <= 20:
                score -= 2   # Slight penalty
            else:
                score -= (word_count - 20) * 1  # Reduced penalty for very long
                
            # Context-based adjustments
            score += self._get_contextual_adjustments(candidate, candidates, doc_stats)
                
            candidate.confidence_score = max(score, 0)
    
    def _analyze_document_characteristics(self, candidates: List[HeadingCandidate]) -> Dict:
        """Analyze document-level characteristics for adaptive scoring"""
        if not candidates:
            return {
                'has_mixed_formatting': False,
                'uppercase_ratio': 0,
                'avg_first_quarter_y': 0
            }
            
        # Check for mixed formatting (both bold and non-bold text)
        has_bold = any(c.is_bold for c in candidates)
        has_non_bold = any(not c.is_bold for c in candidates)
        has_mixed_formatting = has_bold and has_non_bold
        
        # Calculate uppercase ratio
        uppercase_count = sum(1 for c in candidates if c.is_uppercase)
        uppercase_ratio = uppercase_count / len(candidates) if candidates else 0
        
        # Find average Y position of first quarter of page content
        # (headings often appear in top portion of pages)
        y_positions = [c.block.y0 for c in candidates]
        if y_positions:
            sorted_y = sorted(y_positions)
            first_quarter_idx = len(sorted_y) // 4
            avg_first_quarter_y = np.mean(sorted_y[:max(1, first_quarter_idx)])
        else:
            avg_first_quarter_y = 0
            
        return {
            'has_mixed_formatting': has_mixed_formatting,
            'uppercase_ratio': uppercase_ratio,
            'avg_first_quarter_y': avg_first_quarter_y
        }
    
    def _get_contextual_adjustments(self, candidate: HeadingCandidate, 
                                   all_candidates: List[HeadingCandidate],
                                   doc_stats: Dict) -> float:
        """Get context-based score adjustments"""
        adjustment = 0.0
        
        # Check for common heading patterns
        text = candidate.text.lower().strip()
        
        # Common unnumbered heading words (avoid overfitting to samples)
        unnumbered_heading_indicators = {
            'abstract', 'summary', 'introduction', 'conclusion', 'references',
            'acknowledgements', 'acknowledgments', 'appendix', 'bibliography',
            'background', 'methodology', 'results', 'discussion', 'overview',
            'table of contents', 'toc', 'contents', 'index', 'preface',
            'executive summary', 'objectives', 'scope', 'limitations',
            'recommendations', 'future work', 'related work'
        }
        
        # Check if text matches common heading patterns
        for indicator in unnumbered_heading_indicators:
            if indicator in text:
                adjustment += 8  # Moderate boost for common heading words
                break
                
        # Check for structural indicators (like being alone on a line)
        # This is inferred from having significant spacing
        if candidate.has_spacing_cue and candidate.word_count <= 8:
            adjustment += 5
            
        # Penalty for obviously non-heading content
        non_heading_indicators = [
            'page', 'copyright', '©', 'reserved', 'all rights',
            'note:', 'figure', 'table', 'chart', 'graph'
        ]
        
        for indicator in non_heading_indicators:
            if indicator in text:
                adjustment -= 10
                break
                
        return adjustment
            
    def get_high_confidence_candidates(self, candidates: List[HeadingCandidate], 
                                     min_confidence: float = 25.0) -> List[HeadingCandidate]:
        """Filter candidates by confidence threshold"""
        return [c for c in candidates if c.confidence_score >= min_confidence]


# Integration function
def detect_headings(blocks: List['TextBlock'], 
                   size_threshold_ratio: float = 1.15,
                   pdf_path: Optional[str] = None) -> List[HeadingCandidate]:
    """Main function to detect heading candidates"""
    detector = HeadingDetector(size_threshold_ratio=size_threshold_ratio)
    candidates = detector.detect_candidates(blocks, pdf_path)
    
    # Log statistics
    high_conf = detector.get_high_confidence_candidates(candidates)
    logging.info(f"Found {len(candidates)} heading candidates, {len(high_conf)} high confidence")
    
    return candidates