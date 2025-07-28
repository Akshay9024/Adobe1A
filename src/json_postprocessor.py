import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re

@dataclass
class OutlineEntry:
    """Represents a single entry in the document outline"""
    level: str  # "H1", "H2", or "H3"
    text: str
    page: int  # 0-indexed internally, will be converted to 1-indexed for output
    y_position: float  # For sorting
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to output dictionary format"""
        return {
            "level": self.level,
            "text": self.text,
            "page": self.page  # Keep 0-indexed as required
        }


class TextNormalizer:
    """Handles text normalization and preservation"""
    
    def __init__(self, preserve_trailing_spaces: bool = True,
                 preserve_exact_case: bool = True):
        self.preserve_trailing_spaces = preserve_trailing_spaces
        self.preserve_exact_case = preserve_exact_case
        
    def normalize_heading_text(self, text: str) -> str:
        """Normalize heading text while preserving required formatting"""
        if not text:
            return ""
            
        # Remove control characters but preserve normal whitespace
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Handle internal whitespace (collapse multiple spaces to single)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Handle leading whitespace (always remove)
        text = text.lstrip()
        
        # Handle trailing whitespace based on setting
        if not self.preserve_trailing_spaces:
            text = text.rstrip()
            
        # Preserve exact case as required
        # (no transformation needed if preserve_exact_case is True)
        
        return text
        
    def validate_json_compatibility(self, text: str) -> str:
        """Ensure text is JSON-safe"""
        # The json library handles most cases, but we ensure no issues
        # with special characters that might break JSON
        
        # Replace any remaining problematic characters
        text = text.replace('\u0000', '')  # Null character
        
        return text


class OutlineBuilder:
    """Builds the final document outline"""
    
    def __init__(self, text_normalizer: Optional[TextNormalizer] = None):
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.entries = []
        
    def add_heading(self, candidate: 'HeadingCandidate'):
        """Add a heading candidate to the outline"""
        if not candidate.heading_level:
            return
            
        # Normalize text
        normalized_text = self.text_normalizer.normalize_heading_text(
            candidate.text
        )
        
        # Validate JSON compatibility
        safe_text = self.text_normalizer.validate_json_compatibility(
            normalized_text
        )
        
        # Create outline entry
        entry = OutlineEntry(
            level=f"H{candidate.heading_level}",
            text=safe_text,
            page=candidate.page_num,  # Still 0-indexed
            y_position=candidate.block.y0
        )
        
        self.entries.append(entry)
        
    def build_sorted_outline(self) -> List[Dict[str, Any]]:
        """Build the final sorted outline"""
        # Sort by page first, then by y-position (reading order)
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.page, e.y_position)
        )
        
        # Convert to output format
        outline = [entry.to_dict() for entry in sorted_entries]
        
        return outline
        
    def validate_outline_hierarchy(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and optionally fix outline hierarchy issues"""
        # Check for common issues
        issues = []
        
        for i, entry in enumerate(outline):
            level = entry['level']
            
            # Check for orphaned H3s (H3 without preceding H2)
            if level == 'H3' and i > 0:
                # Look for preceding H2 on same or previous page
                found_h2 = False
                for j in range(i-1, -1, -1):
                    if outline[j]['level'] == 'H2':
                        found_h2 = True
                        break
                    elif outline[j]['level'] == 'H1':
                        break  # Stop at H1
                        
                if not found_h2:
                    issues.append(f"Orphaned H3 at index {i}: {entry['text'][:50]}")
                    
        if issues:
            logging.warning(f"Outline hierarchy issues: {issues}")
            
        return outline


class JSONFormatter:
    """Formats the final JSON output"""
    
    def __init__(self, indent: Optional[int] = 2,
                 ensure_ascii: bool = False,
                 sort_keys: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.sort_keys = sort_keys
        
    def format_output(self, title: str, outline: List[Dict[str, Any]]) -> str:
        """Format the complete output as JSON"""
        output = {
            "title": title,
            "outline": outline
        }
        
        # Convert to JSON with specified formatting
        json_str = json.dumps(
            output,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            sort_keys=self.sort_keys
        )
        
        return json_str
        
    def validate_json(self, json_str: str) -> bool:
        """Validate that the JSON is well-formed"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {e}")
            return False


class PostProcessor:
    """Main postprocessing orchestrator"""
    
    def __init__(self, 
                 preserve_trailing_spaces: bool = True,
                 validate_hierarchy: bool = True):
        self.text_normalizer = TextNormalizer(
            preserve_trailing_spaces=preserve_trailing_spaces
        )
        self.outline_builder = OutlineBuilder(self.text_normalizer)
        self.json_formatter = JSONFormatter()
        self.validate_hierarchy = validate_hierarchy
        
    def process(self, 
                title: str,
                heading_candidates: List['HeadingCandidate']) -> str:
        """Process candidates and generate final JSON output"""
        
        # Filter to only confirmed headings with levels
        confirmed_headings = [
            c for c in heading_candidates 
            if c.heading_level is not None
        ]
        
        logging.info(f"Processing {len(confirmed_headings)} confirmed headings")
        
        # Build outline
        self.outline_builder.entries.clear()  # Reset
        for candidate in confirmed_headings:
            self.outline_builder.add_heading(candidate)
            
        # Get sorted outline
        outline = self.outline_builder.build_sorted_outline()
        
        # Validate hierarchy if requested
        if self.validate_hierarchy:
            outline = self.outline_builder.validate_outline_hierarchy(outline)
            
        # Normalize title
        normalized_title = self.text_normalizer.normalize_heading_text(title)
        safe_title = self.text_normalizer.validate_json_compatibility(normalized_title)
        
        # Format as JSON
        json_output = self.json_formatter.format_output(safe_title, outline)
        
        # Validate JSON
        if not self.json_formatter.validate_json(json_output):
            raise ValueError("Generated invalid JSON output")
            
        # Log summary
        self._log_output_summary(safe_title, outline)
        
        return json_output
        
    def _log_output_summary(self, title: str, outline: List[Dict[str, Any]]):
        """Log summary of output"""
        level_counts = {'H1': 0, 'H2': 0, 'H3': 0}
        for entry in outline:
            level_counts[entry['level']] += 1
            
        logging.info(f"Output summary:")
        logging.info(f"  Title: '{title[:50]}...' ({len(title)} chars)")
        logging.info(f"  Total headings: {len(outline)}")
        logging.info(f"  Distribution: H1={level_counts['H1']}, "
                    f"H2={level_counts['H2']}, H3={level_counts['H3']}")
        
        if outline:
            pages = set(entry['page'] for entry in outline)
            logging.info(f"  Page range: {min(pages)}-{max(pages)}")


# Convenience functions
def create_json_output(title: str,
                      heading_candidates: List['HeadingCandidate'],
                      preserve_trailing_spaces: bool = True) -> str:
    """Create JSON output from title and heading candidates"""
    processor = PostProcessor(preserve_trailing_spaces=preserve_trailing_spaces)
    return processor.process(title, heading_candidates)


def save_json_output(json_str: str, output_path: str):
    """Save JSON output to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    logging.info(f"JSON output saved to {output_path}")


def validate_output_format(json_str: str) -> bool:
    """Validate output matches expected format"""
    try:
        data = json.loads(json_str)
        
        # Check required fields
        if 'title' not in data or 'outline' not in data:
            logging.error("Missing required fields: 'title' or 'outline'")
            return False
            
        # Check title is string
        if not isinstance(data['title'], str):
            logging.error("Title must be a string")
            return False
            
        # Check outline is list
        if not isinstance(data['outline'], list):
            logging.error("Outline must be a list")
            return False
            
        # Check each outline entry
        for i, entry in enumerate(data['outline']):
            if not isinstance(entry, dict):
                logging.error(f"Outline entry {i} must be a dictionary")
                return False
                
            # Check required fields in entry
            required = {'level', 'text', 'page'}
            if not all(field in entry for field in required):
                logging.error(f"Outline entry {i} missing required fields")
                return False
                
            # Check field types
            if not isinstance(entry['level'], str) or entry['level'] not in ['H1', 'H2', 'H3']:
                logging.error(f"Invalid level in entry {i}: {entry['level']}")
                return False
                
            if not isinstance(entry['text'], str):
                logging.error(f"Text must be string in entry {i}")
                return False
                
            if not isinstance(entry['page'], int) or entry['page'] < 1:
                logging.error(f"Invalid page number in entry {i}: {entry['page']}")
                return False
                
        return True
        
    except json.JSONDecodeError:
        logging.error("Invalid JSON format")
        return False


# Example usage showing complete pipeline integration
def generate_final_output(pdf_path: str, 
                         heading_candidates: List['HeadingCandidate'],
                         document_title: str) -> Dict[str, Any]:
    """Generate final output dictionary (before JSON serialization)"""
    
    # Create processor
    processor = PostProcessor(preserve_trailing_spaces=True)
    
    # Generate JSON string
    json_output = processor.process(document_title, heading_candidates)
    
    # Parse back to dict for potential further processing
    output_dict = json.loads(json_output)
    
    # Validate
    if validate_output_format(json_output):
        logging.info("Output validation passed")
    else:
        logging.error("Output validation failed")
        
    return output_dict