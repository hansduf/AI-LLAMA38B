"""
📊 Table Parser & Detector
Advanced markdown table detection and parsing for beautiful frontend rendering
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TableCell:
    """Individual table cell data"""
    content: str
    type: str = "text"  # text, number, date, url, email
    alignment: str = "left"  # left, center, right
    
@dataclass 
class TableRow:
    """Table row containing cells"""
    cells: List[TableCell]
    is_header: bool = False

@dataclass
class TableData:
    """Complete table structure"""
    id: str
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any]
    column_types: List[str]
    start_position: int
    end_position: int
    raw_markdown: str

@dataclass
class TableMatch:
    """Table detection result"""
    table_data: TableData
    start_pos: int
    end_pos: int
    confidence: float

class TableDetector:
    """🔍 Advanced table detection in AI responses"""
    
    def __init__(self):
        # Regex patterns for different table formats
        self.markdown_table_pattern = re.compile(
            r'(\|[^\n]*\|[\s]*\n\|[\s]*:?[-:]+:?[\s]*\|[^\n]*\n(?:\|[^\n]*\|[\s]*\n)*)',
            re.MULTILINE
        )
        
        # Enhanced patterns for better detection
        self.table_header_pattern = re.compile(r'\|([^|\n]+\|)+')
        self.table_separator_pattern = re.compile(r'\|[\s]*:?[-:]+:?[\s]*\|')
        
    def detect_tables(self, text: str) -> List[TableMatch]:
        """
        🎯 Detect all tables in text with confidence scoring
        
        Args:
            text: Input text to search for tables
            
        Returns:
            List of TableMatch objects with confidence scores
        """
        try:
            matches = []
            
            # Find all markdown table patterns
            for match in self.markdown_table_pattern.finditer(text):
                table_text = match.group(1).strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Validate and parse table
                if self._validate_table_structure(table_text):
                    table_data = self._parse_markdown_table(
                        table_text, start_pos, end_pos
                    )
                    
                    confidence = self._calculate_confidence(table_data)
                    
                    matches.append(TableMatch(
                        table_data=table_data,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence
                    ))
                    
                    logger.info(f"📊 Table detected with {confidence:.1%} confidence")
            
            return sorted(matches, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting tables: {e}")
            return []
    
    def _validate_table_structure(self, table_text: str) -> bool:
        """Validate if text is a proper table structure"""
        lines = table_text.strip().split('\n')
        
        # Must have at least 3 lines (header, separator, data)
        if len(lines) < 3:
            return False
            
        # Check for header separator pattern
        if not any(self.table_separator_pattern.search(line) for line in lines):
            return False
            
        # Check for consistent column count
        pipe_counts = [line.count('|') for line in lines if line.strip()]
        if len(set(pipe_counts)) > 2:  # Allow for slight variations
            return False
            
        return True
    
    def _clean_cell_content(self, content: str) -> str:
        """Clean cell content from markdown formatting artifacts"""
        cleaned = content.strip()
        
        # Remove double asterisks (bold markdown)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove single asterisks (italic markdown) 
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
        
        # Remove backticks (code markdown)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _parse_markdown_table(self, table_text: str, start_pos: int, end_pos: int) -> TableData:
        """Parse markdown table into structured data"""
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        
        # Extract headers (first line)
        header_line = lines[0]
        headers = [self._clean_cell_content(cell) for cell in header_line.split('|')[1:-1]]
        
        # Skip separator line (second line) and process data rows
        data_rows = []
        for line in lines[2:]:
            if line.strip():
                cells = [self._clean_cell_content(cell) for cell in line.split('|')[1:-1]]
                # Ensure consistent column count
                while len(cells) < len(headers):
                    cells.append("")
                data_rows.append(cells[:len(headers)])
        
        # Analyze column types
        column_types = self._analyze_column_types(data_rows, headers)
        
        # Generate unique table ID
        table_id = f"table_{hash(table_text) % 100000}"
        
        return TableData(
            id=table_id,
            headers=headers,
            rows=data_rows,
            metadata={
                "row_count": len(data_rows),
                "column_count": len(headers),
                "detected_at": datetime.now().isoformat(),
                "has_numbers": any("number" in types for types in column_types),
                "total_cells": len(data_rows) * len(headers)
            },
            column_types=column_types,
            start_position=start_pos,
            end_position=end_pos,
            raw_markdown=table_text
        )
    
    def _analyze_column_types(self, rows: List[List[str]], headers: List[str]) -> List[str]:
        """🔍 Smart column type detection"""
        column_types = []
        
        for col_idx in range(len(headers)):
            column_values = [row[col_idx] for row in rows if col_idx < len(row)]
            col_type = self._detect_column_type(column_values)
            column_types.append(col_type)
            
        return column_types
    
    def _detect_column_type(self, values: List[str]) -> str:
        """Detect data type of a column"""
        if not values:
            return "text"
            
        # Remove empty values for analysis
        non_empty = [v.strip() for v in values if v.strip()]
        if not non_empty:
            return "text"
            
        # Check for numbers
        number_count = 0
        for value in non_empty:
            try:
                float(value.replace(',', ''))
                number_count += 1
            except ValueError:
                pass
                
        if number_count > len(non_empty) * 0.7:  # 70% numbers
            return "number"
            
        # Check for URLs
        url_pattern = re.compile(r'https?://\S+')
        if any(url_pattern.search(v) for v in non_empty):
            return "url"
            
        # Check for emails  
        email_pattern = re.compile(r'\S+@\S+\.\S+')
        if any(email_pattern.search(v) for v in non_empty):
            return "email"
            
        # Check for dates
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}-\d{1,2}-\d{4}'
        ]
        
        for pattern in date_patterns:
            if any(re.search(pattern, v) for v in non_empty):
                return "date"
                
        return "text"
    
    def _calculate_confidence(self, table_data: TableData) -> float:
        """Calculate confidence score for table detection"""
        score = 0.5  # Base score
        
        # More columns = higher confidence
        if table_data.metadata["column_count"] >= 3:
            score += 0.2
        elif table_data.metadata["column_count"] >= 2:
            score += 0.1
            
        # More rows = higher confidence  
        if table_data.metadata["row_count"] >= 3:
            score += 0.2
        elif table_data.metadata["row_count"] >= 2:
            score += 0.1
            
        # Consistent data types boost confidence
        if len(set(table_data.column_types)) > 1:
            score += 0.1
            
        return min(score, 1.0)

class TableParser:
    """🎨 Advanced table parsing and formatting"""
    
    def __init__(self):
        self.detector = TableDetector()
        
    def parse_response_tables(self, ai_response: str) -> Dict[str, Any]:
        """
        🚀 Main method to parse tables from AI response
        
        Returns:
            {
                'has_tables': bool,
                'table_count': int,
                'tables': List[TableData],
                'text_parts': List[str],
                'enhanced_response': str
            }
        """
        try:
            # Detect all tables
            table_matches = self.detector.detect_tables(ai_response)
            
            if not table_matches:
                return {
                    'has_tables': False,
                    'table_count': 0,
                    'tables': [],
                    'text_parts': [ai_response],
                    'enhanced_response': ai_response
                }
            
            # Extract text parts and tables
            text_parts = []
            tables = []
            last_pos = 0
            
            for match in table_matches:
                # Add text before table
                if match.start_pos > last_pos:
                    text_part = ai_response[last_pos:match.start_pos].strip()
                    if text_part:
                        text_parts.append(text_part)
                
                # Add table
                tables.append(match.table_data)
                last_pos = match.end_pos
            
            # Add remaining text after last table
            if last_pos < len(ai_response):
                remaining_text = ai_response[last_pos:].strip()
                if remaining_text:
                    text_parts.append(remaining_text)
            
            logger.info(f"✅ Parsed {len(tables)} tables from AI response")
            
            return {
                'has_tables': True,
                'table_count': len(tables),
                'tables': tables,
                'text_parts': text_parts,
                'enhanced_response': ai_response,
                'parsing_metadata': {
                    'total_cells': sum(t.metadata['total_cells'] for t in tables),
                    'avg_confidence': sum(m.confidence for m in table_matches) / len(table_matches),
                    'detected_types': list(set(
                        col_type for table in tables for col_type in table.column_types
                    ))
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing tables from response: {e}")
            return {
                'has_tables': False,
                'table_count': 0,
                'tables': [],
                'text_parts': [ai_response],
                'enhanced_response': ai_response,
                'error': str(e)
            }
