"""
🎨 Response Formatter
Enhanced AI response formatting with table support and rich content processing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json
import re  # Add missing re import

from .table_parser import TableParser, TableData

logger = logging.getLogger(__name__)

@dataclass
class TextPart:
    """Text content part"""
    content: str
    type: str = "paragraph"  # paragraph, heading, list, code
    metadata: Dict[str, Any] = None

@dataclass
class FormattedResponse:
    """Complete formatted response structure"""
    original_response: str
    has_tables: bool
    has_enhanced_content: bool
    text_parts: List[TextPart]
    tables: List[TableData]
    metadata: Dict[str, Any]
    formatting_applied: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'original_response': self.original_response,
            'has_tables': self.has_tables,
            'has_enhanced_content': self.has_enhanced_content,
            'text_parts': [asdict(part) for part in self.text_parts],
            'tables': [asdict(table) for table in self.tables],
            'metadata': self.metadata,
            'formatting_applied': self.formatting_applied
        }

class ResponseFormatter:
    """🚀 Advanced response formatting with multiple enhancements"""
    
    def __init__(self):
        self.table_parser = TableParser()
        self.supported_features = [
            'table_parsing',
            'heading_detection', 
            'list_formatting',
            'code_block_detection',
            'url_linking',
            'emphasis_detection'
        ]
        
        # Import markdown processor for rich text formatting
        from .markdown_processor import MarkdownProcessor, DocumentMarkdownEnhancer
        self.markdown_processor = MarkdownProcessor()
        self.doc_markdown_enhancer = DocumentMarkdownEnhancer()
        
    def format_response(self, ai_response: str, options: Dict[str, Any] = None) -> FormattedResponse:
        """
        🎨 Main formatting method - transform AI response into rich format
        
        Args:
            ai_response: Raw AI response text
            options: Formatting options and preferences
            
        Returns:
            FormattedResponse with enhanced content structure
        """
        try:
            logger.info(f"🎨 Starting response formatting for {len(ai_response)} chars")
            
            options = options or {}
            formatting_applied = []
            
            # Step 1: Parse tables first
            table_result = self.table_parser.parse_response_tables(ai_response)
            formatting_applied.append('table_parsing')
            
            # Step 2: Process text parts for additional formatting
            enhanced_text_parts = []
            
            if table_result['has_tables']:
                # Process text parts between tables
                for text_content in table_result['text_parts']:
                    processed_parts = self._process_text_content(text_content, options)
                    enhanced_text_parts.extend(processed_parts)
            else:
                # Process entire response as text
                enhanced_text_parts = self._process_text_content(ai_response, options)
            
            # Step 3: Apply additional formatting
            if options.get('detect_headings', True):
                enhanced_text_parts = self._enhance_headings(enhanced_text_parts)
                formatting_applied.append('heading_detection')
                
            if options.get('format_lists', True):
                enhanced_text_parts = self._format_lists(enhanced_text_parts)
                formatting_applied.append('list_formatting')
                
            if options.get('detect_code', True):
                enhanced_text_parts = self._detect_code_blocks(enhanced_text_parts)
                formatting_applied.append('code_block_detection')
                
            # Step 4: Build final response
            formatted_response = FormattedResponse(
                original_response=ai_response,
                has_tables=table_result['has_tables'],
                has_enhanced_content=len(formatting_applied) > 1,
                text_parts=enhanced_text_parts,
                tables=table_result['tables'],
                metadata={
                    'formatting_timestamp': datetime.now().isoformat(),
                    'original_length': len(ai_response),
                    'table_count': table_result['table_count'],
                    'text_parts_count': len(enhanced_text_parts),
                    'processing_features': formatting_applied,
                    'table_metadata': table_result.get('parsing_metadata', {}),
                    'enhancement_applied': True
                },
                formatting_applied=formatting_applied
            )
            
            logger.info(f"✅ Response formatting completed: {len(formatting_applied)} features applied")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            
            # Fallback to basic structure
            return FormattedResponse(
                original_response=ai_response,
                has_tables=False,
                has_enhanced_content=False,
                text_parts=[TextPart(content=ai_response, type="paragraph")],
                tables=[],
                metadata={
                    'error': str(e),
                    'fallback_used': True,
                    'formatting_timestamp': datetime.now().isoformat()
                },
                formatting_applied=['fallback']
            )
    
    def _clean_text_formatting(self, text: str) -> str:
        """Clean text from unwanted markdown formatting artifacts"""
        cleaned = text
        
        # Remove double asterisks that might cause formatting issues
        # But preserve intentional markdown formatting in context
        # Only remove standalone double asterisks or obvious artifacts
        cleaned = re.sub(r'^\*\*([^*]+)\*\*$', r'\1', cleaned, flags=re.MULTILINE)  # Line-only bold
        cleaned = re.sub(r'\*\*\s*\*\*', '', cleaned)  # Empty bold formatting
        cleaned = re.sub(r'\*\*([^*]{1,3})\*\*', r'\1', cleaned)  # Very short bold (likely artifacts)
        
        # Clean up orphaned asterisks
        cleaned = re.sub(r'(?<!\*)\*(?!\*)', '', cleaned)  # Single asterisks not part of pairs
        
        return cleaned.strip()
    
    def _process_text_content(self, text: str, options: Dict[str, Any]) -> List[TextPart]:
        """Process text content into structured parts"""
        if not text.strip():
            return []
        
        # Clean text formatting first
        cleaned_text = self._clean_text_formatting(text)
            
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        
        text_parts = []
        for paragraph in paragraphs:
            part_type = self._detect_content_type(paragraph)
            text_parts.append(TextPart(
                content=paragraph,
                type=part_type,
                metadata={'word_count': len(paragraph.split())}
            ))
            
        return text_parts
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of text content"""
        text = text.strip()
        
        # Check for headings
        if text.startswith('#'):
            return "heading"
        
        # Check for lists
        if any(text.startswith(marker) for marker in ['- ', '* ', '1. ', '2. ', '3.']):
            return "list"
            
        # Check for code blocks
        if text.startswith('```') or text.startswith('    '):
            return "code"
            
        # Check for quotes
        if text.startswith('>'):
            return "quote"
            
        return "paragraph"
    
    def _enhance_headings(self, text_parts: List[TextPart]) -> List[TextPart]:
        """Enhance heading detection and formatting"""
        enhanced_parts = []
        
        for part in text_parts:
            if part.type == "paragraph":
                # Look for heading patterns
                lines = part.content.split('\n')
                if len(lines) == 1 and len(part.content) < 100:
                    # Short single line might be a heading
                    if any(word in part.content.lower() for word in ['overview', 'summary', 'conclusion', 'introduction']):
                        part.type = "heading"
                        part.metadata = part.metadata or {}
                        part.metadata['heading_level'] = 2
                        
            enhanced_parts.append(part)
            
        return enhanced_parts
    
    def _format_lists(self, text_parts: List[TextPart]) -> List[TextPart]:
        """Enhanced list formatting and detection"""
        enhanced_parts = []
        
        for part in text_parts:
            if part.type == "list" or '\n-' in part.content or '\n*' in part.content:
                part.type = "list"
                part.metadata = part.metadata or {}
                
                # Count list items
                list_markers = ['-', '*', '1.', '2.', '3.', '4.', '5.']
                item_count = sum(part.content.count(f'\n{marker}') + 
                               (1 if part.content.startswith(marker) else 0) 
                               for marker in list_markers)
                
                part.metadata['list_items'] = max(item_count, 1)
                part.metadata['list_type'] = 'ordered' if any(char.isdigit() for char in part.content[:10]) else 'unordered'
                
            enhanced_parts.append(part)
            
        return enhanced_parts
    
    def _detect_code_blocks(self, text_parts: List[TextPart]) -> List[TextPart]:
        """Detect and format code blocks"""
        enhanced_parts = []
        
        for part in text_parts:
            if '```' in part.content or part.content.startswith('    '):
                part.type = "code"
                part.metadata = part.metadata or {}
                
                # Try to detect language
                if part.content.startswith('```'):
                    first_line = part.content.split('\n')[0]
                    if len(first_line) > 3:
                        part.metadata['language'] = first_line[3:].strip()
                    else:
                        part.metadata['language'] = 'text'
                else:
                    part.metadata['language'] = 'text'
                    
                part.metadata['code_lines'] = len(part.content.split('\n'))
                
            enhanced_parts.append(part)
            
        return enhanced_parts
    
    def format_for_frontend(self, formatted_response: FormattedResponse) -> Dict[str, Any]:
        """
        🎯 Format specifically for frontend consumption
        
        Returns optimized structure for React components
        """
        try:
            frontend_data = {
                'response_id': f"resp_{hash(formatted_response.original_response) % 100000}",
                'content_type': 'enhanced' if formatted_response.has_enhanced_content else 'simple',
                'has_tables': formatted_response.has_tables,
                'sections': [],
                'metadata': formatted_response.metadata
            }
            
            # Build sections combining text and tables
            section_id = 0
            
            for text_part in formatted_response.text_parts:
                section_id += 1
                frontend_data['sections'].append({
                    'id': f"section_{section_id}",
                    'type': 'text',
                    'content': text_part.content,
                    'text_type': text_part.type,
                    'metadata': text_part.metadata or {}
                })
            
            # Add table sections
            for table in formatted_response.tables:
                section_id += 1
                frontend_data['sections'].append({
                    'id': f"section_{section_id}",
                    'type': 'table',
                    'table_id': table.id,
                    'headers': table.headers,
                    'rows': table.rows,
                    'column_types': table.column_types,
                    'metadata': table.metadata
                })
            
            # Add rendering hints for frontend
            frontend_data['rendering_hints'] = {
                'total_sections': len(frontend_data['sections']),
                'table_count': len(formatted_response.tables),
                'text_sections': len(formatted_response.text_parts),
                'recommended_theme': 'modern' if formatted_response.has_tables else 'simple',
                'mobile_optimized': True,
                'export_supported': formatted_response.has_tables
            }
            
            return frontend_data
            
        except Exception as e:
            logger.error(f"Error formatting for frontend: {e}")
            return {
                'response_id': 'error',
                'content_type': 'simple',
                'has_tables': False,
                'sections': [{
                    'id': 'section_1',
                    'type': 'text',
                    'content': formatted_response.original_response,
                    'text_type': 'paragraph'
                }],
                'error': str(e)
            }
    
    def remove_markdown_tables_from_text(self, text: str) -> str:
        """
        ✂️ Remove markdown table syntax from text when tables are rendered separately
        This prevents duplicate table content (markdown + enhanced table)
        """
        import re
        
        # Pattern to match markdown tables (including incomplete ones)
        table_pattern = r'\|[^\n]*\|[^\n]*\n(?:\|[-:\s]*\|[-:\s]*\n)?(?:\|[^\n]*\|[^\n]*\n)*'
        
        # Remove markdown tables
        cleaned_text = re.sub(table_pattern, '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        logger.info(f"✂️ Removed markdown tables from text content")
        return cleaned_text
