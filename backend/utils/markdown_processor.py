"""
Enhanced Markdown Processing for Document Discussion
Provides rich text formatting capabilities for AI responses
"""

import re
import logging
import markdown
import bleach
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MarkdownMetadata:
    """Metadata about processed markdown content"""
    has_headings: bool = False
    has_lists: bool = False
    has_emphasis: bool = False
    has_code_blocks: bool = False
    has_links: bool = False
    has_tables: bool = False
    has_blockquotes: bool = False
    word_count: int = 0
    processing_time: float = 0.0

@dataclass
class ProcessedMarkdown:
    """Container for processed markdown content"""
    html_content: str
    raw_markdown: str
    metadata: MarkdownMetadata
    is_enhanced: bool = False
    formatting_applied: List[str] = None

class MarkdownProcessor:
    """Advanced markdown processor for document discussion enhancement"""
    
    def __init__(self):
        # Configure markdown with extensions for rich formatting
        self.markdown_converter = markdown.Markdown(
            extensions=[
                'extra',           # Tables, footnotes, definition lists
                'codehilite',      # Syntax highlighting
                'toc',             # Table of contents
                'nl2br',           # Newline to <br>
                'sane_lists',      # Better list handling
                'smarty',          # Smart quotes and dashes
                'wikilinks',       # Wiki-style links
                'attr_list',       # Add attributes to elements
                'def_list',        # Definition lists
                'abbr',            # Abbreviations
                'footnotes'        # Footnotes
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': False  # Use CSS classes only
                },
                'toc': {
                    'permalink': True,
                    'title': 'Daftar Isi'
                }
            }
        )
        
        # Configure HTML sanitizer for security
        self.allowed_tags = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'br', 'hr',
            'strong', 'em', 'b', 'i', 'u', 'strike', 'del', 'ins',
            'ul', 'ol', 'li',
            'blockquote', 'cite',
            'code', 'pre', 'kbd', 'samp', 'var',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'a', 'img',
            'div', 'span',
            'sup', 'sub',
            'dl', 'dt', 'dd',
            'abbr', 'acronym'
        ]
        
        self.allowed_attributes = {
            '*': ['class', 'id', 'title'],
            'a': ['href', 'title', 'target', 'rel'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'th': ['scope', 'colspan', 'rowspan'],
            'td': ['colspan', 'rowspan'],
            'table': ['summary'],
            'code': ['class'],
            'pre': ['class'],
            'blockquote': ['cite']
        }
    
    def extract_links(self, text: str) -> List[Dict[str, str]]:
        """Extract and process links from text"""
        try:
            # Markdown links: [text](url)
            markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
            
            # Plain URLs
            url_pattern = re.compile(r'https?://[^\s]+')
            plain_urls = url_pattern.findall(text)
            
            links = []
            
            # Process markdown links
            for link_text, url in markdown_links:
                links.append({
                    'type': 'markdown',
                    'text': link_text,
                    'url': url,
                    'display': link_text
                })
            
            # Process plain URLs
            for url in plain_urls:
                if not any(link['url'] == url for link in links):
                    links.append({
                        'type': 'plain',
                        'text': url,
                        'url': url,
                        'display': url
                    })
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
    
    def clean_markdown(self, text: str) -> str:
        """Clean markdown syntax for plain text display"""
        try:
            # Remove markdown table syntax
            text = re.sub(r'\|', '', text)
            text = re.sub(r'^[\s]*:?[-:]+:?[\s]*$', '', text, flags=re.MULTILINE)
            
            # Remove emphasis markers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
            text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code
            text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
            
            # Remove heading markers
            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
            
            # Remove list markers
            text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
            
            # Clean up extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning markdown: {e}")
            return text
    
    def enhance_ai_prompt_for_formatting(self, base_prompt: str, document_context: str = None) -> str:
        """Enhance AI prompt to encourage rich markdown formatting"""
        
        formatting_instructions = """
INSTRUKSI FORMATTING:
Gunakan markdown formatting untuk membuat jawaban yang lebih mudah dibaca:

📝 **Struktur**:
- Gunakan heading (# ## ###) untuk mengorganisir jawaban
- Pisahkan bagian dengan garis horizontal (---)

🎯 **Penekanan**:
- **Bold** untuk poin penting
- *Italic* untuk penekanan ringan
- `code` untuk istilah teknis
- > Quote untuk kutipan dari dokumen

📊 **Lists & Tables**:
- Gunakan bullet points (- atau *) untuk daftar
- Gunakan numbered lists (1. 2. 3.) untuk urutan
- Buat tabel dengan | untuk data terstruktur

💡 **Code & Examples**:
```
Gunakan code blocks untuk contoh panjang
atau data terstruktur
```

🔗 **Links & References**:
- [Teks Link](URL) untuk referensi
- Footnotes[^1] untuk catatan tambahan

PENTING: Jawab dengan format markdown yang kaya untuk pengalaman membaca yang lebih baik.
"""
        
        enhanced_prompt = f"{formatting_instructions}\n\n{base_prompt}"
        
        if document_context:
            enhanced_prompt += f"\n\nKONTEKS DOKUMEN:\n{document_context[:1000]}..."
        
        return enhanced_prompt
    
    def process_markdown(self, content: str) -> ProcessedMarkdown:
        """Process raw text/markdown into enhanced HTML"""
        start_time = datetime.now()
        
        try:
            # Clean and prepare content
            cleaned_content = self._clean_input(content)
            
            # Auto-enhance basic text with markdown
            enhanced_markdown = self._auto_enhance_text(cleaned_content)
            
            # Clean up any markdown formatting conflicts
            enhanced_markdown = self.clean_markdown_conflicts(enhanced_markdown)
            
            # Convert markdown to HTML
            html_content = self.markdown_converter.convert(enhanced_markdown)
            
            # Sanitize HTML for security
            safe_html = bleach.clean(
                html_content,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
            
            # Generate metadata
            metadata = self._analyze_content(enhanced_markdown, safe_html)
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata.processing_time = processing_time
            
            # Determine formatting applied
            formatting_applied = self._detect_formatting_types(enhanced_markdown)
            
            result = ProcessedMarkdown(
                html_content=safe_html,
                raw_markdown=enhanced_markdown,
                metadata=metadata,
                is_enhanced=True,
                formatting_applied=formatting_applied
            )
            
            logger.info(f"✨ Markdown processed: {len(formatting_applied)} formats applied in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Markdown processing failed: {e}")
            # Fallback to basic processing
            return self._create_fallback_result(content)
    
    def _clean_input(self, content: str) -> str:
        """Clean and normalize input text"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Fix common formatting issues
        content = re.sub(r'(\w)\*(\w)', r'\1 *\2', content)  # Fix joined asterisks
        content = re.sub(r'(\w)_(\w)', r'\1 _\2', content)   # Fix joined underscores
        
        return content.strip()
    
    def _auto_enhance_text(self, content: str) -> str:
        """Auto-enhance plain text with markdown formatting"""
        enhanced = content
        
        # Clean up any existing markdown before enhancing to prevent duplication
        enhanced = re.sub(r'\*\*([^*]+)\*\*', r'\1', enhanced)  # Remove existing bold
        enhanced = re.sub(r'\*([^*]+)\*', r'\1', enhanced)      # Remove existing italic
        
        # Enhance question patterns
        enhanced = re.sub(
            r'^(Apa|Siapa|Kapan|Dimana|Mengapa|Bagaimana|Berapa)([^?]*\?)',
            r'### 🤔 \1\2',
            enhanced,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # Enhance important statements first (before other formatting)
        enhanced = re.sub(
            r'\b(PENTING|CATATAN|PERHATIAN|INGAT|KESIMPULAN):\s*(.+)',
            r'> **💡 \1:** *\2*',
            enhanced,
            flags=re.IGNORECASE
        )
        
        # CONSERVATIVE: Only enhance if it's clearly a title or heading pattern
        # Look for standalone lines that look like headings
        enhanced = re.sub(
            r'^([A-Z][^.!?]*[^.!?\s])$',  # Capitalized line without punctuation
            r'## \1',
            enhanced,
            flags=re.MULTILINE
        )
        
        # DISABLE numbered item enhancement to prevent double asterisks
        # The AI should provide proper markdown formatting from the prompt
        
        # DISABLE bullet point enhancement to prevent formatting conflicts
        # Let the AI handle proper list formatting
        
        # Clean up any accidentally doubled formatting
        enhanced = re.sub(r'\*\*\*\*([^*]+)\*\*\*\*', r'**\1**', enhanced)  # Fix quadruple asterisks
        enhanced = re.sub(r'\*\*\*([^*]+)\*\*\*', r'**\1**', enhanced)      # Fix triple asterisks
        enhanced = re.sub(r'###+', '###', enhanced)  # Fix multiple hashes
        
        return enhanced
    
    def _analyze_content(self, markdown_text: str, html_content: str) -> MarkdownMetadata:
        """Analyze content to generate metadata"""
        metadata = MarkdownMetadata()
        
        # Check for various markdown elements
        metadata.has_headings = bool(re.search(r'^#{1,6}\s', markdown_text, re.MULTILINE))
        metadata.has_lists = bool(re.search(r'^[\-\*\+]\s|\d+\.\s', markdown_text, re.MULTILINE))
        metadata.has_emphasis = bool(re.search(r'\*\*.*?\*\*|\*.*?\*|__.*?__|_.*?_', markdown_text))
        metadata.has_code_blocks = bool(re.search(r'```|`.*?`', markdown_text))
        metadata.has_links = bool(re.search(r'\[.*?\]\(.*?\)', markdown_text))
        metadata.has_tables = bool(re.search(r'\|.*?\|', markdown_text))
        metadata.has_blockquotes = bool(re.search(r'^>\s', markdown_text, re.MULTILINE))
        
        # Count words
        words = re.findall(r'\b\w+\b', markdown_text)
        metadata.word_count = len(words)
        
        return metadata
    
    def _detect_formatting_types(self, markdown_text: str) -> List[str]:
        """Detect which formatting types were applied"""
        formatting_types = []
        
        if re.search(r'^#{1,6}\s', markdown_text, re.MULTILINE):
            formatting_types.append('headings')
        
        if re.search(r'\*\*.*?\*\*', markdown_text):
            formatting_types.append('bold')
            
        if re.search(r'\*.*?\*', markdown_text):
            formatting_types.append('italic')
            
        if re.search(r'`.*?`', markdown_text):
            formatting_types.append('code')
            
        if re.search(r'^[\-\*\+]\s', markdown_text, re.MULTILINE):
            formatting_types.append('lists')
            
        if re.search(r'^>\s', markdown_text, re.MULTILINE):
            formatting_types.append('blockquotes')
            
        if re.search(r'\|.*?\|', markdown_text):
            formatting_types.append('tables')
            
        if re.search(r'\[.*?\]\(.*?\)', markdown_text):
            formatting_types.append('links')
            
        if not formatting_types:
            formatting_types.append('basic')
            
        return formatting_types
    
    def _create_fallback_result(self, content: str) -> ProcessedMarkdown:
        """Create fallback result for processing failures"""
        # Simple HTML escaping for safety
        safe_content = (content
                       .replace('&', '&amp;')
                       .replace('<', '&lt;')
                       .replace('>', '&gt;')
                       .replace('\n', '<br>'))
        
        return ProcessedMarkdown(
            html_content=f"<p>{safe_content}</p>",
            raw_markdown=content,
            metadata=MarkdownMetadata(word_count=len(content.split())),
            is_enhanced=False,
            formatting_applied=['fallback']
        )
    
    def clean_markdown_conflicts(self, content: str) -> str:
        """Clean up markdown formatting conflicts and duplications"""
        cleaned = content
        
        # AGGRESSIVE cleanup for double asterisks
        # Remove any standalone double asterisks
        cleaned = re.sub(r'^\*\*\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\*\*\s*\*\*', '', cleaned)  # Remove empty bold formatting
        
        # Fix multiple asterisks patterns
        cleaned = re.sub(r'\*{4,}([^*]*)\*{4,}', r'**\1**', cleaned)  # Fix 4+ asterisks
        cleaned = re.sub(r'\*{3}([^*]*)\*{3}', r'**\1**', cleaned)    # Fix triple asterisks
        
        # Remove double bold patterns that might appear consecutively
        cleaned = re.sub(r'\*\*([^*]*)\*\*\s*\*\*([^*]*)\*\*', r'**\1 \2**', cleaned)
        
        # Clean up orphaned formatting markers
        cleaned = re.sub(r'(?<!\*)\*(?!\*)', '', cleaned)  # Remove single asterisks that aren't part of pairs
        
        # Fix heading conflicts
        cleaned = re.sub(r'^#{7,}', '######', cleaned, flags=re.MULTILINE)  # Max 6 heading levels
        cleaned = re.sub(r'^#+\s*$', '', cleaned, flags=re.MULTILINE)  # Remove empty headings
        
        # Clean up extra spaces around formatting
        cleaned = re.sub(r'\*\*\s+([^*]+)\s+\*\*', r'**\1**', cleaned)
        cleaned = re.sub(r'\*\s+([^*]+)\s+\*', r'*\1*', cleaned)
        
        # Remove any lines that are just formatting characters
        cleaned = re.sub(r'^\s*[\*#]+\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

class DocumentMarkdownEnhancer:
    """Specialized markdown enhancer for document analysis"""
    
    def __init__(self):
        self.processor = MarkdownProcessor()
    
    def enhance_document_analysis_prompt(self, question: str, document_context: str) -> str:
        """Create enhanced prompt specifically for document analysis with rich formatting"""
        
        enhanced_prompt = f"""
# 📖 ANALISIS DOKUMEN

## Pertanyaan
{question}

## Instruksi Formatting
Jawab dengan format markdown yang kaya untuk pengalaman membaca yang optimal:

### 🎯 Struktur Jawaban
1. **Ringkasan Cepat** - Jawaban singkat di awal
2. **Analisis Detail** - Penjelasan mendalam dengan formatting
3. **Kutipan Relevan** - Gunakan blockquotes untuk referensi
4. **Kesimpulan** - Rangkuman dengan poin-poin penting

### 📝 Panduan Formatting
- Gunakan **bold** untuk konsep kunci
- Gunakan *italic* untuk penekanan
- Gunakan `code` untuk istilah teknis
- Gunakan > blockquote untuk kutipan dokumen
- Gunakan lists untuk organisasi informasi
- Gunakan tabel jika ada data terstruktur
- Gunakan --- untuk memisahkan bagian

## Konteks Dokumen
{document_context[:2000]}{"..." if len(document_context) > 2000 else ""}

---

**Jawab sekarang dengan format markdown yang kaya:**
"""
        
        return enhanced_prompt
    
    def process_document_response(self, ai_response: str) -> Dict[str, Any]:
        """Process AI response for document analysis with enhanced formatting"""
        
        # Process with markdown
        processed = self.processor.process_markdown(ai_response)
        
        # Additional document-specific enhancements
        enhanced_html = self._add_document_styling(processed.html_content)
        
        return {
            "html_content": enhanced_html,
            "raw_markdown": processed.raw_markdown,
            "metadata": {
                "has_rich_formatting": processed.is_enhanced,
                "formatting_types": processed.formatting_applied,
                "word_count": processed.metadata.word_count,
                "has_headings": processed.metadata.has_headings,
                "has_lists": processed.metadata.has_lists,
                "has_emphasis": processed.metadata.has_emphasis,
                "has_code": processed.metadata.has_code_blocks,
                "has_quotes": processed.metadata.has_blockquotes,
                "has_tables": processed.metadata.has_tables,
                "processing_time": processed.metadata.processing_time
            },
            "display_type": "markdown_enhanced",
            "rendering_hints": {
                "use_syntax_highlighting": processed.metadata.has_code_blocks,
                "show_table_of_contents": processed.metadata.has_headings,
                "enable_copy_buttons": True,
                "style_theme": "document_analysis"
            }
        }
    
    def _add_document_styling(self, html_content: str) -> str:
        """Add document-specific CSS classes for styling"""
        
        # Add document-specific classes
        styled_html = html_content
        
        # Style headings
        styled_html = re.sub(
            r'<h([1-6])>',
            r'<h\1 class="doc-heading doc-heading-\1">',
            styled_html
        )
        
        # Style blockquotes (for document references)
        styled_html = re.sub(
            r'<blockquote>',
            r'<blockquote class="doc-quote">',
            styled_html
        )
        
        # Style tables
        styled_html = re.sub(
            r'<table>',
            r'<table class="doc-table">',
            styled_html
        )
        
        # Style code blocks
        styled_html = re.sub(
            r'<pre><code',
            r'<pre class="doc-code"><code',
            styled_html
        )
        
        return styled_html
