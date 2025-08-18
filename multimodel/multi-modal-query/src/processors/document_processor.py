"""
Document processor for handling document-based queries.
Supports PDF, DOCX, and other document formats.
"""

import io
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import openai
import PyPDF2
from docx import Document
import pandas as pd
from .base_processor import BaseProcessor, ProcessingResult

class DocumentProcessor(BaseProcessor):
    """Processes document queries and extracts textual information."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Initialize OpenAI client
        openai_key = api_key or os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                openai.api_key = openai_key
                self.client = openai.OpenAI(api_key=openai_key)
                self.logger.info("✅ OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None
    
    def can_process(self, input_data: Union[str, bytes, Path]) -> bool:
        """Check if input is document-based."""
        if isinstance(input_data, Path):
            return input_data.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml']
        elif isinstance(input_data, str):
            path = Path(input_data)
            return path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml']
        elif isinstance(input_data, bytes):
            return self._detect_document_type(input_data)
        return False
    
    def process(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """Process document input and extract information."""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Extract text from document
        document_text = self._extract_text(input_data)
        
        # Extract metadata
        metadata = self.extract_metadata(input_data)
        metadata.update({
            "document_type": self._get_document_type(input_data),
            "text_length": len(document_text),
            "word_count": len(document_text.split())
        })
        
        # Process with GPT-3.5-turbo
        processed_content = self._process_with_gpt(document_text, **kwargs)
        
        return ProcessingResult(
            content=processed_content,
            metadata=metadata,
            confidence=0.9,
            modality="document",
            processed_data={
                "original_text": document_text,
                "document_structure": self._analyze_structure(document_text)
            }
        )
    
    def _extract_text(self, input_data: Union[str, bytes, Path]) -> str:
        """Extract text from various document formats."""
        if isinstance(input_data, bytes):
            return self._extract_from_bytes(input_data)
        elif isinstance(input_data, Path):
            return self._extract_from_file(input_data)
        elif isinstance(input_data, str):
            path = Path(input_data)
            return self._extract_from_file(path)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _extract_from_file(self, file_path: Path) -> str:
        """Extract text from file based on extension."""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension == '.docx':
            return self._extract_from_docx(file_path)
        elif extension == '.txt':
            return file_path.read_text(encoding='utf-8')
        elif extension == '.csv':
            return self._extract_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def _extract_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            self.logger.error(f"Error extracting text from CSV: {e}")
            return ""
    
    def _process_with_gpt(self, text: str, **kwargs) -> str:
        """Process document text with GPT-3.5-turbo."""
        if not self.client:
            self.logger.warning("No OpenAI client available, using basic document analysis")
            return f"Document Analysis: {len(text)} characters, {len(text.split())} words"
            
        try:
            prompt = f"""
            Analyze this document and provide a structured summary:
            
            Document Content: {text[:4000]}
            
            Please provide:
            1. Document type and purpose
            2. Main topics and key points
            3. Important information or data
            4. Structure and organization
            5. Key takeaways or conclusions
            
            Format as a clear, structured summary.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes and structures document content for further processing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error processing document with GPT: {e}")
            return f"Document Analysis: {len(text)} characters, {len(text.split())} words"
    
    def _detect_document_type(self, data: bytes) -> bool:
        """Detect if bytes represent a document."""
        return data.startswith(b'%PDF') or data.startswith(b'PK\x03\x04')
    
    def _get_document_type(self, input_data: Union[str, bytes, Path]) -> str:
        """Get document type."""
        if isinstance(input_data, Path):
            return input_data.suffix.lower()
        elif isinstance(input_data, str):
            return Path(input_data).suffix.lower()
        elif isinstance(input_data, bytes):
            if data.startswith(b'%PDF'):
                return '.pdf'
            elif data.startswith(b'PK\x03\x04'):
                return '.docx'
            else:
                return 'unknown'
        return 'unknown'
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure."""
        lines = text.split('\n')
        
        structure = {
            "total_lines": len(lines),
            "paragraphs": len([line for line in lines if line.strip()]),
            "has_tables": '\t' in text or '  ' in text,
            "has_lists": any(line.strip().startswith(('-', '•', '*')) for line in lines)
        }
        
        return structure
