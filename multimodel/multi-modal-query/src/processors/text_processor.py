"""
Text processor for handling text-based queries.
Uses GPT-3.5-turbo for cost-effective processing.
"""

import re
import nltk
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import openai
from .base_processor import BaseProcessor, ProcessingResult

# Import alternative AI providers
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

class TextProcessor(BaseProcessor):
    """Processes text queries and extracts key information."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        
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
                self.openai_client = openai.OpenAI(api_key=openai_key)
                self.logger.info("✅ OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize OpenAI client: {e}")
        
        # Initialize Anthropic client as fallback
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                self.logger.info("✅ Anthropic client initialized successfully")
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize Anthropic client: {e}")
        
        # Initialize Google Gemini client as another fallback
        google_key = os.getenv('GOOGLE_API_KEY')
        if GOOGLE_AVAILABLE and google_key and google_key != 'your_google_api_key_here':
            try:
                genai.configure(api_key=google_key)
                self.google_client = genai.GenerativeModel('gemini-pro')
                self.logger.info("✅ Google Gemini client initialized successfully")
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize Google client: {e}")
        
        # Legacy compatibility
        self.client = self.openai_client
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def can_process(self, input_data: Union[str, bytes, Path]) -> bool:
        """Check if input is text-based."""
        if isinstance(input_data, str):
            # Check if it's a file path first
            if self._looks_like_file_path(input_data):
                path = Path(input_data)
                return path.suffix.lower() in ['.txt', '.md', '.json', '.csv']
            # Otherwise treat as direct text input
            return len(input_data.strip()) > 0
        elif isinstance(input_data, bytes):
            try:
                input_data.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
        elif isinstance(input_data, Path):
            return input_data.suffix.lower() in ['.txt', '.md', '.json', '.csv']
        return False
    
    def _looks_like_file_path(self, text: str) -> bool:
        """Check if a string looks like a file path."""
        # Check for common file path indicators
        if '/' in text or '\\' in text:
            return True
        # Check for file extensions
        if '.' in text and len(text.split('.')[-1]) <= 4:
            return True
        return False
    
    def process(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """Process text input and extract structured information."""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Convert input to string
        text_content = self._extract_text(input_data)
        
        # Extract metadata
        metadata = self.extract_metadata(input_data)
        metadata.update({
            "text_length": len(text_content),
            "word_count": len(text_content.split()),
            "sentences": len(nltk.sent_tokenize(text_content))
        })
        
        # Process with GPT-3.5-turbo
        processed_content = self._process_with_gpt(text_content, **kwargs)
        
        return ProcessingResult(
            content=processed_content,
            metadata=metadata,
            confidence=0.9,  # High confidence for text processing
            modality="text",
            processed_data={
                "original_text": text_content,
                "keywords": self._extract_keywords(text_content),
                "entities": self._extract_entities(text_content),
                "sentiment": self._analyze_sentiment(text_content)
            }
        )
    
    def _extract_text(self, input_data: Union[str, bytes, Path]) -> str:
        """Extract text from various input types."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, bytes):
            return input_data.decode('utf-8')
        elif isinstance(input_data, Path):
            return input_data.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _process_with_gpt(self, text: str, **kwargs) -> str:
        """Process text using available AI models (OpenAI, Anthropic, or Google)."""
        # Check if text is too long and needs chunking
        if len(text) > 50000:  # Approximately 12K tokens
            self.logger.info("📄 Large text detected, using chunked processing")
            return self._process_large_text(text, **kwargs)
        
        prompt = self._create_processing_prompt(text, **kwargs)
        
        # Try OpenAI first
        if self.openai_client:
            try:
                self.logger.info("🤖 Using OpenAI GPT for text processing")
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes and structures text queries for further analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.error(f"❌ OpenAI error: {e}")
                # If it's a rate limit error for large text, try chunking
                if "Request too large" in str(e) or "rate_limit_exceeded" in str(e):
                    self.logger.info("📄 Retrying with chunked processing due to size limits")
                    return self._process_large_text(text, **kwargs)
        
        # Try Anthropic Claude as fallback
        if self.anthropic_client:
            try:
                self.logger.info("🤖 Using Anthropic Claude as fallback")
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",  # Cheaper model
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text.strip()
            except Exception as e:
                self.logger.error(f"❌ Anthropic error: {e}")
        
        # Try Google Gemini as final fallback
        if self.google_client:
            try:
                self.logger.info("🤖 Using Google Gemini as fallback")
                response = self.google_client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                self.logger.error(f"❌ Google error: {e}")
        
        # All AI models failed, use basic processing
        self.logger.warning("❌ All AI models unavailable, using basic processing")
        return self._basic_text_processing(text)
    
    def _process_large_text(self, text: str, **kwargs) -> str:
        """Process large text by chunking it into smaller pieces."""
        try:
            # Split text into chunks of approximately 40K characters (about 10K tokens)
            chunk_size = 40000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            self.logger.info(f"📄 Processing {len(chunks)} chunks")
            
            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                try:
                    prompt = f"Analyze this part ({i+1}/{len(chunks)}) of a larger document:\n\n{chunk}\n\nProvide a brief summary of the key points in this section."
                    
                    if self.openai_client:
                        response = self.openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are analyzing part of a larger document. Focus on the key points and main content."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,  # Smaller output for chunks
                            temperature=0.3
                        )
                        chunk_results.append(f"Section {i+1}: {response.choices[0].message.content.strip()}")
                    else:
                        # Fallback for chunks
                        chunk_results.append(f"Section {i+1}: Basic analysis - {len(chunk)} characters")
                        
                except Exception as e:
                    self.logger.warning(f"❌ Error processing chunk {i+1}: {e}")
                    chunk_results.append(f"Section {i+1}: Processing failed")
            
            # Combine results
            final_summary = f"Document Analysis Summary (Total: {len(text)} characters, {len(chunks)} sections):\n\n" + "\n\n".join(chunk_results)
            return final_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error in chunked processing: {e}")
            return self._basic_text_processing(text)
    
    def _create_processing_prompt(self, text: str, **kwargs) -> str:
        """Create a prompt for GPT processing."""
        task = kwargs.get('task', 'general')
        
        prompts = {
            'general': f"""
            Analyze the following text and provide a structured summary:
            
            Text: {text}
            
            Please provide:
            1. Main topic/subject
            2. Key points or questions
            3. Any specific requests or intents
            4. Relevant context or background information
            
            Format your response as a clear, structured summary.
            """,
            'question': f"""
            Analyze this question and extract key components:
            
            Question: {text}
            
            Please identify:
            1. Question type (what, how, why, when, where, who)
            2. Main subject/topic
            3. Specific requirements or constraints
            4. Expected answer format
            
            Format as a structured analysis.
            """,
            'command': f"""
            Analyze this command or instruction:
            
            Command: {text}
            
            Please identify:
            1. Action required
            2. Target or subject
            3. Parameters or conditions
            4. Expected outcome
            
            Format as a structured command analysis.
            """
        }
        
        return prompts.get(task, prompts['general'])
    
    def _basic_text_processing(self, text: str) -> str:
        """Fallback text processing without GPT."""
        sentences = nltk.sent_tokenize(text)
        words = text.split()
        
        return f"""
        Text Analysis Summary:
        - Length: {len(text)} characters
        - Words: {len(words)}
        - Sentences: {len(sentences)}
        - Main content: {text[:200]}{'...' if len(text) > 200 else ''}
        """
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 10 keywords
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple entity extraction (can be enhanced with NER)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'poor', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
