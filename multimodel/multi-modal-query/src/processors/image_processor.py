"""
Image processor for handling image-based queries.
Uses GPT-4V for visual analysis and GPT-3.5-turbo for text processing.
"""

import base64
import io
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import openai
from PIL import Image
import cv2
import numpy as np
from .base_processor import BaseProcessor, ProcessingResult

class ImageProcessor(BaseProcessor):
    """Processes image queries and extracts visual information."""
    
    def __init__(self, api_key: Optional[str] = None, vision_model: str = "gpt-4o", text_model: str = "gpt-4o"):
        super().__init__(model_name=vision_model)
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
        
        self.vision_model = vision_model
        self.text_model = text_model
        self.logger.info(f"🔧 ImageProcessor initialized with vision_model: {vision_model}, text_model: {text_model}")
    
    def can_process(self, input_data: Union[str, bytes, Path]) -> bool:
        """Check if input is image-based."""
        if isinstance(input_data, bytes):
            try:
                Image.open(io.BytesIO(input_data))
                return True
            except:
                return False
        elif isinstance(input_data, Path):
            return input_data.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        elif isinstance(input_data, str):
            # Check if it's a base64 encoded image or URL
            if input_data.startswith('data:image') or input_data.startswith('http'):
                return True
        return False
    
    def process(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """Process image input and extract visual information."""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Load and preprocess image
        image = self._load_image(input_data)
        
        # Extract metadata
        metadata = self.extract_metadata(input_data)
        metadata.update({
            "image_size": image.size,
            "image_mode": image.mode,
            "image_format": image.format
        })
        
        # Process with GPT-4V
        visual_analysis = self._analyze_with_gpt4v(image, **kwargs)
        
        # Further process with GPT-3.5-turbo for structured output
        structured_content = self._structure_with_gpt35(visual_analysis, **kwargs)
        
        # Convert numpy types to Python types for JSON serialization
        image_features = self._extract_image_features(image)
        if image_features and 'dominant_colors' in image_features:
            image_features['dominant_colors'] = [tuple(map(int, color)) for color in image_features['dominant_colors']]
        
        return ProcessingResult(
            content=structured_content,
            metadata=metadata,
            confidence=0.85,  # Good confidence for image processing
            modality="image",
            processed_data={
                "visual_analysis": visual_analysis,
                "image_features": image_features,
                "objects_detected": self._detect_objects(image),
                "text_in_image": self._extract_text_from_image(image)
            }
        )
    
    def _load_image(self, input_data: Union[str, bytes, Path]) -> Image.Image:
        """Load image from various input types."""
        if isinstance(input_data, bytes):
            return Image.open(io.BytesIO(input_data))
        elif isinstance(input_data, Path):
            return Image.open(input_data)
        elif isinstance(input_data, str):
            if input_data.startswith('data:image'):
                # Handle base64 encoded image
                header, encoded = input_data.split(",", 1)
                image_data = base64.b64decode(encoded)
                return Image.open(io.BytesIO(image_data))
            elif input_data.startswith('http'):
                # Handle URL (would need requests library)
                import requests
                response = requests.get(input_data)
                return Image.open(io.BytesIO(response.content))
            else:
                # Assume it's a file path
                return Image.open(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _analyze_with_gpt4v(self, image: Image.Image, **kwargs) -> str:
        """Analyze image using GPT-4V."""
        if not self.client:
            self.logger.warning("No OpenAI client available, using basic image analysis")
            return self._basic_image_analysis(image)
            
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create prompt based on task
            task = kwargs.get('task', 'general')
            prompts = {
                'general': "Analyze this image and describe what you see in detail. Include objects, people, text, colors, and any relevant context.",
                'object_detection': "Identify and list all objects visible in this image. Include their locations and any relevant details.",
                'text_extraction': "Extract and read any text visible in this image. Include signs, labels, documents, or any written content.",
                'scene_analysis': "Describe the scene, setting, and context of this image. What is happening and where?",
                'document_analysis': "If this appears to be a document, extract the key information, structure, and content."
            }
            
            prompt = prompts.get(task, prompts['general'])
            
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error analyzing image with GPT-4V: {e}")
            return self._basic_image_analysis(image)
    
    def _structure_with_gpt35(self, visual_analysis: str, **kwargs) -> str:
        """Structure the visual analysis using GPT-3.5-turbo."""
        if not self.client:
            self.logger.warning("No OpenAI client available, returning visual analysis as-is")
            return visual_analysis
            
        try:
            prompt = f"""
            Structure the following visual analysis into a clear, organized format:
            
            Visual Analysis: {visual_analysis}
            
            Please organize this into:
            1. Main subjects/objects
            2. Scene description
            3. Text content (if any)
            4. Key details and context
            5. Relevant keywords for search/processing
            
            Format as a structured summary suitable for further processing.
            """
            
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that structures visual analysis for further processing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error structuring with GPT-3.5: {e}")
            return visual_analysis
    
    def _basic_image_analysis(self, image: Image.Image) -> str:
        """Fallback image analysis without GPT-4V."""
        return f"""
        Basic Image Analysis:
        - Size: {image.size}
        - Mode: {image.mode}
        - Format: {image.format}
        - Dominant colors: {self._get_dominant_colors(image)}
        - Brightness: {self._analyze_brightness(image)}
        """
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image features."""
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        features = {
            "size": image.size,
            "aspect_ratio": float(image.size[0] / image.size[1]),  # Ensure Python float
            "brightness": self._analyze_brightness(image),
            "dominant_colors": self._get_dominant_colors(image),
            "edges": self._detect_edges(img_array),  # Already converted to int in method
            "blur_level": self._analyze_blur(img_array)
        }
        
        return features
    
    def _detect_objects(self, image: Image.Image) -> List[str]:
        """Basic object detection using OpenCV."""
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    objects.append(f"object_{len(objects)}")
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR (basic implementation)."""
        try:
            # This would typically use Tesseract OCR
            # For now, return a placeholder
            return "Text extraction not implemented (requires Tesseract OCR)"
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[tuple]:
        """Get dominant colors from image."""
        try:
            # Resize for faster processing
            small_image = image.resize((150, 150))
            colors = small_image.getcolors(150*150)
            
            if colors:
                # Sort by frequency and get top colors
                colors.sort(key=lambda x: x[0], reverse=True)
                # Ensure color values are Python ints, not numpy types
                return [tuple(int(c) if hasattr(c, 'item') else c for c in color[1]) for color in colors[:num_colors]]
            
            return []
        except Exception as e:
            self.logger.error(f"Error getting dominant colors: {e}")
            return []
    
    def _analyze_brightness(self, image: Image.Image) -> str:
        """Analyze image brightness."""
        try:
            gray = image.convert('L')
            pixels = list(gray.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            
            if avg_brightness < 85:
                return "dark"
            elif avg_brightness > 170:
                return "bright"
            else:
                return "medium"
        except Exception as e:
            self.logger.error(f"Error analyzing brightness: {e}")
            return "unknown"
    
    def _detect_edges(self, img_array: np.ndarray) -> int:
        """Detect edges in image."""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return int(np.sum(edges > 0))  # Convert numpy int64 to Python int
        except Exception as e:
            self.logger.error(f"Error detecting edges: {e}")
            return 0
    
    def _analyze_blur(self, img_array: np.ndarray) -> str:
        """Analyze image blur level."""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # Convert to Python float
            
            if laplacian_var < 100:
                return "blurry"
            elif laplacian_var > 500:
                return "sharp"
            else:
                return "moderate"
        except Exception as e:
            self.logger.error(f"Error analyzing blur: {e}")
            return "unknown"
