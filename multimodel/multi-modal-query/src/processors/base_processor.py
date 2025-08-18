"""
Base processor class for multimodal query processing.
All processors should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Result of processing a multimodal input."""
    content: str
    metadata: Dict[str, Any]
    confidence: float
    modality: str
    processed_data: Optional[Dict[str, Any]] = None

class BaseProcessor(ABC):
    """Base class for all multimodal processors."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_tokens: int = 1000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def can_process(self, input_data: Union[str, bytes, Path]) -> bool:
        """Check if this processor can handle the given input."""
        pass
    
    @abstractmethod
    def process(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """Process the input and return a structured result."""
        pass
    
    def validate_input(self, input_data: Union[str, bytes, Path]) -> bool:
        """Validate input before processing."""
        if input_data is None:
            return False
        return True
    
    def extract_metadata(self, input_data: Union[str, bytes, Path]) -> Dict[str, Any]:
        """Extract metadata from input data."""
        metadata = {
            "processor": self.__class__.__name__,
            "model": self.model_name,
            "input_type": type(input_data).__name__
        }
        
        if isinstance(input_data, Path):
            metadata.update({
                "file_path": str(input_data),
                "file_size": input_data.stat().st_size if input_data.exists() else 0,
                "file_extension": input_data.suffix
            })
        
        return metadata
