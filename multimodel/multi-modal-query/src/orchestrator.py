"""
Main orchestrator for multimodal query processing.
Coordinates all processors and provides a unified interface.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

from .processors import TextProcessor, ImageProcessor, AudioProcessor, DocumentProcessor
from .processors.base_processor import ProcessingResult

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    openai_api_key: Optional[str] = None
    gpt_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"
    max_tokens: int = 2000
    enable_fallback: bool = True
    log_level: str = "INFO"

class MultimodalOrchestrator:
    """Main orchestrator for processing multimodal queries."""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment if not provided
        api_key = self.config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("No OpenAI API key provided. Some features may not work.")
        
        # Initialize processors
        self.processors = {
            'text': TextProcessor(api_key=api_key, model_name=self.config.gpt_model),
            'image': ImageProcessor(api_key=api_key, vision_model=self.config.vision_model, text_model=self.config.gpt_model),
            'audio': AudioProcessor(api_key=api_key, model_name=self.config.gpt_model),
            'document': DocumentProcessor(api_key=api_key, model_name=self.config.gpt_model)
        }
        
        self.logger.info("Multimodal Orchestrator initialized successfully")
    
    def process_query(self, input_data: Union[str, bytes, Path], **kwargs) -> ProcessingResult:
        """
        Process a multimodal query and return structured results.
        
        Args:
            input_data: The input data (text, image, audio, or document)
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessingResult: Structured result with content, metadata, and confidence
        """
        try:
            # Determine the appropriate processor
            processor = self._select_processor(input_data)
            
            if not processor:
                if self.config.enable_fallback:
                    self.logger.warning("No specific processor found, using text processor as fallback")
                    processor = self.processors['text']
                else:
                    raise ValueError(f"No suitable processor found for input type: {type(input_data)}")
            
            # Process the input
            result = processor.process(input_data, **kwargs)
            
            # Add orchestrator metadata
            result.metadata.update({
                "orchestrator_version": "1.0.0",
                "selected_processor": processor.__class__.__name__,
                "processing_timestamp": self._get_timestamp()
            })
            
            self.logger.info(f"Successfully processed query with {processor.__class__.__name__}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            if self.config.enable_fallback:
                return self._fallback_processing(input_data, str(e))
            else:
                raise
    
    def process_multiple(self, inputs: List[Union[str, bytes, Path]], **kwargs) -> List[ProcessingResult]:
        """
        Process multiple inputs and return results for each.
        
        Args:
            inputs: List of input data
            **kwargs: Additional processing parameters
            
        Returns:
            List[ProcessingResult]: Results for each input
        """
        results = []
        
        for i, input_data in enumerate(inputs):
            try:
                self.logger.info(f"Processing input {i+1}/{len(inputs)}")
                result = self.process_query(input_data, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing input {i+1}: {e}")
                if self.config.enable_fallback:
                    fallback_result = self._fallback_processing(input_data, str(e))
                    results.append(fallback_result)
                else:
                    # Create error result
                    error_result = ProcessingResult(
                        content=f"Error processing input: {str(e)}",
                        metadata={"error": True, "error_message": str(e)},
                        confidence=0.0,
                        modality="error"
                    )
                    results.append(error_result)
        
        return results
    
    def _select_processor(self, input_data: Union[str, bytes, Path]) -> Optional[Any]:
        """Select the appropriate processor for the input data."""
        # Try each processor to see which one can handle the input
        for modality, processor in self.processors.items():
            try:
                if processor.can_process(input_data):
                    self.logger.debug(f"Selected {modality} processor for input")
                    return processor
            except Exception as e:
                self.logger.debug(f"Processor {modality} cannot handle input: {e}")
                continue
        
        return None
    
    def _fallback_processing(self, input_data: Union[str, bytes, Path], error_message: str) -> ProcessingResult:
        """Fallback processing when no specific processor is available."""
        try:
            # Try to convert to string and use text processor
            if isinstance(input_data, bytes):
                try:
                    text_content = input_data.decode('utf-8')
                except UnicodeDecodeError:
                    text_content = f"Binary data (size: {len(input_data)} bytes)"
            elif isinstance(input_data, Path):
                text_content = f"File: {input_data.name} (size: {input_data.stat().st_size} bytes)"
            else:
                text_content = str(input_data)
            
            # Use text processor for fallback
            result = self.processors['text'].process(text_content)
            result.metadata.update({
                "fallback_processing": True,
                "original_error": error_message
            })
            result.confidence *= 0.5  # Reduce confidence for fallback processing
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback processing also failed: {e}")
            # Return minimal result
            return ProcessingResult(
                content=f"Unable to process input. Error: {error_message}",
                metadata={
                    "error": True,
                    "error_message": error_message,
                    "fallback_failed": True
                },
                confidence=0.0,
                modality="error"
            )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_processor_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available processors."""
        info = {}
        for modality, processor in self.processors.items():
            info[modality] = {
                "class": processor.__class__.__name__,
                "model": processor.model_name,
                "max_tokens": processor.max_tokens
            }
        return info
    
    def update_config(self, **kwargs):
        """Update orchestrator configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all processors."""
        health_status = {
            "orchestrator": "healthy",
            "processors": {},
            "api_key_configured": bool(self.config.openai_api_key or os.getenv('OPENAI_API_KEY'))
        }
        
        for modality, processor in self.processors.items():
            try:
                # Simple health check - try to create a minimal processing result
                test_result = ProcessingResult(
                    content="Health check",
                    metadata={"health_check": True},
                    confidence=1.0,
                    modality=modality
                )
                health_status["processors"][modality] = "healthy"
            except Exception as e:
                health_status["processors"][modality] = f"unhealthy: {str(e)}"
                health_status["orchestrator"] = "degraded"
        
        return health_status
