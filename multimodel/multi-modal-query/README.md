# Multimodal Query Processing System

A comprehensive system for processing multimodal queries (text, images, audio, and documents) using GPT models. This system provides a unified interface for analyzing various types of input and extracting structured information for further processing by your orchestrator.

## Features

- **Multi-modal Support**: Process text, images, audio, and documents
- **Cost-Effective**: Uses GPT-3.5-turbo for text processing and GPT-4V for image analysis
- **Modular Architecture**: Separate processors for each modality
- **Fallback Mechanisms**: Graceful degradation when specific processors fail
- **Structured Output**: Consistent JSON responses with metadata
- **Command-line Interface**: Easy-to-use CLI for processing queries
- **Interactive Mode**: Interactive shell for testing and development

## Supported Formats

### Text
- Plain text
- Markdown
- JSON
- CSV

### Images
- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP
- GIF

### Audio
- MP3
- WAV
- M4A
- FLAC
- OGG
- AAC

### Documents
- PDF
- DOCX/DOC
- TXT
- CSV
- JSON
- XML

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Query-Processing
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Process text query
python main.py "What is the weather like today?"

# Process image file
python main.py image.jpg

# Process audio file
python main.py audio.wav

# Process document
python main.py document.pdf
```

#### Advanced Options
```bash
# Use specific model
python main.py "Analyze this text" --model gpt-4

# Set maximum tokens
python main.py "Long text..." --max-tokens 2000

# Specify task type
python main.py "Question here" --task question

# Save output to file
python main.py input.txt --output result.json

# Verbose logging
python main.py input.jpg --verbose

# Health check
python main.py --health-check
```

#### Interactive Mode
```bash
python main.py
```
This starts an interactive session where you can enter queries one by one.

### Programmatic Usage

```python
from src.orchestrator import MultimodalOrchestrator, OrchestratorConfig

# Create orchestrator
config = OrchestratorConfig(
    openai_api_key="your_api_key",
    gpt_model="gpt-3.5-turbo",
    vision_model="gpt-4-vision-preview"
)
orchestrator = MultimodalOrchestrator(config)

# Process text
result = orchestrator.process_query("What is machine learning?")

# Process image
with open("image.jpg", "rb") as f:
    image_data = f.read()
result = orchestrator.process_query(image_data)

# Process multiple inputs
inputs = ["text query", image_data, "another query"]
results = orchestrator.process_multiple(inputs)

# Get processor information
info = orchestrator.get_processor_info()

# Health check
health = orchestrator.health_check()
```

## API Reference

### OrchestratorConfig

Configuration class for the orchestrator:

```python
@dataclass
class OrchestratorConfig:
    openai_api_key: Optional[str] = None
    gpt_model: str = "gpt-3.5-turbo"
    vision_model: str = "gpt-4-vision-preview"
    max_tokens: int = 1000
    enable_fallback: bool = True
    log_level: str = "INFO"
```

### ProcessingResult

Result structure returned by processors:

```python
@dataclass
class ProcessingResult:
    content: str                    # Processed content
    metadata: Dict[str, Any]        # Processing metadata
    confidence: float               # Confidence score (0-1)
    modality: str                   # Input modality
    processed_data: Optional[Dict[str, Any]] = None  # Additional data
```

### Available Processors

- **TextProcessor**: Handles text queries
- **ImageProcessor**: Handles image analysis
- **AudioProcessor**: Handles audio transcription and analysis
- **DocumentProcessor**: Handles document parsing and analysis

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Model Selection

- **Text Processing**: GPT-3.5-turbo (default) or GPT-4
- **Image Processing**: GPT-4V for visual analysis, GPT-3.5-turbo for text processing
- **Audio Processing**: Whisper-1 for transcription, GPT-3.5-turbo for analysis
- **Document Processing**: GPT-3.5-turbo for content analysis

## Error Handling

The system includes comprehensive error handling:

- **Fallback Processing**: If a specific processor fails, falls back to text processing
- **Graceful Degradation**: Continues processing even if some features fail
- **Detailed Error Messages**: Provides clear error information
- **Health Checks**: Monitor system status

## Examples

### Text Analysis
```bash
python main.py "What are the key benefits of renewable energy?"
```

Output:
```json
{
  "success": true,
  "content": "Renewable energy offers several key benefits...",
  "modality": "text",
  "confidence": 0.9,
  "metadata": {
    "processor": "TextProcessor",
    "model": "gpt-3.5-turbo",
    "text_length": 45,
    "word_count": 8
  }
}
```

### Image Analysis
```bash
python main.py photo.jpg
```

Output:
```json
{
  "success": true,
  "content": "This image shows a mountain landscape...",
  "modality": "image",
  "confidence": 0.85,
  "metadata": {
    "processor": "ImageProcessor",
    "image_size": [1920, 1080],
    "image_format": "JPEG"
  },
  "processed_data": {
    "visual_analysis": "Detailed visual description...",
    "objects_detected": ["mountain", "trees", "sky"]
  }
}
```

### Audio Processing
```bash
python main.py audio.wav
```

Output:
```json
{
  "success": true,
  "content": "The audio contains a question about...",
  "modality": "audio",
  "confidence": 0.8,
  "metadata": {
    "processor": "AudioProcessor",
    "duration": 5.2,
    "sample_rate": 44100
  },
  "processed_data": {
    "transcription": "What is the capital of France?",
    "language_detection": "english"
  }
}
```

## Development

### Project Structure
```
Query-Processing/
├── src/
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base_processor.py
│   │   ├── text_processor.py
│   │   ├── image_processor.py
│   │   ├── audio_processor.py
│   │   └── document_processor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py
│   ├── __init__.py
│   └── orchestrator.py
├── main.py
├── requirements.txt
├── README.md
└── .env
```

### Adding New Processors

1. Create a new processor class inheriting from `BaseProcessor`
2. Implement required methods: `can_process()` and `process()`
3. Add the processor to the orchestrator's processor dictionary
4. Update the `__init__.py` file in the processors module

### Testing

```bash
# Run health check
python main.py --health-check

# Test with sample files
python main.py "test query"
python main.py sample_image.jpg
python main.py sample_audio.wav
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `OPENAI_API_KEY` is set in environment or `.env` file
2. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
3. **File Not Found**: Check file paths and ensure files exist
4. **Memory Issues**: For large files, consider processing in chunks

### Logging

Enable verbose logging for debugging:
```bash
python main.py input.jpg --verbose
```

### Performance Tips

- Use appropriate model sizes for your use case
- Process large files in chunks
- Cache results for repeated queries
- Use batch processing for multiple inputs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on GitHub
4. Check the logs with verbose mode
