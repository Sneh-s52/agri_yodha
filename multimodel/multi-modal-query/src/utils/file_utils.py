"""
Utility functions for file handling and validation.
"""

import os
import mimetypes
from pathlib import Path
from typing import Union, Optional, Dict, Any
import hashlib

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive file information."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    
    info = {
        "name": file_path.name,
        "path": str(file_path.absolute()),
        "size": stat.st_size,
        "extension": file_path.suffix.lower(),
        "mime_type": mimetypes.guess_type(str(file_path))[0],
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "readable": os.access(file_path, os.R_OK),
        "writable": os.access(file_path, os.W_OK)
    }
    
    return info

def validate_file_type(file_path: Union[str, Path], allowed_extensions: list) -> bool:
    """Validate if file has an allowed extension."""
    file_path = Path(file_path)
    return file_path.suffix.lower() in allowed_extensions

def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """Calculate file hash."""
    file_path = Path(file_path)
    
    if algorithm == "md5":
        hash_func = hashlib.md5()
    elif algorithm == "sha1":
        hash_func = hashlib.sha1()
    elif algorithm == "sha256":
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def is_supported_file_type(file_path: Union[str, Path]) -> bool:
    """Check if file type is supported by the system."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    supported_extensions = {
        # Images
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif',
        # Audio
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac',
        # Documents
        '.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml',
        # Video (for future expansion)
        '.mp4', '.avi', '.mov', '.mkv'
    }
    
    return extension in supported_extensions

def get_file_category(file_path: Union[str, Path]) -> Optional[str]:
    """Get the category of a file based on its extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    categories = {
        # Images
        'image': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'},
        # Audio
        'audio': {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'},
        # Documents
        'document': {'.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml'},
        # Video
        'video': {'.mp4', '.avi', '.mov', '.mkv'}
    }
    
    for category, extensions in categories.items():
        if extension in extensions:
            return category
    
    return None

def create_temp_file(content: Union[str, bytes], extension: str = ".tmp") -> Path:
    """Create a temporary file with given content."""
    import tempfile
    
    mode = 'wb' if isinstance(content, bytes) else 'w'
    encoding = None if isinstance(content, bytes) else 'utf-8'
    
    with tempfile.NamedTemporaryFile(mode=mode, suffix=extension, delete=False, encoding=encoding) as f:
        f.write(content)
        return Path(f.name)

def cleanup_temp_files(temp_files: list):
    """Clean up temporary files."""
    for temp_file in temp_files:
        try:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
        except Exception:
            pass  # Ignore cleanup errors
