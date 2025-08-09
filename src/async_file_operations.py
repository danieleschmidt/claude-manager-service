"""
Async File Operations for Claude Manager Service

This module provides async file I/O operations for configuration loading,
log file operations, and other file-based tasks with non-blocking I/O.

Features:
- Async file reading and writing
- JSON configuration loading with async I/O
- Non-blocking file operations using aiofiles
- Error handling and performance monitoring
- Backward compatibility with synchronous operations
"""

import asyncio
import json
import aiofiles
from typing import Dict, Any, Optional, List
from pathlib import Path

from .logger import get_logger
from .performance_monitor import monitor_performance
from .error_handler import FileOperationError, JsonParsingError


logger = get_logger(__name__)


async def read_file_async(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read file contents asynchronously
    
    Args:
        file_path: Path to file to read
        encoding: File encoding
        
    Returns:
        File contents as string
        
    Raises:
        FileOperationError: For file access issues
    """
    try:
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            content = await f.read()
        
        logger.debug(f"Successfully read file: {file_path}")
        return content
        
    except FileNotFoundError as e:
        raise FileOperationError(str(e), file_path, "async_read", e)
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "async_read", e)
    except UnicodeDecodeError as e:
        raise FileOperationError(f"Encoding error: {str(e)}", file_path, "async_read", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "async_read", e)


async def write_file_async(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """
    Write file contents asynchronously
    
    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: File encoding
        
    Raises:
        FileOperationError: For file operation issues
    """
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
            await f.write(content)
        
        logger.debug(f"Successfully wrote file: {file_path}")
        
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "async_write", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "async_write", e)


async def load_json_async(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file asynchronously
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileOperationError: For file access issues
        JsonParsingError: For JSON parsing issues
    """
    try:
        content = await read_file_async(file_path)
        data = json.loads(content)
        
        logger.debug(f"Successfully loaded JSON: {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        raise JsonParsingError(str(e), file_path, e)
    except FileOperationError:
        raise  # Re-raise file operation errors


async def save_json_async(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Save data to JSON file asynchronously
    
    Args:
        file_path: Path to JSON file
        data: Data to save
        indent: JSON indentation
        
    Raises:
        FileOperationError: For file operation issues
    """
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        await write_file_async(file_path, content)
        
        logger.debug(f"Successfully saved JSON: {file_path}")
        
    except (TypeError, ValueError) as e:
        raise JsonParsingError(f"JSON serialization error: {str(e)}", file_path, e)


async def read_lines_async(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    Read file lines asynchronously
    
    Args:
        file_path: Path to file to read
        encoding: File encoding
        
    Returns:
        List of lines from file
        
    Raises:
        FileOperationError: For file access issues
    """
    try:
        lines = []
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            async for line in f:
                lines.append(line.rstrip('\n\r'))
        
        logger.debug(f"Successfully read {len(lines)} lines from: {file_path}")
        return lines
        
    except FileNotFoundError as e:
        raise FileOperationError(str(e), file_path, "async_read_lines", e)
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "async_read_lines", e)
    except UnicodeDecodeError as e:
        raise FileOperationError(f"Encoding error: {str(e)}", file_path, "async_read_lines", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "async_read_lines", e)


async def write_lines_async(file_path: str, lines: List[str], encoding: str = 'utf-8') -> None:
    """
    Write lines to file asynchronously
    
    Args:
        file_path: Path to file to write
        lines: Lines to write
        encoding: File encoding
        
    Raises:
        FileOperationError: For file operation issues
    """
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
            for line in lines:
                await f.write(line + '\n')
        
        logger.debug(f"Successfully wrote {len(lines)} lines to: {file_path}")
        
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "async_write_lines", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "async_write_lines", e)


async def append_file_async(file_path: str, content: str, encoding: str = 'utf-8') -> None:
    """
    Append content to file asynchronously
    
    Args:
        file_path: Path to file to append to
        content: Content to append
        encoding: File encoding
        
    Raises:
        FileOperationError: For file operation issues
    """
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'a', encoding=encoding) as f:
            await f.write(content)
        
        logger.debug(f"Successfully appended to file: {file_path}")
        
    except PermissionError as e:
        raise FileOperationError(str(e), file_path, "async_append", e)
    except OSError as e:
        raise FileOperationError(f"OS error: {str(e)}", file_path, "async_append", e)


async def file_exists_async(file_path: str) -> bool:
    """
    Check if file exists asynchronously
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists
    """
    try:
        # Use thread executor for os.path operations to avoid blocking
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(None, Path(file_path).exists)
        return exists
        
    except Exception as e:
        logger.warning(f"Error checking file existence for {file_path}: {e}")
        return False


async def get_file_size_async(file_path: str) -> Optional[int]:
    """
    Get file size asynchronously
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if error
    """
    try:
        loop = asyncio.get_event_loop()
        size = await loop.run_in_executor(None, lambda: Path(file_path).stat().st_size)
        return size
        
    except Exception as e:
        logger.warning(f"Error getting file size for {file_path}: {e}")
        return None


async def bulk_read_files_async(file_paths: List[str], encoding: str = 'utf-8') -> Dict[str, str]:
    """
    Read multiple files concurrently
    
    Args:
        file_paths: List of file paths to read
        encoding: File encoding
        
    Returns:
        Dictionary mapping file paths to their contents
    """
    logger.info(f"Reading {len(file_paths)} files concurrently")
    
    async def read_single_file(file_path: str) -> tuple[str, Optional[str]]:
        try:
            content = await read_file_async(file_path, encoding)
            return file_path, content
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return file_path, None
    
    # Create concurrent tasks
    with monitor_performance("bulk_read_files"):
        tasks = [read_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
    
    # Build result dictionary
    file_contents = {}
    successful_reads = 0
    
    for file_path, content in results:
        file_contents[file_path] = content
        if content is not None:
            successful_reads += 1
    
    logger.info(f"Successfully read {successful_reads}/{len(file_paths)} files")
    return file_contents


async def bulk_write_files_async(file_data: Dict[str, str], encoding: str = 'utf-8') -> List[bool]:
    """
    Write multiple files concurrently
    
    Args:
        file_data: Dictionary mapping file paths to content
        encoding: File encoding
        
    Returns:
        List of success flags
    """
    logger.info(f"Writing {len(file_data)} files concurrently")
    
    async def write_single_file(file_path: str, content: str) -> bool:
        try:
            await write_file_async(file_path, content, encoding)
            return True
        except Exception as e:
            logger.warning(f"Failed to write file {file_path}: {e}")
            return False
    
    # Create concurrent tasks
    with monitor_performance("bulk_write_files"):
        tasks = [write_single_file(path, content) for path, content in file_data.items()]
        results = await asyncio.gather(*tasks)
    
    successful_writes = sum(results)
    logger.info(f"Successfully wrote {successful_writes}/{len(file_data)} files")
    
    return results


# Example usage and testing
async def example_async_file_operations():
    """Example of using async file operations"""
    try:
        # Test basic file operations
        test_file = "test_async.txt"
        test_content = "Hello, async world!"
        
        # Write and read
        await write_file_async(test_file, test_content)
        read_content = await read_file_async(test_file)
        
        logger.info(f"File operation test: {'PASS' if read_content == test_content else 'FAIL'}")
        
        # Test JSON operations
        test_data = {"test": "data", "async": True}
        json_file = "test_async.json"
        
        await save_json_async(json_file, test_data)
        loaded_data = await load_json_async(json_file)
        
        logger.info(f"JSON operation test: {'PASS' if loaded_data == test_data else 'FAIL'}")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        Path(json_file).unlink(missing_ok=True)
        
        logger.info("Async file operations example completed")
        
    except Exception as e:
        logger.error(f"Async file operations example failed: {e}")
        raise


if __name__ == "__main__":
    # Test async file operations
    asyncio.run(example_async_file_operations())