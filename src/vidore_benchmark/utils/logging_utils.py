import os
import sys
from pathlib import Path

# Replace standard logging with loguru
from loguru import logger

# Remove the default loguru handler
logger.remove()

def setup_logging(log_level: str = "WARNING", log_file: str = None) -> None:
    """
    Setup loguru logging configuration with console and optional file output.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file. If None, only console logging is used.
    """
    # Normalize log level to uppercase
    log_level = log_level.upper()
    
    # Validate the log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        print(f"Invalid log level: {log_level}, defaulting to WARNING")
        log_level = "WARNING"
    
    # Create logs directory if a file path is provided
    if log_file is not None:
        log_file_path = Path(log_file)
        log_dir = log_file_path.parent
        if not log_dir.exists():
            os.makedirs(log_dir, exist_ok=True)
    
    # Setup console handler with nice formatting including colors
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file is not None:
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            rotation="10 MB",  # Rotate files when they reach 10MB
            compression="zip",  # Compress rotated files
            encoding="utf-8",
        )
    
    # Log some initial messages to confirm setup
    logger.debug(f"Logging initialized at level: {log_level}")
    logger.info(f"Logging level set to {log_level}")
    # logger.warning(f"This is a test WARNING message - if you can't see this, check your log level")
    
    # Print confirmation to ensure at least some output is visible
    print(f"Loguru initialized: level={log_level}, file={'yes' if log_file else 'no'}")

# Export the logger for use in other modules 
get_logger = lambda name=None: logger.bind(name=name if name else "vidore_benchmark")
