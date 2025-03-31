from loguru import logger
from vidore_benchmark.utils.logging_utils import setup_logging

def test_logging():
    """
    Test function to demonstrate proper loguru usage.
    
    Run this function to test if logging is working properly:
    
    >>> from vidore_benchmark.utils.logger_example import test_logging
    >>> test_logging()
    """
    # These messages will only appear if the log level is set accordingly
    logger.debug("This is a DEBUG message - only visible at debug level")
    logger.info("This is an INFO message - visible at info level and below")
    logger.warning("This is a WARNING message - visible at warning level and below")
    logger.error("This is an ERROR message - visible at error level and below")
    logger.critical("This is a CRITICAL message - always visible")
    
    # This demonstrates how to use formatting in log messages
    user_input = "test_user"
    logger.info(f"Processing input for user: {user_input}")
    
    # This demonstrates how to log exceptions with traceback
    try:
        result = 1 / 0
    except Exception as e:
        logger.exception("An error occurred during calculation")
        
    # Using structured logging with extra context
    logger.bind(user="admin", module="example").info("This is a structured log message with context")
        
    print("Logging test complete. Check the logs to see which messages appeared.")

if __name__ == "__main__":
    # Setup logging when run directly
    setup_logging(log_level="DEBUG")
    test_logging()
