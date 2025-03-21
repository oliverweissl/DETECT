import logging
import sys


def enable_logging() -> None:
    """Allow logging outputs in notebooks."""
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
    )
    logger = logging.getLogger()
    logger.handlers[0].stream = sys.stdout
