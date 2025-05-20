import logging
import sys
from app.core.config import LOG_LEVEL


def setup_logging():

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=LOG_LEVEL.upper(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {LOG_LEVEL.upper()} level.")

    theme_parser_logger = logging.getLogger("theme_parser")

    theme_parser_logger.setLevel(LOG_LEVEL.upper())

    logger.info(
        f"Theme parser logger ('theme_parser') configured to follow level: {theme_parser_logger.getEffectiveLevel()}."
    )
