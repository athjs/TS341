"""Nothing for the moment only to please docstring :)."""

import logging

logger = logging.getLogger(__name__)


def main():
    """Only showing logs for the moment."""
    logging.basicConfig(filename="myapp.log", level=logging.INFO)
    logger.info("Started")
    logger.info("Finished")
    print("hello world!")


main()
