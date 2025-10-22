# forge/utils/color_formatter.py
import logging

class ColoredFormatter(logging.Formatter):
    """
    A standard logging formatter.
    """
    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        return super().format(record)

