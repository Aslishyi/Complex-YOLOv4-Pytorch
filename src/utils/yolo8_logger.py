### Modified logger.py to support YOLOv8 training logs ###

import logging
import os

class Logger:
    def __init__(self, logs_dir, log_name='train_log'):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # Create log directory if it does not exist
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # File handler to save logs
        log_file_path = os.path.join(logs_dir, f"{log_name}.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Console handler to output logs to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)