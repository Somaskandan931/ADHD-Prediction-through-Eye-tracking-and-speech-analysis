# logger_utils.py

import logging

def setup_logger():
    logger = logging.getLogger("adhd_backend")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("adhd_backend.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
