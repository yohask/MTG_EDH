import logging
import sys
import traceback

# Set up a root logger to file and console
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

fh = logging.FileHandler('pipeline_debug.log', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Log uncaught exceptions

def log_uncaught_exceptions(exctype, value, tb):
    logger.error("Uncaught exception:", exc_info=(exctype, value, tb))

sys.excepthook = log_uncaught_exceptions

logger.info("Logger initialized. Any errors or debug output will be written to pipeline_debug.log.")
