import sys

# orig_stdout = sys.stdout
# orig_stderr = sys.stderr


class log2file(object):
    """
    logs to console and file using the python logging module
    Usage:
    # set up logger at main Script
    logging.basicConfig(level=logging.DEBUG, filename="testout_logger.out",
    filemode="a", format="%(message)s")
    log = logging.getLogger(__name__)
    sys.stdout = log2file(log, logging.INFO, orig_stdout)
    sys.stderr = log2file(log, logging.INFO, orig_stdout)
    print("stuff")
    """

    def __init__(self, logger, level, stream):
        """
        Set up things
        """
        self.logger = logger
        self.level = level
        self.linebuf = ""
        self.stream = stream

    def write(self, data):
        """
        writes output
        """
        for line in data.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

        self.stream.write(data)

    def flush(self):
        pass
