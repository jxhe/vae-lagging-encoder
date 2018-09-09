import sys
class Logger(object):
    def __init__(self, log_name):
        self.terminal = sys.stdout
        self.log = open(log_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()