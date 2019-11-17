import os
import logging

logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)


class Logger:
    # log path
    dir_log = '_log'

    def __init__(self, name='_log'):
        formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')

        handler1 = logging.StreamHandler()
        handler1.setFormatter(formatter)

        handler2 = logging.FileHandler(filename=os.path.join(self.dir_log, '{}.log'.format(name)))
        handler2.setFormatter((formatter))

        logger.addHandler(handler1)
        logger.addHandler(handler2)

    def info(self, msg):
        logger.info(msg)
