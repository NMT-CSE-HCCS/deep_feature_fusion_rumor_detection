import random
import time


def get_timebaesed_random_seed():
    """return a timebased random seed"""
    random.seed(time.time())
    seed = random.randint(0,100000)
    return seed

