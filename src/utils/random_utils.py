import random
import time


def seed_generator():

    R = random.SystemRandom(
        time.time()
    )

    seed = int(R.random() * 100000)

    return seed
