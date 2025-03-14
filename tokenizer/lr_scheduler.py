import numpy as np
import math

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)

class LambdaWarmUpConstantScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_max, lr_start, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            lr = self.lr_max
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)
    

class LambdaWarmUpDecayScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_max, lr_min, lr_start, decay=0.8, interval=10000, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.last_lr = lr_start
        self.verbosity_interval = verbosity_interval
        self.interval = interval
        self.decay = decay
        self.lr_min = lr_min

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            if (n - self.lr_warm_up_steps) % self.interval == 0 and n > self.lr_warm_up_steps:
                lr = max(self.last_lr * self.decay, self.lr_min)
                self.last_lr = lr
            else:
                lr = self.last_lr
            return lr

    def __call__(self, n):
        return self.schedule(n) 
