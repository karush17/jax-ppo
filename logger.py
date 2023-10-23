"""Implements the custom logger for storing results."""

import json
import os

class Logger:
    """Implements a custom logger object.
    
    Attributes:
        logs: logger dictionary of results.
        path: path for saving logs.
        seed: random seed.
        total: total training steps of the agent.
        filename: filename for saving logs.
    """
    def __init__(self, log_dir: str, env_name: str, seed: int, epochs: int):
        """Initializes the logger object."""
        self.logs = {
            'returns' : [],
        }
        path = log_dir+'/'+str(env_name)+'/'
        self.seed = str(seed)
        self.total = epochs
        if not os.path.exists(path):
            os.makedirs(path)
        self.filename = path+self.seed

    def log(self, stats: list):
        """Logs the input to a json file."""
        self.logs['returns'].append(stats[0])
        with open(self.filename+'.json', 'w') as fp:
            json.dump(self.logs, fp)

    def print(self, steps: int):
        """Prints the current logs to terminal."""
        print("----------------------------------------")
        print('\tIterations Completed:', steps, '/', self.total)
        print('\tAverage Return:', self.logs['returns'][-1])
        print('\tAverage Cost:', self.logs['cost'][-1])
        print("----------------------------------------")
