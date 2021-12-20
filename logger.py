import json
import os

class Logger(object):
    def __init__(self, log_dir, env_name, seed, epochs):
        self.logs = {
            'returns' : [],
        }
        path = log_dir+'/'+str(env_name)+'/'
        self.seed = str(seed)
        self.total = epochs
        if not os.path.exists(path):
            os.makedirs(path)
        self.filename = path+self.seed
    
    def log(self, stats):
        self.logs['returns'].append(stats[0])
        with open(self.filename+'.json', 'w') as fp:
            json.dump(self.logs, fp)
    
    def print(self, steps):
        print("----------------------------------------")
        print('\tIterations Completed:', steps, '/', self.total)
        print('\tAverage Return:', self.logs['returns'][-1])
        print('\tAverage Cost:', self.logs['cost'][-1])
        print("----------------------------------------")