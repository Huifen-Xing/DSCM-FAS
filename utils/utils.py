import torch
# import math
import time
import os

import pickle
from collections import defaultdict
import copy

def mkdirs(dir_):
    if os.path.exists(dir_):
        key = input("{} existed. Rewrite (y/N)?".format(dir_))
        if key.lower() != 'y':
            dir_ = dir_ + time.strftime("-%m%d%H%M")
    os.makedirs(dir_, exist_ok=True)
    return dir_

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class to_Pickle:
    def __init__(self,path):
        self.metrics = defaultdict(lambda:[])
        self.path = os.path.join(path,'logs.pkl')        
        
    def add(self, key, value):
        self.metrics[key].append(value)

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))
        
    def save_logs(self):  
        open_file = open(self.path, "wb")
        pickle.dump(self.get_dict(), open_file)
        open_file.close()
    
    def info(self, epoch, keys):
        output = "Epoch: {:d} ".format(epoch)
        for key in keys:
            output += "{}: {:.5f} ".format(key, self.metrics[key][-1])
        print(output)