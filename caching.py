import os
import math
import numpy as np

class CachingHelper:
    def __init__(self, nasconfig, cache=False):
        self.nasconfig = nasconfig
        self.best_reward = -math.inf

        if (cache):
            self.fromcache = self._fromcache
            self.tocache = self._tocache
        else:
            self.fromcache = self._nocache
            self.tocache = self._keepbest

    def getBestState(self):
        return self.best_state

    def _keepbest(self, state, level, loss, acc, test_loss, test_acc):
        if self.best_reward <= acc:
            self.best_reward = acc
            self.best_state = np.copy(state)
            self.best_state[:] = state[:]

    def _nocache(self, state, level):
        # do nothing
        return False,0,0,0,0

    def _fromcache(self, state, level):
        path = self.nasconfig.output_dir + os.sep + "cache" + str(level) + os.sep
        for s in state: path += str(int(s))+os.sep
        filename = path + "res.txt"

        if os.path.exists(filename):
            with open(filename) as f:
                values = []
                for line in f:
                    values.append(float(line))
            incache = True
            loss, acc, test_loss, test_acc = values[0], values[1], values[2], values[3]

        else:
            incache = False
            loss, acc, test_loss, test_acc = 0,0,0,0

        return incache, loss, acc, test_loss, test_acc

    def _tocache(self, state, level, loss, acc, test_loss, test_acc):
        path = self.nasconfig.output_dir+os.sep+"cache"+str(level)+os.sep
        if not os.path.exists(path): os.mkdir(path)
        for s in state:
            path += str(int(s)) + os.sep
            if not os.path.exists(path): os.mkdir(path)
        filename = path + "res.txt"

        self._keepbest(state, level, loss, acc, test_loss, test_acc)

        file = open(filename, "w+")
        file.write(str(loss)+"\n")
        file.write(str(acc)+"\n")
        file.write(str(test_loss)+"\n")
        file.write(str(test_acc)+"\n")
        file.close()



