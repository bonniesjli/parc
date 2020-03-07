'''
USAGE:
======
LOG = logger(logdir)
LOG.create(name)
for __ in learning loop:
    LOG.log(data, timestep)
# LOG.save_data()
# LOG.visualize(some_variable)
LOG.close()
'''
import csv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########## TO DO's ##########
# - seed plotters

class Logger():
    """
    Log data into appropriate files
    """
    def __init__(self, logdir):
        self.logdir = logdir
        self.variable_names = {}
        self.it = 0

    def create(self, variable_name):
        """Create a variable name to be logged"""
        for name in self.variable_names:
            if variable_name == name:
                raise ValueError("variable name already exist")
        self.variable_names[variable_name] = Variable(variable_name)


    def log(self, variable_name, data, t_step):
        """Log data under given variable name"""
        self.variable_names[variable_name].step(data, t_step)
        # for every 200 data points logged, save all data
        self.it += 1
        if self.it % 200 == 0:
            self.save_data()

    def save(self, variable_name):
        """Save data under variable name"""
        d = {'data': self.variable_names[variable_name].data, 'time': self.variable_names[variable_name].t}
        file_name = str(self.logdir) + str("/") + str(variable_name) + str(".csv")
        print ("saving to: ", file_name)
        df = pd.DataFrame(d)
        df.to_csv(file_name, index = False)

    def save_data(self):
        """Save all data"""
        for variable_name in self.variable_names:
            self.save(variable_name)

    def visualize(self, variable_name):
        """Visualize data under variable name"""
        file_name = str(self.logdir) + str("/") + str(variable_name) + str(".png")

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(self.variable_names[variable_name].t, self.variable_names[variable_name].data)
        plt.ylabel(str(variable_name))
        plt.xlabel('Steps')
        plt.savefig(file_name)

    def close(self):
        '''save data and visualize reward data'''
        self.save_data()
        for name in self.variable_names:
            if name == "test_reward":
                self.visualize("test_reward")
            if name == "train_reward":
                self.visualize("train_reward")

class Variable():
    def __init__(self, name):
        self.name = name
        self.data = []
        self.t = []

    def add(self, data_point):
        self.data.append(data_point)

    def step(self, data_point, t_step):
        self.data.append(data_point)
        self.t.append(t_step)
