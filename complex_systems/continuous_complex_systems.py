#!/usr/bin/env python3
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class ContinuousComplexSystem(object):

    def evolve(self):
        """Evolve the system.
        """
        raise NotImplementedError

    def plot(self):
        """Plot the evolution of the system.
        """
        raise NotImplementedError


class OregonatorModel(ContinuousComplexSystem):
    """The Oregonator is the simplest realistic model of the chemical dynamics 
    of the oscillatory Belousov–Zhabotinsky reaction.
    """
    
    def __init__(self):
        self.s = 77.27
        self.q = 8.375e-6
        self.w = 0.161
        self.y = None
        
    def __func_derivatives(self, y, t):
        A = y[0]
        B = y[1]
        C = y[2]
        dA = 320 * self.s * (A - A * B + B - self.q * A * A)
        dB = 320 * (C - B - A * B) / self.s
        dC = 320 * self.w * (A - C)
        return [dA, dB, dC]
    
    def evolve(self, y0, t):
        self.y = odeint(self.__func_derivatives, y0, t)
    
    def plot(self, t):
        plt.plot(t, self.y[:, 0], label='A')
        plt.plot(t, self.y[:, 1], label='B')
        plt.plot(t, self.y[:, 2], label='C')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.show()
