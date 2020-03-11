#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import roadrunner


class ContinuousComplexSystem(object):

    def evolve(self):
        """Evolve the system.
        """
        raise NotImplementedError

    def deltas(self, y, t):
        """Computes the derivatives of the differential equations.

        Parameters
        ----------
        y : list
            the values of the system
        t : numpy.linspace
            the time points

        Returns
        -------
        the partial derivatives
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

    def __init__(self, t):
        self.s = 77.27
        self.q = 8.375e-6
        self.w = 0.161
        self.A = []
        self.B = []
        self.C = []
        self.t = t # the time points

    def deltas(self, y, t):
        A = y[0]
        B = y[1]
        C = y[2]
        dA = 320 * self.s * (A - A * B + B - self.q * A * A)
        dB = 320 * (C - B - A * B) / self.s
        dC = 320 * self.w * (A - C)
        return [dA, dB, dC]

    def evolve(self, y0):
        y = odeint(self.deltas, y0, self.t)
        self.A = y[:, 0]
        self.B = y[:, 1]
        self.C = y[:, 2]

    def plot(self):
        plt.figure('OregonatorModel')
        plt.plot(self.t, self.A, label='A')
        plt.plot(self.t, self.B, label='B')
        plt.plot(self.t, self.C, label='C')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.show()


class LotkaVolterraModel(ContinuousComplexSystem):
    """Lotka-Volterra model of predator-prey system.
    """

    def __init__(self, t, growth_rate=10, death_rate=10, meeting_rate=0.01, n_offsprings=1):
        self.r = growth_rate
        self.s = death_rate
        self.a = meeting_rate # predator-prey meetings
        self.b = n_offsprings # number of offsprings produced for each hunting
        self.V = [] # preys
        self.P = [] # predators
        self.t = t

    def deltas(self, y, t):
        V = y[0]
        P = y[1]
        dV = self.r * V - self.a * V * P
        dP = - self.s * P + self.a * self.b * V * P
        return [dV, dP]

    def evolve(self, y0):
        y = odeint(self.deltas, y0, self.t)
        self.V = y[:, 0]
        self.P = y[:, 1]

    def plot(self):
        plt.figure('LotkaVolterraModel')
        plt.plot(self.t, self.V, label='V')
        plt.plot(self.t, self.P, label='P')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.show()


class SIRModel(ContinuousComplexSystem):
    """Susceptible-Infected-Recovered/Resistant model of epidemic phenomena.
    S + I + R = 1
    """

    def __init__(self, t, infection_coefficient=6, recovery_rate=2, growth_rate=2, vaccination_rate=0.1):
        self.S = []
        self.I = []
        self.R = []
        self.beta = infection_coefficient
        self.gamma = recovery_rate
        self.mi = growth_rate # μ
        self.p = vaccination_rate
        self.t = t

    def deltas(self, y, t):
        S = y[0]
        I = y[1]
        R = y[2]
        dS = (1 - self.p) * self.mi - self.beta * S * I - self.mi * S
        dI = self.beta * S * I - self.gamma * I - self.mi * I
        dR = self.p * self.mi + self.gamma * I - self.mi * R
        return [dS, dI, dR]

    def evolve(self, y0):
        y = odeint(self.deltas, y0, self.t)
        self.S = y[:, 0]
        self.I = y[:, 1]
        self.R = y[:, 2]

    def vaccination_threshold(self):
        return 1 - (self.mi + self.gamma) / self.beta

    def plot(self):
        plt.figure('SIRModel')
        plt.plot(self.t, self.S, label='S')
        plt.plot(self.t, self.I, label='I')
        plt.plot(self.t, self.R, label='R')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.show()


class PPVModel(ContinuousComplexSystem):
    """Predator.Prey-Vegetation Model Exercise.

    Define a variant of the Lotka-Volterra model with 3 kinds of individual:
        predator, prey and vegetation.
    Predators eat preys, preys eat vegetation, and vegetation
    (in the absence of preys) grows exponentially.
    """

    def __init__(self, t, growth_rate=10, death_rate=10, meeting_rate=0.01, n_offsprings=1, growth_rate_veg=2):
        self.r = growth_rate
        self.s = death_rate
        self.a = meeting_rate # predator-prey meetings
        self.b = n_offsprings # number of offsprings produced for each hunting
        self.v = growth_rate_veg # vegetation growth rate
        self.V = [] # preys
        self.P = [] # predators
        self.F = [] # vegetations
        self.t = t

    def deltas(self, y, t):
        V = y[0]
        P = y[1]
        F = y[2]
        dV = self.r * V - self.a * V * P
        dP = - self.s * P + self.a * self.b * V * P
        dF = (F ** self.v) - V
        return [dV, dP, dF]

    def evolve(self, y0):
        y = odeint(self.deltas, y0, self.t)
        self.V = y[:, 0]
        self.P = y[:, 1]
        self.F = y[:, 2]

    def plot(self):
        plt.figure('PPVModel')
        plt.plot(self.t, self.V, label='V')
        plt.plot(self.t, self.P, label='P')
        plt.plot(self.t, self.F, label='F')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.show()


class BrusselatorModel(ContinuousComplexSystem):
    """The Brusselator is a theoretical model for a type of autocatalytic reaction.

    https://en.wikipedia.org/wiki/Brusselator
    """
    def __init__(self):
        import os
        path = 'complex_systems/models/brusselator.xml'
        self.rr = roadrunner.RoadRunner(os.path.abspath(path))

    def evolve(self, start=0, end=10, outputs=100):
        result = self.rr.simulate(start, end, outputs)
        return result

    def plot(self):
        self.rr.plot()
