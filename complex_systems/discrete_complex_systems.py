#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


class DiscreteComplexSystem(object):

    def evolve(self):
        """Evolve the system.
        """
        raise NotImplementedError

    def plot(self):
        """Plot the evolution of the system.
        """
        raise NotImplementedError


class LinearModel(DiscreteComplexSystem):
    """Linear Model Discrete System.
    """

    def __init__(self, rd=1.5, sd=0.9, beta=10):
        """
        Parameters
        ----------
        rd : float
            The birth rate.
            Each individual has λ children every σ time units.
            rd = 1 + λ * (∆t / σ)
        sd : float
            The death rate, 0 ≤ sd ≤ 1
        beta : float
            The migration rate.
            β ≥ 0 describing the constant migration rate:
            number of individuals entering the population every ∆t time units.
        """
        self.rd = rd
        self.sd = sd
        # the net growth rate
        self.alpha = self.rd - self.sd
        self.beta = beta
        self.N = None

    def evolve(self, initial_population=20, t_max=20):
        self.N = [None] * t_max
        self.N[0] = initial_population
        for t in range(1, 20):
            self.N[t] = self.alpha * self.N[t-1] + self.beta

    def evolution_at_t(self, t):
        """Evolution at time t.
        """
        sum = .0
        for i in range(0, t-1):
            sum += (self.alpha ** i) * self.beta
        Nt = (self.alpha ** t) * self.N[0] + sum
        return Nt

    def equilibrium(self):
        """Computes equilibrium of the population.
        """
        return self.beta / (1 - self.alpha)

    def plot(self):
        plt.figure('LinearModel')
        plt.plot(self.N)
        plt.xlabel("Time")
        plt.ylabel("N")
        plt.show()


class NonLinearModel(DiscreteComplexSystem):
    """Nonlinear Model Discrete System.
    """

    def __init__(self, rd=2, k=50):
        """
        Parameters
        ----------
        rd : float
            The birth rate.
            Each individual has λ children every σ time units.
            rd = 1 + λ * (∆t / σ)
        k : int
            The carrying capacity of the environment.
        """
        self.rd = rd
        self.k = k
        self.N = None

    def evolve(self, initial_population=20, t_max=20):
        self.N = [None] * t_max
        self.N[0] = initial_population
        for t in range(1, t_max):
            self.N[t] = self.rd * self.N[t-1] * (1 - (self.N[t-1] / self.k))

    def plot(self):
        plt.figure('NonLinearModel')
        plt.plot(self.N)
        plt.xlabel("Time")
        plt.ylabel("N")
        plt.show()


class ACModel(NonLinearModel):
    """Adult-Children Model Exercise.

    Consider a population of adults and children. Assume that:
     - the population evolves by discrete steps corresponding to 1 year
     - α is the net growth rate of adults
     - every year each adult generates β children
     - children become adults after 3 years (this can be used to estimate the
       rate γ of transformation of children into adults)
     - children do not die

    Define a system of recurrence equations to model this adults/children
    population.

    Think about reasonable parameters:
     - in which cases the population exhibits exponential growth, dynamic
       equilibrium and extinction?
     - is dynamic equilibrium independent from the initial values of the
       variables?
    """

    def __init__(self, alpha=.5, beta=.6):
        """
        Parameters
        ----------
        alpha : float
            The net growth rate
        beta : float
            The migration rate.
            β ≥ 0 describing the constant migration rate:
            number of individuals entering the population every ∆t time units.
        """
        self.alpha = alpha
        self.beta = beta
        self.A = None # adults
        self.C = None # childrens

    def evolve(self, initial_a_population=50, initial_c_population=5, max_time=20):
        self.A = [0] * max_time
        self.C = [0] * max_time
        self.A[0] = initial_a_population
        self.C[0] = initial_c_population
        for t in range(1, max_time):
            self.A[t] = self.alpha * self.A[t-1] + self.C[t-1] * (1 / 3)
            self.C[t] = self.beta * self.A[t-1] - self.C[t-1] * (1 - (1 / 3))

    def plot(self):
        plt.figure('ACModel')
        plt.plot(self.A)
        plt.plot(self.C)
        plt.xlabel("Time")
        plt.ylabel("N")
        plt.legend(['Adults', 'Childrens'], loc='upper left')
        plt.show()
