#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import roadrunner
import tellurium as te


class StochasticComplexSystem(object):

    def evolve(self):
        """Evolve the system.
        """
        raise NotImplementedError

    def plot(self):
        """Plot the evolution of the system.
        """
        raise NotImplementedError


class EnzymaticActivityModel(StochasticComplexSystem):

    def __init__(self):
        model = """
            E + S -> ES; k1*E*S;
            ES -> E + S; k2*ES;
            ES -> E + P; k3*ES;
            
            k1 = 0.3;
            k2 = 10;
            k3 = 0.01;
            
            ES = 0.0;
            P  = 0.0;
            E  = 100;
            S  = 100;
        """
        self.rr = te.loada(model)
        self.rr.setIntegrator('gillespie')

    def evolve(self, start=0, end=1000, outputs=100):
        self.rr.simulate(start, end, outputs)

    def plot(self):
        self.rr.plot()
