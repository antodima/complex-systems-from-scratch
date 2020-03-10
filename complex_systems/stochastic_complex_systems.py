#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import roadrunner


class StochasticComplexSystem(object):

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


