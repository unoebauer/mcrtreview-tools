#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2018 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Python module containing tools to perform simple MCRT simulations to determine
the escape probability from a homogeneous sphere. This test is presented in the
MCRT review.
"""
from __future__ import print_function
import numpy as np
import numpy.random as random


def p_esc_analytic(t):
    """Calculate the escape probability analytically

    Note: it is assumed that there is no scattering within the sphere, but
    that photons/packets can only be absorbed.

    Parameter
    ---------
    t : float, np.ndarray
        total optical depth of the sphere

    Returns
    -------
    p : float, np.ndarray
        escape probability
    """
    return (3. / (4. * t) * (1. - 1. / (2. * t**2) +
                             (1. / t + 1. / (2. * t**2)) * np.exp(-2. * t)))


class homogeneous_sphere_esc_abs(object):
    """Homogeneous Sphere class

    Attributes
    ----------
    p_esc : float
        escape probability as determined by MCRT

    """
    def __init__(self, tau, albedo=0.1, N=10000):

        self.RNG = random.RandomState(seed=None)
        self.N = N
        self.tau_sphere = tau
        self.albedo = albedo

        self.tau_i = self.tau_sphere * (self.RNG.rand(self.N))**(1./3.)
        self.mu_i = 2 * self.RNG.rand(self.N) - 1.

        self.N_esc = 0
        self.N_active = self.N
        # TODO: add check to avoid multiple propagation calls
        # TODO: hide routines from user
        self.propagate()

    @property
    def p_esc(self):
        return self.N_esc / float(self.N)

    def propagate(self):

        i = 0
        while self.N_active > 0:
            self.propagate_step()
            i = i + 1
            if i > 1e6:
                print("Safety exit")
                break
        print("{:d} Iterations".format(i))

    def propagate_step(self):

        self.tau = -np.log(self.RNG.rand(self.N_active))
        self.tau_edge = np.sqrt(self.tau_sphere**2 - self.tau_i**2 *
                                (1. - self.mu_i**2)) - self.tau_i * self.mu_i

        self.esc_mask = self.tau_edge < self.tau
        self.N_esc += self.esc_mask.sum()
        self.nesc_mask = np.logical_not(self.esc_mask)

        self.abs_mask = self.RNG.rand(self.nesc_mask.sum()) >= self.albedo
        self.scat_mask = np.logical_not(self.abs_mask)

        self.tau = self.tau[self.nesc_mask][self.scat_mask]
        self.tau_i = self.tau_i[self.nesc_mask][self.scat_mask]
        self.mu_i = self.mu_i[self.nesc_mask][self.scat_mask]

        self.N_active = self.scat_mask.sum()
        self.tau_i = np.sqrt(self.tau_i**2 + self.tau**2 +
                             2. * self.tau * self.tau_i * self.mu_i)
        self.mu_i = 2 * self.RNG.rand(self.N_active) - 1.


def main():

    mcrt_esc_prop = homogeneous_sphere_esc_abs(2)
    print("tau: {:.4e}, escape probability: {:.4e}".format(
        mcrt_esc_prop.tau_sphere, mcrt_esc_prop.p_esc))

if __name__ == "__main__":

    main()
