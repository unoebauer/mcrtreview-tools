#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import numpy.random as random


def p_esc_analytic(t):
    return (3. / (4. * t) * (1. - 1. / (2. * t**2) +
                             (1. / t + 1. / (2. * t**2)) * np.exp(-2. * t)))


class homogeneous_sphere_esc_abs(object):
    def __init__(self, tau, albedo=0.1, N=10000):

        self.RNG = random.RandomState(seed=None)
        self.N = N
        self.tau_sphere = tau
        self.albedo = albedo

        self.tau_i = self.tau_sphere * (self.RNG.rand(self.N))**(1./3.)
        self.mu_i = 2 * self.RNG.rand(self.N) - 1.

        self.N_esc = 0
        self.N_active = self.N

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
