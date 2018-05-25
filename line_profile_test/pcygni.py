#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from astropy import units, constants
import matplotlib.pyplot as plt
try:
    import pcygni_profile as pcyg
    analytic_prediction_available = True
except ImportError:
    analytic_prediction_available = False
    pass


c1 = plt.rcParams["axes.color_cycle"][0]
c2 = plt.rcParams["axes.color_cycle"][1]
c3 = plt.rcParams["axes.color_cycle"][2]
c4 = plt.rcParams["axes.color_cycle"][3]


np.random.seed(0)


class mc_packet(object):
    def __init__(self, Rmin, Rmax, lam_min, lam_max, lam_line, tau_sobolev, t,
                 verbose=False):

        self.verbose = verbose

        self.nu_max = lam_min.to("Hz", equivalencies=units.spectral())
        self.nu_min = lam_max.to("Hz", equivalencies=units.spectral())
        self.Rmin = Rmin.to("cm")
        self.Rmax = Rmax.to("cm")

        self.nu_line = lam_line.to("Hz", equivalencies=units.spectral())
        self.tau_sob = tau_sob

        self.r = self.Rmin
        self.mu = np.sqrt(np.random.rand(1)[0])
        self.nu = nu_min + (nu_max - nu_min) * np.random.rand(1)[0]

        self.t = t
        self.emergent_nu = []

        self.draw_new_tau()
        self.check_for_boundary_intersection()
        self.calc_distance_to_sobolev_point()

    def draw_new_tau(self):

        self.tau_int = -np.log(np.random.rand(1)[0])

    def update_position_direction(self, l):

        ri = self.r.to("cm")

        self.r = np.sqrt(self.r**2 + l**2 + 2 * l * self.r * self.mu).to("cm")
        self.mu = ((l + self.mu * ri) / self.r).to("")

    def check_for_boundary_intersection(self):

        if self.mu <= -np.sqrt(1 - (self.Rmin / self.r)**2):
            # packet will intersect inner boundary if not interrupted
            sgn = -1.
            rbound = self.Rmin
            self.boundint = "left"
        else:
            # packet will intersect outer boundary if not interrupted
            sgn = 1.
            rbound = self.Rmax
            self.boundint = "right"

        self.lbound = (
            -self.mu * self.r + sgn * np.sqrt((self.mu * self.r)**2 -
                                              self.r**2 + rbound**2)).to("cm")

    def perform_interaction(self):

        mui = self.mu
        beta = self.r / self.t / constants.c

        self.mu = 2. * np.random.rand(1)[0] - 1.
        self.mu = (self.mu + beta) / (1 + beta * self.mu)

        self.nu = (self.nu * (1. - beta * mui) /
                   (1. - beta * self.mu)).to("Hz")

    def calc_distance_to_sobolev_point(self):

        self.lsob = (constants.c * self.t * (1 - self.nu_line / self.nu) -
                     self.r * self.mu).to("cm")

    def propagate(self):

        while True:
            if self.verbose:
                print(
                    "r = {:e}; mu = {:e}; lbound = {:e}; lsob = {:e}".format(
                        self.r, self.mu, self.lbound, self.lsob))
            if self.lbound < self.lsob or self.lsob < 0:
                if self.verbose:
                    print("Reaching boundary")
                if self.boundint == "left":
                    if self.verbose:
                        print("Intersecting inner boundary")
                    self.emergent_nu = None
                    break
                else:
                    if self.verbose:
                        print("Escaping through outer boundary")
                    self.emergent_nu = self.nu
                    break
            else:
                if self.verbose:
                    print("Reaching Sobolev point")
                self.update_position_direction(self.lsob)
                if self.tau_sob >= self.tau_int:
                    if self.verbose:
                        print("Line Interaction")
                    self.perform_interaction()
                else:
                    if self.verbose:
                        print("No Line Interaction")
                    self.nu_line = self.nu_max * 1.1

            self.draw_new_tau()
            self.check_for_boundary_intersection()
            self.calc_distance_to_sobolev_point()


class homologous_sphere(object):
    def __init__(self, Rmin, Rmax, nu_min, nu_max, nu_line, tau_sob, t, npack,  verbose = False):

        self.Rmin = Rmin
        self.Rmax = Rmax

        self.nu_min = nu_min
        self.nu_max = nu_max

        self.nu_line = nu_line
        self.tau_sob = tau_sob

        self.t = t

        self.verbose = verbose
        self.packets = [mc_packet(Rmin, Rmax, nu_min, nu_max, nu_line, tau_sob, t, verbose = verbose) for i in xrange(npack)]

        self.emergent_nu = []

    def perform_simulation(self):

        for pack in self.packets:
            pack.propagate()
            if pack.emergent_nu is not None:
                self.emergent_nu.append(pack.emergent_nu)

        self.emergent_nu = np.array(self.emergent_nu)


def main():

    lam_line = 1215.6 * 1e-8
    lam_min = 1185.0 * 1e-8
    lam_max = 1245.0 * 1e-8
    tau_sob = 1

    t = 13.5 * 86400.

    vmin = 1e-4 * c
    vmax = 0.01 * c

    Rmin = vmin * t
    Rmax = vmax * t

    nu_min = c / lam_max
    nu_max = c / lam_min
    nu_line = c / lam_line

    npack = 100000
    nbins = 200
    npoints = 500
    verbose = False

    sphere = homologous_sphere(Rmin, Rmax, nu_min, nu_max, nu_line, tau_sob, t, npack,  verbose = verbose)
    sphere.perform_simulation()

    solver = pcyg.homologous_sphere(rmin = Rmin, rmax = Rmax, vmax = vmax, Ip = 1, tauref = tau_sob, vref = 1e8, ve = 1e40, lam0 = lam_line)
    solution = solver.save_line_profile(nu_min, nu_max, vs_nu = True, npoints = npoints)

    fig = plt.figure(figsize = (props["pagewidth"], 1.5 * props["columnwidth"]))
    ax = fig.add_subplot(111)
    ax.plot(solution[0] * 1e-15, solution[1] / solution[1,0], label = r"prediction", color = c2)
    ax.hist(sphere.emergent_nu * 1e-15, bins = np.linspace(nu_min, nu_max, nbins) * 1e-15, histtype = "step", weights = np.ones(len(sphere.emergent_nu)) * float(nbins) / float(npack), label = "Monte Carlo", color = c3)

    ax.set_xlabel(r"$\nu$ [$10^{15} \, \mathrm{Hz}$]")
    ax.set_xlim([nu_min * 1e-15, nu_max * 1e-15])
    pax = ax.twiny()
    pax.set_xlabel(r"$\lambda$ [\AA]")
    pax.set_xlim([1.e8 * lam_min, 1e8 * lam_max])
    ax.set_ylabel(r"$F_{\nu}/F_{\nu}^{\mathrm{cont}}$")
    ax.legend()
    plt.savefig("line_profile.pdf")

    f = h5py.File("pcygni_profile.h5", "w")
    grp = f.create_group("solution")
    dset = grp.create_dataset("nu", data=solution[0])
    dset = grp.create_dataset("I", data=solution[1])
    grp = f.create_group("mc")
    grp.attrs["nu_min"] = nu_min
    grp.attrs["nu_max"] = nu_max
    grp.attrs["n_pack"] = npack
    dset = grp.create_dataset("nu", data=sphere.emergent_nu)
    f.close()

if __name__ == "__main__":

    main()
    plt.show()



