#!/usr/bin/env python
"""
Abbreviations used:
    CMF: co-moving frame
    LF: lab frame
    MC: Monte Carlo
"""
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


np.random.seed(0)

tau_sobolev_default = 1
t_default = 13.5 * units.d
lam_min_default = 1185 * units.AA
lam_max_default = 1245 * units.AA
lam_line_default = 1215.6 * units.AA
vmin_default = 1e-4 * constants.c
vmax_default = 1e-2 * constants.c
Rmin_default = vmin_default * t_default
Rmax_default = vmax_default * t_default


class PropagationError(Exception):
    pass


class mc_packet(object):
    """Monte Carlo packet class

    Class describing a Monte Carlo packet propagating in a spherical homologous
    flow, bounded by the radii Rmin and Rmax. With the class methods, the
    propagation of the packet, including resonant line interactions within the
    Sobolev approximation can be performed.

    Parameters
    ----------
    Rmin : units.quantity.Quantity object
        inner radius of the spherical homologous flow;
        must have a length dimension (default vmin_default * t_default)
    Rmax : units.quantity.Quantity object
        outer radius of the spherical homologous flow;
        must have a length dimension (default vmax_default * t_default)
    lam_min : units.quantity.Quantity object
        minimum wavelength of the spectral range considered, must have
        a length dimension (default lam_min_default)
    lam_max : units.quantity.Quantity object
        maximum wavelength of the spectral range considered, must have
        a length dimension (default lam_max_default)
    lam_line : units.quantity.Quantity object
        rest wavelength of the line transition, must have a length
        dimension (default lam_line_default)
    tau_sobolev : float
        Sobolev optical depth of the line transition; assumed to constant
        throughout the domain (default tau_sobolev_default)
    t : units.quantity.Quantity object
        time since explosion, must have dimension of time (default t_default)
    verbose : boolean
        flag controlling the output to stdout (default False)
    """
    def __init__(self, Rmin=Rmin_default, Rmax=Rmax_default,
                 lam_min=lam_min_default, lam_max=lam_max_default,
                 lam_line=lam_line_default, tau_sobolev=tau_sobolev_default,
                 t=t_default, verbose=False):

        self.verbose = verbose

        self.nu_max = lam_min.to("Hz", equivalencies=units.spectral())
        self.nu_min = lam_max.to("Hz", equivalencies=units.spectral())
        self.Rmin = Rmin.to("cm")
        self.Rmax = Rmax.to("cm")

        self.nu_line = lam_line.to("Hz", equivalencies=units.spectral())
        self.tau_sob = tau_sobolev

        # consistency check
        assert(self.Rmin < self.Rmax)
        assert(self.nu_max > self.nu_min)

        # initializing the packets at the inner boundary; no limb darkening,
        # flat SED between nu_min and nu_max
        self.r = self.Rmin
        self.mu = np.sqrt(np.random.rand(1)[0])
        self.nu = (self.nu_min +
                   (self.nu_max - self.nu_min) * np.random.rand(1)[0])
        self.t = t

        # LF frequency of packet when it emerges from the surface of the
        # homologous sphere
        self.emergent_nu = None
        # distance to next interaction in optical depth space
        self.tau_int = None
        # distance to nearest boundary
        self.lbound = None
        # flag describing which boundary is intersected first on current path
        # either 'inner' or 'outer'
        self.boundint = None
        # flag describing the ultimate fate of the packet, whether it escaped
        # or was absorbed
        self.fate = None
        # flag describing whether packet has been propagated or not
        self.propagated = False

        self.draw_new_tau()
        self.check_for_boundary_intersection()
        self.calc_distance_to_sobolev_point()

    def draw_new_tau(self):
        """Draw new distance to next interaction based on Beer-Lambert law"""

        self.tau_int = -np.log(np.random.rand(1)[0])

    def update_position_direction(self, l):
        """Update the packet state during propagation

        Calculate the new radial position and propagation direction after
        having covered the distance l along the current trajectory.

        Parameters
        ----------
        l : units.quantity.Quantity object
            distance the packet travelled along the current trajectory, must
            dimension of length
        """

        ri = self.r.to("cm")

        self.r = np.sqrt(self.r**2 + l**2 + 2 * l * self.r * self.mu).to("cm")
        self.mu = ((l + self.mu * ri) / self.r).to("")

    def check_for_boundary_intersection(self):
        """Check which boundary of the spherical domain is intersected first

        Checks whether the inner or the outer boundary of the spherical domain
        is intersected first on the current trajectory. Sets the flag
        self.boundint' accordingly and calculates the physical distance to the
        nearest boundary and stores it in self.lbound.
        """

        if self.mu <= -np.sqrt(1 - (self.Rmin / self.r)**2):
            # packet will intersect inner boundary if not interrupted
            sgn = -1.
            rbound = self.Rmin
            self.boundint = "inner"
        else:
            # packet will intersect outer boundary if not interrupted
            sgn = 1.
            rbound = self.Rmax
            self.boundint = "outer"

        self.lbound = (
            -self.mu * self.r + sgn * np.sqrt((self.mu * self.r)**2 -
                                              self.r**2 + rbound**2)).to("cm")

    def perform_interaction(self):
        """Performs line interaction

        Updates the LF frequency of the packet according to the first order
        Doppler shift formula and assuming resonant scattering. A new
        propagation LF direction is drawn assuming isotropy in the CMF.
        """

        mui = self.mu
        beta = self.r / self.t / constants.c

        self.mu = 2. * np.random.rand(1)[0] - 1.
        self.mu = (self.mu + beta) / (1 + beta * self.mu)

        self.nu = (self.nu * (1. - beta * mui) /
                   (1. - beta * self.mu)).to("Hz")

    def calc_distance_to_sobolev_point(self):
        """Calculated physical distance to Sobolev point"""

        self.lsob = (constants.c * self.t * (1 - self.nu_line / self.nu) -
                     self.r * self.mu).to("cm")

    def propagate(self):
        """Perform packet propagation

        The packet is propagated through the spherical domain until it either
        escapes through the outer boundary and contributes to the spectrum or
        until it intersects the inner boundary and is discarded.

        """

        if self.propagated:
            raise PropagationError(
                "Packet has already been propagated!"
            )

        # counter
        n = 0

        while True:
            if n > 1:
                raise PropagationError(
                    "Propagation takes more steps than expected!")
            if self.verbose:
                print(
                    "step = {:d} r = {:e}; mu = {:e}; ".format(
                        n, self.r, self.mu, ) +
                    "lbound = {:e}; lsob = {:e}".format(
                        self.lbound, self.lsob))
            if self.lbound < self.lsob or self.lsob < 0:
                if self.verbose:
                    print("Reaching boundary")
                if self.boundint == "inner":
                    if self.verbose:
                        print("Intersecting inner boundary")
                    self.emergent_nu = None
                    self.fate = "absorbed"
                    break
                else:
                    if self.verbose:
                        print("Escaping through outer boundary")
                    self.emergent_nu = self.nu
                    self.fate = "escaped"
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

            n += 1

        self.propagated = True


class homologous_sphere(object):
    def __init__(self, Rmin, Rmax, nu_min, nu_max, nu_line, tau_sobolev,
                 t, npack, verbose=False):

        self.Rmin = Rmin
        self.Rmax = Rmax

        self.nu_min = nu_min
        self.nu_max = nu_max

        self.nu_line = nu_line
        self.tau_sobolev = tau_sobolev

        self.t = t

        self.verbose = verbose
        self.packets = [mc_packet(Rmin=Rmin, Rmax=Rmax, nu_min=nu_min,
                                  nu_max=nu_max, nu_line=nu_line,
                                  tau_sobolev=tau_sobolev, t=t,
                                  verbose=verbose) for i in range(npack)]

        self.emergent_nu = []

    def perform_simulation(self):

        for pack in self.packets:
            pack.propagate()
            if pack.fate == "escaped":
                self.emergent_nu.append(pack.emergent_nu)

        self.emergent_nu = np.array(self.emergent_nu)


def example():

    lam_line = 1215.6 * units.AA
    lam_min = 1185.0 * units.AA
    lam_max = 1245.0 * units.AA
    tau_sob = 1

    t = 13.5 * units.d

    vmin = 1e-4 * constants.c
    vmax = 0.01 * constants.c

    Rmin = vmin * t
    Rmax = vmax * t

    nu_min = constants.c / lam_max
    nu_max = constants.c / lam_min
    nu_line = constants.c / lam_line

    npack = 100000
    nbins = 200
    npoints = 500
    verbose = False

    sphere = homologous_sphere(
        Rmin, Rmax, nu_min, nu_max, nu_line, tau_sob, t, npack,
        verbose=verbose)
    sphere.perform_simulation()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if analytic_prediction_available:
        # WARNING: untested
        solver = pcyg.homologous_sphere(rmin=Rmin, rmax=Rmax, vmax=vmax, Ip=1,
                                        tauref=tau_sob, vref=1e8, ve=1e40,
                                        lam0=lam_line)
        solution = solver.save_line_profile(nu_min, nu_max, vs_nu=True,
                                            npoints=npoints)
        ax.plot(solution[0] * 1e-15, solution[1] / solution[1, 0],
                label=r"prediction")

    ax.hist(sphere.emergent_nu * 1e-15,
            bins=np.linspace(nu_min, nu_max, nbins) * 1e-15,
            histtype="step",
            weights=np.ones(len(sphere.emergent_nu)) * float(nbins) / npack,
            label="Monte Carlo")

    ax.set_xlabel(r"$\nu$ [$10^{15} \, \mathrm{Hz}$]")
    ax.set_xlim([nu_min * 1e-15, nu_max * 1e-15])
    pax = ax.twiny()
    pax.set_xlabel(r"$\lambda$ [\AA]")
    pax.set_xlim([1.e8 * lam_min, 1e8 * lam_max])
    ax.set_ylabel(r"$F_{\nu}/F_{\nu}^{\mathrm{cont}}$")
    ax.legend()
    plt.savefig("line_profile.pdf")


def main():

    example()


if __name__ == "__main__":

    main()
    plt.show()
