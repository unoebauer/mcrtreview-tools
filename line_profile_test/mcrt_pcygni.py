#!/usr/bin/env python
"""
Python module containing tools to perform simple MCRT simulations for the
line profile test in a homologously expanding spherical flow. This test is
presented in the MCRT review.

There is the possibility to compare the MCRT results to analytic predictions
obtained from a formal integration of the radiative transfer problem, following
the procedure outlined by Jeffery & Branch 1990. For this, a external module
has to be imported which can be obtained from the github repository

https://github.com/unoebauer/public-astro-tools.git


References
----------
Jeffery, D. J. & Branch,  Analysis of Supernova Spectra in
    Analysis of Supernova Spectra Supernovae,
    Jerusalem Winter School for Theoretical Physics, 1990, 149

Abbreviations used:
    CMF: co-moving frame
    LF: lab frame
    MC: Monte Carlo
"""
from __future__ import print_function
import os
import numpy as np
from astropy import units, constants
import matplotlib
if "DISPLAY" not in os.environ:
    # backend that works without an X-server
    matplotlib.use("agg")
import matplotlib.pyplot as plt
try:
    # Available from https://github.com/unoebauer/public-astro-tools.git
    import pcygni_profile as pcyg
    analytic_prediction_available = True
except ImportError:
    analytic_prediction_available = False
    pass


# Set RNG seed for reproducibility
np.random.seed(42)

# Parameters used for test calculation shown in the review
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
    Rmin : float
        inner radius of the spherical homologous flow;
        must be in cm (default 3.5e12)
    Rmax : float
        outer radius of the spherical homologous flow;
        must be in cm (default 3.5e14)
    nu_min : units.quantity.Quantity object
        minimum fequency of the spectral range considered, must be in Hz
        (default 2.4e15)
    nu_max : units.quantity.Quantity object
        maximum frequency of the spectral range considered, must be in Hz
        (default 2.5e15)
    lam_line : units.quantity.Quantity object
        rest frequency of the line transition, must be in Hz
        (default 2.47e15)
    tau_sobolev : float
        Sobolev optical depth of the line transition; assumed to constant
        throughout the domain, must be dimensionless
        (default 1)
    t : units.quantity.Quantity object
        time since explosion, must be in s (default 1.1e6)
    verbose : boolean
        flag controlling the output to stdout (default False)
    """
    def __init__(self, Rmin=3.5e12, Rmax=3.5e14,
                 nu_min=2.4e15, nu_max=2.5e15,
                 nu_line=2.47e15, tau_sobolev=1,
                 t=1.1e6, verbose=False):

        self.verbose = verbose

        self.nu_min = nu_min
        self.nu_max = nu_max
        self.Rmin = Rmin
        self.Rmax = Rmax

        self.nu_line = nu_line
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

        ri = self.r

        self.r = np.sqrt(self.r**2 + l**2 + 2 * l * self.r * self.mu)
        self.mu = ((l + self.mu * ri) / self.r)

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
                                              self.r**2 + rbound**2))

    def perform_interaction(self):
        """Performs line interaction

        Updates the LF frequency of the packet according to the first order
        Doppler shift formula and assuming resonant scattering. A new
        propagation LF direction is drawn assuming isotropy in the CMF.
        """

        beta = self.r / self.t / constants.c.cgs.value

        self.mu = 2. * np.random.rand(1)[0] - 1.
        self.mu = (self.mu + beta) / (1 + beta * self.mu)

        self.nu = self.nu_line / (1. - beta * self.mu)

    def calc_distance_to_sobolev_point(self):
        """Calculated physical distance to Sobolev point"""

        self.lsob = (constants.c.cgs.value * self.t *
                     (1 - self.nu_line / self.nu) -
                     self.r * self.mu)

    def print_info(self, message):
        if self.verbose:
            print(message)

    def propagate(self):
        """Perform packet propagation

        The packet is propagated through the spherical domain until it either
        escapes through the outer boundary and contributes to the spectrum or
        until it intersects the inner boundary and is discarded.  The
        implementation of the propagation routine is specific to the problem at
        hand and makes use of the fact that a packet can at most interact once.
        """

        if self.propagated:
            raise PropagationError(
                "Packet has already been propagated!"
            )

        if self.lbound < self.lsob or self.lsob < 0:
            self.print_info("Reaching outer boundary")
            self.fate = "escaped"
        else:
            self.print_info("Reaching Sobolev point")
            self.update_position_direction(self.lsob)
            if self.tau_sob >= self.tau_int:
                self.print_info("Line Interaction")
                self.perform_interaction()
                self.check_for_boundary_intersection()
                if self.boundint == "inner":
                    self.print_info("Intersecting inner boundary")
                    self.fate = "absorbed"
                else:
                    self.print_info("Reaching outer boundary")
                    self.fate = "escaped"
            else:
                self.fate = "escaped"

        self.emergent_nu = self.nu
        self.propagated = True


class homologous_sphere(object):
    """
    Class describing the sphere in homologous expansion in which the MCRT
    simulation is performed

    The specified number of MC packets are initialized. Their propagation is
    followed in the main routine of this class. As a result, the emergent
    frequencies of all escaping packets are recorded in self.emergent_nu.

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
    npacks : int
        number of packets in the MCRT simulation (default 10000)
    """
    def __init__(self, Rmin=Rmin_default, Rmax=Rmax_default,
                 lam_min=lam_min_default, lam_max=lam_max_default,
                 lam_line=lam_line_default, tau_sobolev=tau_sobolev_default,
                 t=t_default, verbose=False, npacks=10000):

        t = t.to("s").value
        Rmin = Rmin.to("cm").value
        Rmax = Rmax.to("cm").value

        nu_min = lam_max.to("Hz", equivalencies=units.spectral()).value
        nu_max = lam_min.to("Hz", equivalencies=units.spectral()).value
        nu_line = lam_line.to("Hz", equivalencies=units.spectral()).value

        self.npacks = npacks
        self.packets = [mc_packet(Rmin=Rmin, Rmax=Rmax, nu_min=nu_min,
                                  nu_max=nu_max, nu_line=nu_line,
                                  tau_sobolev=tau_sobolev, t=t,
                                  verbose=verbose) for i in range(npacks)]

        self.emergent_nu = []

    def perform_simulation(self):
        """Perform MCRT simulation in the homologous flow

        All packets are propagated until they either escape from the sphere or
        intersect the photosphere and are discarded.
        """

        for i, pack in enumerate(self.packets):
            pack.propagate()
            if pack.fate == "escaped":
                self.emergent_nu.append(pack.emergent_nu)
            if (i % 100) == 0:
                print("{:d} of {:d} packets done".format(i, self.npacks))

        self.emergent_nu = np.array(self.emergent_nu) * units.Hz


def perform_line_profile_calculation(Rmin=Rmin_default, Rmax=Rmax_default,
                                     lam_min=lam_min_default,
                                     lam_max=lam_max_default,
                                     lam_line=lam_line_default,
                                     tau_sobolev=tau_sobolev_default,
                                     t=t_default, verbose=False, npacks=10000,
                                     nbins=100, npoints=500, save_to_pdf=True,
                                     include_analytic_solution=True):
    """
    Class describing the sphere in homologous expansion in which the MCRT
    simulation is performed

    The specified number of MC packets are initialized. Their propagation is
    followed in the main routine of this class. As a result, the emergent
    frequencies of all escaping packets are recorded in self.emergent_nu.

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
    npacks : int
        number of packets in the MCRT simulation (default 10000)
    nbins : int
        number of bins used for the histogram when plotting the emergent
        spectrum (default 100)
    npoints : int
        number of points used in the formal integration when calculating the
        analytic solution, provided that the module is available and that
        include_analytic_solution is set to True (default 500)
    save_to_pdf : bool
        flag controlling whether the comparison plot is saved to pdf (default
        True)
    include_analytic_solution : bool
        flag controlling whether the analytic solution is included in the plot;
        this requires that the appropriate module is available (default True)
    """

    vmin = (Rmin / t).to("cm/s")
    vmax = (Rmax / t).to("cm/s")

    nu_min = lam_max.to("Hz", equivalencies=units.spectral())
    nu_max = lam_min.to("Hz", equivalencies=units.spectral())

    npoints = 500

    sphere = homologous_sphere(
        Rmin=Rmin, Rmax=Rmax, lam_min=lam_min, lam_max=lam_max,
        lam_line=lam_line, tau_sobolev=tau_sobolev, t=t, npacks=npacks,
        verbose=verbose)
    sphere.perform_simulation()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if include_analytic_solution:
        if analytic_prediction_available:
            # WARNING: untested
            ve = 1e40 * units.cm / units.s
            vref = 1e8 * units.cm / units.s
            solver = pcyg.PcygniCalculator(t=t, vmax=vmax, vphot=vmin,
                                           tauref=tau_sobolev, vref=vref,
                                           ve=ve, lam0=lam_line)
            nu_tmp, Fnu_normed_tmp = solver.calc_profile_Fnu(npoints=npoints)
            Fnu_normed = np.append(np.insert(Fnu_normed_tmp, 0, 1), 1)

            # numpy append has difficulties with astropy quantities
            nu = np.zeros(len(nu_tmp) + 2) * nu_tmp.unit
            nu[1:-1] = nu_tmp[::]
            nu[0] = nu_min
            nu[-1] = nu_max

            ax.plot(nu.to("1e15 Hz"), Fnu_normed,
                    label=r"formal integration")
        else:
            print("Warning: module for analytic solution not available")

    ax.hist(sphere.emergent_nu.to("1e15 Hz"),
            bins=np.linspace(nu_min, nu_max, nbins).to("1e15 Hz"),
            histtype="step",
            weights=np.ones(len(sphere.emergent_nu)) * float(nbins) / npacks,
            label="Monte Carlo")

    ax.set_xlabel(r"$\nu$ [$10^{15} \, \mathrm{Hz}$]")
    ax.set_xlim([nu_min.to("1e15 Hz").value, nu_max.to("1e15 Hz").value])
    pax = ax.twiny()
    pax.set_xlabel(r"$\lambda$ $[\mathrm{\AA}]$")
    pax.set_xlim([lam_min.to("AA").value, lam_max.to("AA").value])
    ax.set_ylabel(r"$F_{\nu}/F_{\nu}^{\mathrm{cont}}$")
    ax.legend()
    if save_to_pdf:
        fig.savefig("line_profile.pdf")


def example():
    """Perform the MCRT test simulation from the review"""

    perform_line_profile_calculation(
        Rmin=Rmin_default, Rmax=Rmax_default, lam_min=lam_min_default,
        lam_max=lam_max_default, lam_line=lam_line_default,
        tau_sobolev=tau_sobolev_default, t=t_default, verbose=False,
        npacks=100000, nbins=100, npoints=500, save_to_pdf=True)


def main():
    """Main routine; performs the example calculation"""

    example()


if __name__ == "__main__":

    main()
    plt.show()
