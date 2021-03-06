from __future__ import print_function
import logging
import tqdm
import numpy as np
from scipy.integrate import quad
"""
This module contains tools to perform time-independent MCRT calculations to
determine the steady-state solution for radiative transfer in the homogeneous
sphere/plane test problems described in the MCRT review.

References:
    Abdikamalov et al. 2012 ApJ, 2012, 755, 111
"""


logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# set random seed for reproducibility
np.random.seed(42)


class GeometryException(Exception):
    """Custom Exception for errors related to the geometry of the setup"""
    pass


class PropagationException(Exception):
    """Custom Exception for errors related to the propagation of packets"""
    pass


class mc_packet_base(object):
    """Base MC Carlo Packet class

    This object contains all properties and class functions which are
    independent of the particular geometry of the problem

    Parameters
    ----------
    i : int
        index of the cell in which the packet is initialized
    grid : mcrt_grid_base object
        grid object containing the computational mesh for the MCRT simulation,
        has to be an instance of mcrt_grid_base
    L : float
        packet luminosity
    """
    def __init__(self, i, grid, L):

        self.grid = grid

        # store some properties of the current parent grid cell for easy access
        self.cell_index = i
        self.cell_xl = self.grid.xl[i]
        self.cell_xr = self.grid.xr[i]
        self.cell_dx = self.grid.dx[i]
        self.cell_dV = self.grid.dV[i]
        self.cell_chi = self.grid.chi[i]

        self.L = L

        # some propagation flags, tracking whether the packet has been absorbed
        # or escaped from the domain
        self.is_absorbed = False
        self.is_escaped = False

        # flag to avoid multiple propagation
        self.is_active = True

        # a safety counter, used in the propagation loop to avoid an infinity
        # loop
        self._prop_cycle_limit = 1000000

    def initialize_direction(self):
        """Set the initial isotropic propagation direction"""

        self.mu = 2. * np.random.rand(1)[0] - 1.

    def calculate_and_set_propagation_distances(self):
        """Calculate and set the two important propagation directions

        Both the distance to the next interaction event and to the next
        grid cell crossing are calculated and stored.
        """

        self.l_edge = self.calculate_distance_edge()
        self.l_int = self.calculate_distance_interaction()

    def calculate_distance_interaction(self):
        """Calculate the physical distance to the next interaction event.

        First the distance in terms of optical depth is determined in a random
        number experiment. This is then translated into a physical distance

        Returns
        -------
        l_int : float
            distance to next interaction
        """

        self.tau = -np.log(np.random.rand(1)[0])
        return self.tau / self.cell_chi

    def update_estimators(self, l, mu_mean):
        """Update the MC estimators.

        Estimators for the first 3 moments of the specific intensity, J, H, K
        are updated along the current trajectory segment with length l.

        Parameters
        ----------
        l : float
            Length of the current trajectory segment
        mu_mean : float
            mean propagation direction
        """
        self.grid.Jestimator[self.cell_index] = \
            self.grid.Jestimator[self.cell_index] + \
            l * self.L / (4. * np.pi * self.cell_dV)
        self.grid.Hestimator[self.cell_index] = \
            self.grid.Hestimator[self.cell_index] + \
            l * mu_mean * self.L / (4. * np.pi * self.cell_dV)
        self.grid.Kestimator[self.cell_index] = \
            self.grid.Kestimator[self.cell_index] + \
            l * mu_mean**2 * self.L / (4. * np.pi * self.cell_dV)

    def interact(self):
        """Perform interaction

        The estimators are updated, the packet is absorbed and the respective
        flag set.
        """
        x, mu = self.update_position_direction(self.l_int)
        mu_mean = self.calculate_mean_mu(self.x, x, self.l_int)
        self.update_estimators(self.l_int, mu_mean)

        self.is_absorbed = True
        self.is_active = False

    def propagate(self):
        """Propagate packet

        Follow the packet until it either leaves the domain or it gets
        absorbed. As a safety measure, the propagation loop stops after a
        predefined number of cycles has been performed (prop_cycle_limit)

        Returns
        -------
        propagation_status : bool
            flag storing whether the propagation was successful or not, i.e. in
            case prop_cycle_limit was reached.

        Raises
        ------
        PropagationException
            if the packet is propagated multiple times
        """

        if not self.is_active:
            raise PropagationException("Packet has already been propagated")

        i = 0
        while 1:
            if i > self._prop_cycle_limit:
                # check safety limit
                raise PropagationException(
                    "Safety limit in propagation Loop reached")
            if self.is_escaped or self.is_absorbed:
                # check for escape or absorption
                break
            if self.l_int < self.l_edge:
                # check which event occurs next
                self.interact()
            else:
                self.change_cell()
            i = i+1


class mc_packet_planar_geom_mixin(object):
    """Mixin class for mc_packet_base, containing all features which pertain
    to planar geometry
    """
    def initialize_position(self):
        """Initialize position of MC packet

        The packet is placed uniformly within the current grid cell.
        """
        self.x = self.cell_xl + self.cell_dx * np.random.rand(1)[0]

    def calculate_distance_edge(self):
        """Calculate distance to next cell edge

        Returns
        -------
        l_edge : float
            distance to next cell edge
        """
        if self.mu > 0:
            # right interface is intersected next
            dx = self.cell_xr - self.x
            self.next_cell_index = self.cell_index + 1
        else:
            # left interface is intersected next
            dx = self.cell_xl - self.x
            self.next_cell_index = self.cell_index - 1

        return dx / self.mu

    def calculate_mean_mu(self, xi, xf, l):
        """Calculate average mu on trajectory segment

        In planar geometry, this is trivial since the direction cosine does
        not change between interactions.

        Parameters
        ----------
        xi : float
            initial position
        xf : float
            final position
        l : float
            length of trajectory segment

        Returns
        -------
        mu_mean : float
            average direction cosine on segment
        """

        return self.mu

    def update_position_direction(self, l):
        """Update position and direction of packet

        Calculate and return the new position and propagation direction after
        having covered the distance l.

        Parameters
        ----------
        l : float
            travel distance

        Returns
        -------
        x : float
            new position
        mu : float
            new propagation direction
        """

        x = self.x + self.mu * l
        mu = self.mu

        return x, mu

    def change_cell(self):
        """Handle propagation through cell interface

        If the next event is a cell crossing, i.e. l_edge < l_int, the packet
        is placed in the target cell. If the packet hereby escapes through the
        right cell interface, the respective flag is set. If it reaches the
        left boundary of the computational domain, it is reflected.
        """
        # TODO: assess whether this may partly moved into the base class

        x, mu = self.update_position_direction(self.l_edge)
        mu_mean = self.calculate_mean_mu(self.x, x, self.l_edge)
        self.update_estimators(self.l_edge, mu_mean)

        if self.next_cell_index == self.grid.Ncells:
            # packet escapes
            self.is_escaped = True
            self.is_active = False
            self.x = self.cell_xr

        elif self.next_cell_index == -1:
            # packets gets reflected

            self.x = self.cell_xl
            self.mu = -self.mu

            self.calculate_and_set_propagation_distances()

        else:
            # packet is transported into target cell
            if self.next_cell_index > self.cell_index:
                # packet is moved one cell to the right

                self.x = self.grid.xl[self.next_cell_index]

            else:
                # packet is moved one cell to the left

                self.x = self.grid.xr[self.next_cell_index]

            # reset cell-based properties for easy access
            self.cell_index = self.next_cell_index
            self.cell_chi = self.grid.chi[self.cell_index]
            self.cell_xl = self.grid.xl[self.cell_index]
            self.cell_xr = self.grid.xr[self.cell_index]
            self.cell_dx = self.grid.dx[self.cell_index]

            # recalculate distances
            self.calculate_and_set_propagation_distances()


class mc_packet_spherical_geom_mixin(object):
    """Mixin class for mc_packet_base, containing all features which pertain
    to spherical geometry
    """
    def initialize_position(self):
        """Initialize position of MC packet

        The packet is placed uniformly within the current grid cell. Hereby,
        the cell-volume growth with radius is taken into account.
        """
        self.x = (self.cell_xl**3 +
                  (self.cell_xr**3 - self.cell_xl**3) *
                  np.random.rand(1)[0])**(1./3.)

    def calculate_distance_edge(self):
        """Calculate distance to next cell edge

        Returns
        -------
        l_edge : float
            distance to next cell edge
        """
        mu_star = -np.sqrt(1. - (self.cell_xl / self.x)**2)

        if self.mu <= mu_star:

            l_edge = (-self.mu * self.x -
                      np.sqrt(self.mu**2 * self.x**2 -
                              self.x**2 + self.cell_xl**2))
            self.next_cell_index = self.cell_index - 1

        else:

            l_edge = (-self.mu * self.x +
                      np.sqrt(self.mu**2 * self.x**2 -
                              self.x**2 + self.cell_xr**2))
            self.next_cell_index = self.cell_index + 1

        return l_edge

    def calculate_mean_mu(self, xi, xf, l):
        """Calculate average mu on trajectory segment

        In spherical geometry, the directional cosine continuously changes
        during propagation. Here, the mean cosine is calculated, specifically
        the integration 1/l \int_0^l \mu d\mu is solved.

        Parameters
        ----------
        xi : float
            initial position
        xf : float
            final position
        l : float
            length of trajectory segment

        Returns
        -------
        mu_mean : float
            average direction cosine on segment
        """

        return (xf - xi) / l

    def update_position_direction(self, l):
        """Update position and direction of packet

        Calculate and return the new position and propagation direction after
        having covered the distance l.

        Parameters
        ----------
        l : float
            travel distance

        Returns
        -------
        x : float
            new position
        mu : float
            new propagation direction
        """

        x = np.sqrt(self.x**2 + l**2 + 2 * l * self.x * self.mu)
        mu = (l + self.x * self.mu) / x

        return x, mu

    def change_cell(self):
        """Handle propagation through cell interface

        If the next event is a cell crossing, i.e. l_edge < l_int, the packet
        is placed in the target cell. If the packet hereby escapes through the
        outer cell interface, the respective flag is set. Since a entire sphere
        is considered, the computation domain does not have an inner boundary

        Raises
        ------
        GeometryException
            if for some reason the next_cell_index has been set to -1, which
            would correspond to a crossing of a (non-existent) inner boundary
        """

        x, mu = self.update_position_direction(self.l_edge)
        mu_mean = self.calculate_mean_mu(self.x, x, self.l_edge)
        self.update_estimators(self.l_edge, mu_mean)

        if self.next_cell_index == self.grid.Ncells:
            # packet escapes
            self.is_escaped = True
            self.is_active = False
            self.mu = mu
            self.x = self.cell_xr

        elif self.next_cell_index == -1:

            raise GeometryException("No inner boundary in homogeneous sphere")

        else:
            # packet is transported into target cell

            self.mu = mu

            if self.next_cell_index > self.cell_index:
                # packet is moved one cell to the right

                self.x = self.grid.xl[self.next_cell_index]

            else:
                # packet is moved one cell to the left

                self.x = self.grid.xr[self.next_cell_index]

            # reset cell-based properties for easy access
            self.cell_index = self.next_cell_index
            self.cell_chi = self.grid.chi[self.cell_index]
            self.cell_xl = self.grid.xl[self.cell_index]
            self.cell_xr = self.grid.xr[self.cell_index]
            self.cell_dx = self.grid.dx[self.cell_index]
            self.cell_dV = self.grid.dV[self.cell_index]

            # recalculate distances
            self.calculate_and_set_propagation_distances()


class mc_packet_planar(mc_packet_base, mc_packet_planar_geom_mixin):
    """Class for MC packets propagating in domains with plane-parallel symmetry

    Parameters
    ----------
    i : int
        index of the cell in which the packet is initialized
    grid : mcrt_grid_base object
        grid object containing the computational mesh for the MCRT simulation,
        has to be an instance of mcrt_grid_base
    L : float
        packet luminosity

    """
    def __init__(self, i, grid, L):

        super(mc_packet_planar, self).__init__(i, grid, L)

        self.initialize_position()
        self.initialize_direction()
        self.calculate_and_set_propagation_distances()


class mc_packet_spherical(mc_packet_base, mc_packet_spherical_geom_mixin):
    """Class for MC packets propagating in domains with spherical symmetry

    Parameters
    ----------
    i : int
        index of the cell in which the packet is initialized
    grid : mcrt_grid_base object
        grid object containing the computational mesh for the MCRT simulation,
        has to be an instance of mcrt_grid_base
    L : float
        packet luminosity

    """
    def __init__(self, i, grid, L):

        super(mc_packet_spherical, self).__init__(i, grid, L)

        self.initialize_position()
        self.initialize_direction()
        self.calculate_and_set_propagation_distances()


class mcrt_grid_base(object):
    """Base class for the computational domain in which the MCRT simulation is
    performed

    This base object contains only geometry-independent features. All geometry
    specific properties are provided in specific mixin classes.

    A domain is set up, which contains a optically thick region and a
    transparent region. Packets will be initialized according to the local
    emissivity and propagated until absorption or escape (through the
    outer/right boundary at xmax).

    Parameters
    ----------
    chi : float
        absorption opacity, units of 1/cm (default 2.5e-4)
    S : float
        source function, units of erg/s/cm^2 (default 10)
    xint : float
        location of the interface between optically thick and transparent
        regions of the computational domain, units of cm; must be smaller than
        xmax (default 1e6)
    xmax : float
        extent of the computational domain, interpreted as the outer/right
        boundary of the domain, units of cm (default 5e6)
    Ncells : int
        number of grid cells in the domain (default 100)
    Npackets : int
        number of MC packets used in the MCRT simulation (default 1e6)
    """

    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=1000000):

        assert(xint < xmax)

        self.S = S
        self.xint = xint
        self.chi_base = chi

        self.Ncells = Ncells
        self.Npackets = Npackets

        self.packets = []
        self.esc_packets_x = []
        self.esc_packets_mu = []
        self.esc_packets_L = []

        # estimators for J, H, K
        self.Jestimator = np.zeros(self.Ncells)
        self.Hestimator = np.zeros(self.Ncells)
        self.Kestimator = np.zeros(self.Ncells)

        # grid cells
        dx = xmax / float(self.Ncells)
        self.xl = np.arange(self.Ncells) * dx
        self.xr = self.xl + dx
        self.dx = np.ones(self.Ncells) * dx

        # opacity and emissivity
        self.chi = np.where(self.xr <= xint, chi, 1e-20)
        self.eta = np.where(self.xr <= xint, S * chi, 1e-20)

        self._Janalytic = None
        self._Hanalytic = None
        self._Kanalytic = None

    @property
    def Janalytic(self):
        """Analytic prediction for the zeroth-moment of the specific
        intensity"""
        if self._Janalytic is None:
            self.determine_analytic_solution()
        return self._Janalytic

    @property
    def Hanalytic(self):
        """Analytic prediction for the first-moment of the specific
        intensity"""
        if self._Hanalytic is None:
            self.determine_analytic_solution()
        return self._Hanalytic

    @property
    def Kanalytic(self):
        """Analytic prediction for the second-moment of the specific
        intensity"""
        if self._Kanalytic is None:
            self.determine_analytic_solution()
        return self._Kanalytic

    def determine_number_of_packets(self):
        """Determine number of packets which are initialized in each cell

        First the local luminosity, i.e. energy injection rate is calculated
        and then uniformly distributed over all packets Npackets. According to
        this packet luminosity, the number of packets initialized in each cell
        is determined.
        """
        self.Ltot = 4. * np.pi * np.sum(self.eta * self.dV)
        self.L = self.Ltot / float(self.Npackets)

        self.npackets_cell = (4. * np.pi * self.eta * self.dV /
                              self.L).astype(np.int)
        self.npackets_cell_cum_frac = (
            np.cumsum(self.npackets_cell).astype(np.float) /
            np.sum(self.npackets_cell))

    def propagate(self):
        """Propagate all packets until escape or absorption

        The properties of escaping packets are stored.
        """

        N = self.Npackets

        for j in tqdm.tqdm(range(N)):
            z = np.random.rand(1)[0]
            i = np.argwhere((self.npackets_cell_cum_frac - z) > 0)[0, 0]
            packet = self.init_packet(i)
            packet.propagate()
            if packet.is_escaped:
                self.esc_packets_x.append(packet.x)
                self.esc_packets_mu.append(packet.mu)
                self.esc_packets_L.append(packet.L)


class mcrt_grid_planar_geom_mixin(object):
    """Mixin class containing all geometry-dependent features for the
    mcrt_grid_base class to set up a plane-parallel domain.
    """
    def determine_cell_volume(self):
        """Determine cell volume"""

        self.dV = self.dx.copy()

    def init_packet(self, i):
        """Initialize a MC packet in planar geometry"""

        return mc_packet_planar(i, self, self.L)

    def determine_analytic_solution(self):
        """Calculate analytic solution for J, H, K in the case of a
        homogeneous plane"""

        self._Janalytic = np.where(self.xr <= self.xint, self.S, 0.5 * self.S)
        self._Hanalytic = np.where(self.xr <= self.xint, 0, 0.25 * self.S)
        self._Kanalytic = np.where(self.xr <= self.xint, 1./3. * self.S,
                                   1./6. * self.S)


class mcrt_grid_spherical_geom_mixin(object):
    """Mixin class containing all geometry-dependent features for the
    mcrt_grid_base class to set up a spherically symmetric domain.
    """
    def determine_cell_volume(self):
        """Determine cell volume"""

        self.dV = 4. * np.pi / 3. * (self.xr**3 - self.xl**3)

    def init_packet(self, i):
        """Initialize a MC packet in spherical geometry"""

        return mc_packet_spherical(i, self, self.L)

    def determine_analytic_solution(self):
        """Calculate analytic solution for J, H, K in the case of a
        homogeneous sphere"""

        solver = analytic_solution_homogeneous_sphere(S=self.S,
                                                      chi=self.chi_base,
                                                      R=self.xint)

        r = 0.5 * (self.xl + self.xr)

        Janalytic = []
        Hanalytic = []
        Kanalytic = []

        for ri in r:
            Janalytic.append(solver.J(ri))
            Hanalytic.append(solver.H(ri))
            Kanalytic.append(solver.K(ri))

        self._Janalytic = np.array(Janalytic)
        self._Hanalytic = np.array(Hanalytic)
        self._Kanalytic = np.array(Kanalytic)


class mcrt_grid_planar(mcrt_grid_base, mcrt_grid_planar_geom_mixin):
    """Class to perform a MCRT simulation for the homogeneous plane problem

    A domain is set up, which contains a optically thick region and a
    transparent region. Packets will be initialized according to the local
    emissivity and propagated until absorption or escape (through the
    right boundary at xmax).

    Parameters
    ----------
    chi : float
        absorption opacity, units of 1/cm (default 2.5e-4)
    S : float
        source function, units of erg/s/cm^2 (default 10)
    xint : float
        location of the interface between optically thick and transparent
        regions of the computational domain, units of cm; must be smaller than
        xmax (default 1e6)
    xmax : float
        extent of the computational domain, interpreted as the outer/right
        boundary of the domain, units of cm (default 5e6)
    Ncells : int
        number of grid cells in the domain (default 100)
    Npackets : int
        number of MC packets used in the MCRT simulation (default 1e6)
    """
    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=1000000):

        super(mcrt_grid_planar, self).__init__(chi=chi, S=S, xint=xint,
                                               xmax=xmax, Ncells=Ncells,
                                               Npackets=Npackets)

        self.determine_cell_volume()
        self.determine_number_of_packets()

        self.propagate()


class mcrt_grid_spherical(mcrt_grid_base, mcrt_grid_spherical_geom_mixin):
    """Class to perform a MCRT simulation for the homogeneous sphere problem

    A domain is set up, which contains a optically thick region and a
    transparent region. Packets will be initialized according to the local
    emissivity and propagated until absorption or escape (through the
    right boundary at xmax).

    Parameters
    ----------
    chi : float
        absorption opacity, units of 1/cm (default 2.5e-4)
    S : float
        source function, units of erg/s/cm^2 (default 10)
    xint : float
        location of the interface between optically thick and transparent
        regions of the computational domain, units of cm; must be smaller than
        xmax (default 1e6)
    xmax : float
        extent of the computational domain, interpreted as the outer/right
        boundary of the domain, units of cm (default 5e6)
    Ncells : int
        number of grid cells in the domain (default 100)
    Npackets : int
        number of MC packets used in the MCRT simulation (default 1e6)
    """
    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=1000000):

        super(mcrt_grid_spherical, self).__init__(chi=chi, S=S, xint=xint,
                                                  xmax=xmax, Ncells=Ncells,
                                                  Npackets=Npackets)

        self.determine_cell_volume()
        self.determine_number_of_packets()

        self.propagate()


class analytic_solution_homogeneous_sphere(object):
    """Class providing functionality to calculate the analytic solution for
    the homogeneous sphere problem

    Parameters
    ----------
    S : float
        source function (default 10)
    R : float
        radius of sphere in cm (default 1e6)
    chi : float
        constant absorption opacity in 1/cm (default 2.5e-4)

    """
    def __init__(self, S=10, R=1e6, chi=2.5e-4):

        self.S = S
        self.R = R
        self.chi = chi

    def mu_star(self, r):
        """Calculate limiting directional cosine

        See Abdikamalov et al. 2012, Eq. 159

        Parameters
        ----------
        r : float
            radius

        Returns
        -------
        mu_star : float
            limiting cosine
        """

        return np.sqrt(1. - (self.R / r)**2)

    def g(self, r, mu):
        """Calculate auxiliary function g

        See Abdikamalov et al. 2012, Eq. 160

        Parameters
        ----------
        r : float
            radius
        mu : float
            directional cosine

        Returns
        -------
        g : float
        """

        return np.sqrt(1. - (r / self.R)**2 * (1. - mu**2))

    def s(self, r, mu):
        """Calculate auxiliary function s

        See Abdikamalov et al. 2012, Eq. 159

        Parameters
        ----------
        r : float
            radius
        mu : float
            directional cosine

        Returns
        -------
        s : float
        """

        assert(mu <= 1)

        if r < self.R:
            return r * mu + self.R * self.g(r, mu)
        else:
            if self.mu_star(r) <= mu:
                return 2. * self.R * self.g(r, mu)
            else:
                return 0

    def J_integ_inside(self, mu, r):
        """Integrand for solving J inside the sphere"""
        res = (np.cosh(self.chi * r * mu) *
               np.exp(-self.chi * self.R * self.g(r, mu)))
        return res

    def J_integ_outside(self, mu, r):
        """Integrand for solving J outside the sphere"""

        return np.exp(-2. * self.chi * self.R * self.g(r, mu))

    def H_integ_inside(self, mu, r):
        """Integrand for solving J inside the sphere"""

        res = (mu * np.sinh(self.chi * r * mu) *
               np.exp(-self.chi * self.R * self.g(r, mu)))

        return res

    def H_integ_outside(self, mu, r):
        """Integrand for solving H outside the sphere"""

        return mu * np.exp(-2. * self.R * self.g(r, mu))

    def K_integ_inside(self, mu, r):
        """Integrand for solving K inside the sphere"""

        res = (mu**2 * np.cosh(self.chi * r * mu) *
               np.exp(-self.chi * self.R * self.g(r, mu)))

        return res

    def K_integ_outside(self, mu, r):
        """Integrand for solving K outside the sphere"""

        return mu**2 * np.exp(-2. * self.chi * self.R * self.g(r, mu))

    def J_inside(self, r):
        """Calculate J inside sphere"""

        return self.S * (1. - quad(self.J_integ_inside, 0, 1, args=(r,))[0])

    def J_outside(self, r):
        """Calculate J outside sphere"""

        mu_star = self.mu_star(r)

        res = 0.5 * self.S * ((1. - mu_star) -
                              quad(self.J_integ_outside, mu_star, 1,
                                   args=(r,))[0])
        return res

    def H_inside(self, r):
        """Calculate H inside sphere"""

        return self.S * quad(self.H_integ_inside, 0, 1, args=(r,))[0]

    def H_outside(self, r):
        """Calculate H outside sphere"""

        mu_star = self.mu_star(r)

        res = 0.5 * self.S * (0.5 * (1. - mu_star**2) -
                              quad(self.H_integ_outside, mu_star, 1,
                                   args=(r,))[0])
        return res

    def K_inside(self, r):
        """Calculate K inside sphere"""

        return self.S * (1./3. - quad(self.K_integ_inside, 0, 1, args=(r,))[0])

    def K_outside(self, r):
        """Calculate K outside sphere"""

        mu_star = self.mu_star(r)

        res = 0.5 * self.S * (1./3. * (1. - mu_star**3) -
                              quad(self.K_integ_outside, mu_star, 1,
                                   args=(r,))[0])

        return res

    def J(self, r):
        """Calculate analytic solution for J at position r"""

        if r <= self.R:
            return self.J_inside(r)
        else:
            return self.J_outside(r)

    def H(self, r):
        """Calculate analytic solution for H at position r"""

        if r <= self.R:
            return self.H_inside(r)
        else:
            return self.H_outside(r)

    def K(self, r):
        """Calculate analytic solution for K at position r"""

        if r <= self.R:
            return self.K_inside(r)
        else:
            return self.K_outside(r)


def perform_example_simulation(mode="spherical", Npackets=10000):
    """Illustration for the use of the homogeneous sphere/plane MCRT simulation
    tools

    This routine also produces a illustration of the results. The corresponding
    figure in the estimators section of the MCRT review has been produced with
    this routine

    WARNING: this routine will perform the MCRT simulation 10 times with
    different seeds to obtain confidence intervals.

    Parameters
    ----------
    mode : {'spherical', 'planar'}
        flag determining the geometry of the MCRT simulation (default
        'spherical')
    Npackets : int
        number of packets used in each MCRT simulation (default 10000)
    """
    import matplotlib.pyplot as plt

    assert(mode in ["planar", "spherical"])

    J_est = []
    H_est = []
    K_est = []
    for i in range(10):
        logging.info("Doing Iteration {:d}".format(i))
        if mode == "planar":
            mcrt = mcrt_grid_planar(Npackets=Npackets)
        else:
            mcrt = mcrt_grid_spherical(Npackets=Npackets)
        J_est.append(mcrt.Jestimator)
        H_est.append(mcrt.Hestimator)
        K_est.append(mcrt.Kestimator)

    J_est = np.array(J_est) / mcrt.S
    H_est = np.array(H_est) / mcrt.S
    K_est = np.array(K_est) / mcrt.S

    colors = plt.rcParams["axes.color_cycle"]
    labels = [r"$J$", r"$H$", r"$K$"]

    x = (mcrt.xl + mcrt.xr) * 0.5 * 1e-5

    for y in [mcrt.Janalytic, mcrt.Hanalytic, mcrt.Kanalytic]:
        plt.plot(x, y / mcrt.S, ls="dashed", color="black")

    for i, y in enumerate([J_est, H_est, K_est]):
        c = colors[i]
        plt.fill_between(x, y.mean(axis=0) - 2. * y.std(axis=0),
                         y.mean(axis=0) + 2. * y.std(axis=0),
                         alpha=0.25, color=c)
        plt.fill_between(x, y.mean(axis=0) - y.std(axis=0),
                         y.mean(axis=0) + y.std(axis=0),
                         alpha=0.5, color=c)
        plt.plot(x, y.mean(axis=0), color=c, marker="o", ls="",
                 label=labels[i], markerfacecolor=(1, 1, 1, 0),
                 markeredgecolor=c)

    plt.legend(frameon=False)
    plt.xlabel(r"$r$ [km]")
    plt.ylabel(r"$J/S$, $H/S$, $K/S$")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()


if __name__ == "__main__":

    perform_example_simulation(mode="spherical")
