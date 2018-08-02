from __future__ import print_function
import sys
import numpy as np


np.random.seed(0)


def progress(count, total, suffix=''):
    # See
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


class mc_packet_base(object):

    def __init__(self, i, grid, L):

        self.grid = grid

        self.cell_index = i
        self.cell_xl = self.grid.xl[i]
        self.cell_xr = self.grid.xr[i]
        self.cell_dx = self.grid.dx[i]
        self.cell_dV = self.grid.dV[i]
        self.cell_chi = self.grid.chi[i]

        self.L = L

        self.x = self.grid.xl[i] + self.grid.dx[i] * np.random.rand(1)[0]
        self.mu = 2. * np.random.rand(1)[0] - 1.

        self.l_edge = self.calculate_distance_edge()
        self.l_int = self.calculate_distance_interaction()

        self.is_absorbed = False
        self.is_escaped = False

        self.prop_cycle_limit = 1000000

    def calculate_distance_interaction(self):

        self.tau = -np.log(np.random.rand(1)[0])
        return self.tau / self.cell_chi

    def update_estimators(self, l):

        self.grid.Jestimator[self.cell_index] = \
            self.grid.Jestimator[self.cell_index] + \
            l * self.L / (4. * np.pi * self.cell_dV)
        self.grid.Hestimator[self.cell_index] = \
            self.grid.Hestimator[self.cell_index] + \
            l * self.mu * self.L / (4. * np.pi * self.cell_dV)
        self.grid.Kestimator[self.cell_index] = \
            self.grid.Kestimator[self.cell_index] + \
            l * self.mu**2 * self.L / (4. * np.pi * self.cell_dV)

    def interact(self):

        self.update_estimators(self.l_int)
        self.is_absorbed = True

    def propagate(self):

        i = 0
        while 1:
            if i > self.prop_cycle_limit:
                print("Cycle Limit reached")
                return False
            if self.is_escaped or self.is_absorbed:
                return True

            if self.l_int < self.l_edge:

                self.interact()
            else:

                self.change_cell()

            i = i+1


class mc_packet_spherical_geom_mixin(object):

    def initialize_position(self):

        pass

    def calculate_distance_edge(self):

        pass

    def change_cell(self):

        pass


class mc_packet_planar_geom_mixin(object):

    def initialize_position(self):

        self.x = self.cell_xl + self.cell_dx * np.random.rand(1)[0]

    def calculate_distance_edge(self):

        if self.mu > 0:
            dx = self.cell_xr - self.x
            self.next_cell_index = self.cell_index + 1
        else:
            dx = self.cell_xl - self.x
            self.next_cell_index = self.cell_index - 1

        return dx / self.mu

    def change_cell(self):

        self.update_estimators(self.l_edge)

        if self.next_cell_index == self.grid.Ncells:

            self.is_escaped = True
            self.x = self.cell_xr

        elif self.next_cell_index == -1:

            self.x = self.cell_xl
            self.mu = -self.mu

            self.l_edge = self.calculate_distance_edge()
            self.l_int = self.calculate_distance_interaction()

        else:

            if self.next_cell_index > self.cell_index:

                self.x = self.grid.xl[self.next_cell_index]

            else:

                self.x = self.grid.xr[self.next_cell_index]

            self.cell_index = self.next_cell_index
            self.cell_chi = self.grid.chi[self.cell_index]
            self.cell_xl = self.grid.xl[self.cell_index]
            self.cell_xr = self.grid.xr[self.cell_index]
            self.cell_dx = self.grid.dx[self.cell_index]

            self.l_edge = self.calculate_distance_edge()
            self.l_int = self.calculate_distance_interaction()


class mc_packet_planar(mc_packet_base, mc_packet_planar_geom_mixin):
    def __init__(self, i, grid, L):

        super(mc_packet_planar, self).__init__(i, grid, L)


class mc_packet_spherical(mc_packet_base, mc_packet_spherical_geom_mixin):
    def __init__(self, i, grid, L):

        super(mc_packet_spherical, self).__init__(i, grid, L)


class mcrt_grid_base(object):
    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=1000000):

        self.Ncells = Ncells
        self.Npackets = Npackets

        self.packets = []
        self.esc_packets_x = []
        self.esc_packets_mu = []
        self.esc_packets_L = []

        self.Jestimator = np.zeros(self.Ncells)
        self.Hestimator = np.zeros(self.Ncells)
        self.Kestimator = np.zeros(self.Ncells)

        dx = xmax / float(self.Ncells)
        self.xl = np.arange(self.Ncells) * dx
        self.xr = self.xl + dx
        self.dx = np.ones(self.Ncells) * dx

        self.chi = np.where(self.xr <= xint, chi, 1e-20)
        self.eta = np.where(self.xr <= xint, S * chi, 1e-20)

    def determine_number_of_packets(self):

        self.Ltot = 4. * np.pi * np.sum(self.eta * self.dV)
        self.L = self.Ltot / float(self.Npackets)

        self.npackets_cell = (4. * np.pi * self.eta * self.dV /
                              self.L).astype(np.int)
        self.npackets_cell_cum_frac = (
            np.cumsum(self.npackets_cell).astype(np.float) /
            np.sum(self.npackets_cell))

    def propagate(self):

        N = self.Npackets

        for j in range(N):
            z = np.random.rand(1)[0]
            i = np.argwhere((self.npackets_cell_cum_frac - z) > 0)[0, 0]
            packet = self.init_packet(i)
            ret = packet.propagate()
            assert(ret)
            if packet.is_escaped:
                self.esc_packets_x.append(packet.x)
                self.esc_packets_mu.append(packet.mu)
                self.esc_packets_L.append(packet.L)

            progress(j, N, suffix='')


class mcrt_grid_planar_geom_mixin(object):

    def determine_cell_volume(self):

        self.dV = self.dx.copy()

    def init_packet(self, i):

        return mc_packet_planar(i, self, self.L)

    def determine_analytic_solution(self):

        pass


class mcrt_grid_spherical_geom_mixin(object):

    def determine_cell_volume(self):

        pass

    def init_packet(self, i):

        pass

    def determine_analytic_solution(self):

        pass


class mcrt_grid_planar(mcrt_grid_base, mcrt_grid_planar_geom_mixin):

    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=1000000):

        super(mcrt_grid_planar, self).__init__(chi=chi, S=S, xint=xint,
                                               xmax=xmax, Ncells=Ncells,
                                               Npackets=Npackets)

        self.determine_cell_volume()
        self.determine_number_of_packets()

        self.propagate()
