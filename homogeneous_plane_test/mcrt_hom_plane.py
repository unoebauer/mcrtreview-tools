from __future__ import print_function
import numpy as np
import astropy.units as u
import astropy.constants as c


class mc_packet(object):

    def __init__(self, i, grid, L):

        self.grid = grid

        self.cell_index = i
        self.cell_xl = self.grid.xl[i]
        self.cell_xr = self.grid.xr[i]
        self.cell_dx = self.grid.dx[i]
        self.cell_chi = self.grid.chi[i]

        self.L = L

        self.x = self.grid.xl[i] + self.grid.dx[i] * np.random.rand(1)[0]
        self.mu = 2. * np.random.rand(1)[0] - 1.
        self.tau = -np.log(np.random.rand(1)[0])

        self.l_edge = self.calculate_distance_edge()
        self.l_int = self.calculate_distance_interaction()

        self.is_absorbed = False
        self.is_escaped = False

        self.prop_cycle_limit = 1000000

    def calculate_distance_interaction(self):

        return self.tau / self.cell_chi

    def calculate_distance_edge(self):

        if self.mu > 0:
            dx = self.cell_xr - self.x
            self.next_cell_index = self.cell_index + 1
        else:
            dx = self.cell_xl - self.x
            self.next_cell_index = self.cell_index - 1

        return dx / self.mu

    def update_position(self, l):

        return self.x + l * self.mu

    def update_estimators(self, l):

        self.grid.Jestimator[self.cell_index] = \
            self.grid.Jestimator[self.cell_index] + \
            l * self.L / (4. * np.pi * self.cell_dx)
            # l * self.L / (4. * np.pi * self.cell_dx * self.grid.dt)
        self.grid.Hestimator[self.cell_index] = \
            self.grid.Hestimator[self.cell_index] + \
            l * self.mu * self.L / (4. * np.pi * self.cell_dx)
        self.grid.Kestimator[self.cell_index] = \
            self.grid.Kestimator[self.cell_index] + \
            l * self.mu**2 * self.L / (4. * np.pi * self.cell_dx)

    def interact(self):

        self.update_estimators(self.l_int)
        self.is_absorbed = True

        # self.x = self.update_position(self.l_int)
        # self.mu = 2. * np.random.rand(1)[0] - 1.

        # self.tau = -np.log(np.random.rand(1)[0])
        # self.l_edge = self.calculate_distance_edge()
        # self.l_int = self.calculate_distance_interaction()

    def change_cell(self):

        self.update_estimators(self.l_edge)

        if self.next_cell_index == self.grid.Ncells:

            self.is_escaped = True

            return False

        if self.next_cell_index == -1:

            self.x = self.cell_xl
            self.mu = -self.mu

            self.tau = -np.log(np.random.rand(1)[0])
            self.l_edge = self.calculate_distance_edge()
            self.l_int = self.calculate_distance_interaction()

            return True

        if self.next_cell_index > self.cell_index:

            self.x = self.grid.xl[self.next_cell_index]

        else:

            self.x = self.grid.xr[self.next_cell_index]

        self.cell_index = self.next_cell_index
        self.cell_chi = self.grid.chi[self.cell_index]
        self.cell_xl = self.grid.xl[self.cell_index]
        self.cell_xr = self.grid.xr[self.cell_index]
        self.cell_dx = self.grid.dx[self.cell_index]

        self.tau = -np.log(np.random.rand(1)[0])
        self.l_edge = self.calculate_distance_edge()
        self.l_int = self.calculate_distance_interaction()

    def propagate(self):

        i = 0
        while 1:
            if i > self.prop_cycle_limit:
                print ("Cycle Limit reached")
                return False
            if self.is_escaped or self.is_absorbed:
                return True

            if self.l_int < self.l_edge:

                self.interact()

            else:

                self.change_cell()

            i = i+1


class mcrt_grid(object):
    def __init__(self, chi=2.5e-4, S=10., xint=1e6, xmax=5e6, Ncells=100,
                 Npackets=100):

        self.Ncells = Ncells
        self.Npackets = Npackets
        # self.dt = 1e-10

        dx = xmax / float(self.Ncells)
        self.xl = np.arange(self.Ncells) * dx
        self.xr = self.xl + dx
        self.dx = np.ones(self.Ncells) * dx

        self.chi = np.where(self.xr <= xint, chi, 1e-20)
        self.eta = np.where(self.xr <= xint, S * chi, 1e-20)

        self.Ltot = np.sum(self.eta * self.dx)
        # self.Ltot = np.sum(self.eta * self.dx * self.dt)
        self.L = self.Ltot / float(self.Npackets)

        self.npackets_cell = (self.eta * self.dx / self.L).astype(np.int)
        # self.npackets_cell = (self.eta * self.dx * self.dt / self.L).astype(np.int)
        self.npackets_cell_cum = np.cumsum(self.npackets_cell)
        self.npackets_cell_cum_frac = np.cumsum(self.npackets_cell).astype(np.float) / np.sum(self.npackets_cell)
        self.packets = []

        # j = 0
        # for i in range(self.Npackets):
        #     if not i < self.npackets_cell_cum[j]:
        #         j = j+1

        #     self.packets.append(mc_packet(j, self, self.L))

        self.Jestimator = np.zeros(self.Ncells)
        self.Hestimator = np.zeros(self.Ncells)
        self.Kestimator = np.zeros(self.Ncells)
        print(self.npackets_cell_cum_frac)

    def propagate(self):

        N = self.Npackets
        n = 0

        while 1:
            if N == 0:
                print("Success")
                break
            if n > 10000000:
                print("Failure")
                break
            z = np.random.rand(1)[0]
            i = np.argwhere((self.npackets_cell_cum_frac - z) > 0)[0, 0]
            packet = mc_packet(i, self, self.L)
            print("Initializing packet in cell {:d}".format(i))
            packet.propagate()
            if packet.is_escaped:
                print("packet escaped")
                N = N - 1
            elif packet.is_absorbed:
                print("packet packet was absorbed")
            else:
                raise ValueError("Something went wrong")
            n = n + 1
            print("{:d} packets processed".format(n))
            print("{:d} packets escaped".format(self.Npackets - N))


        self.Jestimator = self.Jestimator * self.Npackets / float(n)
        self.Hestimator = self.Hestimator * self.Npackets / float(n)
        self.Kestimator = self.Kestimator * self.Npackets / float(n)

        # for i, packet in enumerate(self.packets):

        #     print(i)
        #     packet.propagate()





# class simulator(object):
#     def __init__(self, chi=1, eta=1, L1=10, L2=20, N=10000, Ncells=100):
#
#         self.N = N
#
#         self.dx = L2 / float(Ncells)
#         self.xl = np.arange(Ncells) * self.dx
#         self.xr = self.xl + self.dx
#
#         self.chi = np.where(self.xr <= L2, chi, 1e-20)
#         self.eta = np.where(self.xr <= L2, eta, 1e-20)
#
#         self.Ltot = np.sum(self.eta * self.dx)
#         self.L = self.Ltot / float(N)
#
#         self.npackets_cell = (self.Ltot / self.L).astype(np.int)
#         self.npackets_cell_cum = np.cumsum(self.npackets_cell)
#         self.packets_cell_index = np.zeros(self.N)
#
#         for i in range(self.N):
#             self.packets_cell_index[i] = \
#                 np.argwhere(i < self.npackets_cell_cum)[0, 0]
#
#         self.packets_xl = self.packets_cell_index * self.dx
#         self.packets_xr = self.packets_xl + self.dx
#
#         self.packets_x = np.random.rand(self.N) * self.dx + self.packets_xl
#         self.packets_L = self.L * np.ones(self.N)
#         self.packets_mu = np.random.rand(self.N) * 2. - 1.
#         self.packets_tau = -np.log(np.random.rand(self.N))
#
#         self.packets_is_active = np.ones(self.N).astype(np.bool)
