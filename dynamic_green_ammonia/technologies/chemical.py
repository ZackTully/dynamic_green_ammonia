"""
[1] R. Nayak-Luke, R. Bañares-Alcántara, and I. Wilkinson, “'Green' Ammonia: Impact of 
Renewable Energy Intermittency on Plant Sizing and Levelized Cost of Ammonia,” Ind. Eng.
Chem. Res., vol. 57, no. 43, pp. 14607-14616, Oct. 2018, doi: 10.1021/acs.iecr.8b02447.

This file contains very simple electrolyzer, air separation unit, and haber-bosch 
preformance and financial models.

"""


import numpy as np
import logging
import datetime
import molmass


class Electrolzyer:
    def __init__(self, dt, rating):
        self.t = 0
        self.dt = dt
        self.rating = rating  # kW
        self.energypkg_H2 = 53.4  # kWh/kg
        self.rating_kgH2 = self.rating * self.dt / 3600 / self.energypkg_H2

    def step(self, P_in):
        P_in_kWh = P_in * self.dt / 3600
        H2 = np.min([P_in_kWh / self.energypkg_H2, self.rating_kgH2])
        reject = (P_in_kWh - H2 * self.energypkg_H2) * 3600 / self.dt
        return H2, reject

    def calc_financials(self, rating_elec, rating_h2):
        # from [1]

        self.rating_elec = rating_elec
        self.rating_h2 = rating_h2


        rating_h2 = rating_h2 * 24

        self.K_rated_power = 880
        self.n_rated_power = 1
        self.K_kgpday = 1
        self.n_kgpday = 0.65

        self.capex_rated_power = self.K_rated_power * rating_elec**self.n_rated_power
        self.opex_rated_power = 0.05 * self.capex_rated_power  # per year

        self.capex_kgpdy = self.K_kgpday * rating_h2**self.n_kgpday
        self.opex_kfpday = 0.05 * self.capex_kgpdy


class HaberBosch:
    def __init__(self, dt, rating):
        self.t = 0
        self.dt = dt
        self.rating = rating  # kW

        self.P2NH3 = 0.6  # kWh/kg

        self.molmass_H2 = molmass.Formula("H2").mass  # g/mol
        self.molmass_N2 = molmass.Formula("N2").mass  # g/mol
        self.molmass_NH3 = molmass.Formula("NH3").mass  # g/mol

        # N2 + 3 H2 <-> 2 NH3
        self.kgpkg_H2 = (3 / 2) * self.molmass_H2 / self.molmass_NH3  # kg H2 per kg NH3
        self.kgpkg_N2 = (1 / 2) * self.molmass_N2 / self.molmass_NH3  # kg N2 per kg NH3
        self.energypkg_NH3 = 0.6  # kWh/kg

        self.rating_kgNH3 = (
            self.rating * self.dt / 3600 / self.energypkg_NH3
        )  # [kg] rated production in kg of NH3

    def step(self, H2_in, N2_in, P_in):
        """
        H2_in: hydrogen in [kg/s]
        N2_in: nitrogen in [kg/s]
        P_in: power in [kW]

        NH3: ammonia out [kg/s]
        reject = [
            H2 reject: unused hydrogen [kg/s]
            N2 reject: unused nitrogen [kg/s]
            power reject: unused power [kW]
        ]
        """

        P_in_kWh = P_in * self.dt / 3600

        # calculate NH3 production as if each ingredient were limiting
        NH3_H2 = H2_in / self.kgpkg_H2
        NH3_N2 = N2_in / self.kgpkg_N2
        NH3_energy = P_in_kWh / self.energypkg_NH3

        # actual NH3 production is determined by the limiting ingredient
        NH3 = np.min([NH3_H2, NH3_N2, NH3_energy, self.rating_kgNH3])

        # calculate extra ingredients to reject
        reject = [
            H2_in - NH3 * self.kgpkg_H2,
            N2_in - NH3 * self.kgpkg_N2,
            (P_in_kWh - NH3 * self.energypkg_NH3) * 3600 / self.dt,
        ]

        return NH3, reject

    def calc_financials(self, rating_elec, rating_kg_phr):

        self.rating_elec = rating_elec
        self.rating_NH3 = rating_kg_phr

        rating_ton_pday = rating_kg_phr * 1e-3 * 24

        self.K = 3.4e6
        self.n = 0.50

        self.capex = self.K * rating_ton_pday**self.n
        self.opex = 0.05 * self.capex


class AirSeparationUnit:
    def __init__(self, dt, rating):
        self.t = 0
        self.dt = dt
        self.rating = rating  # kW
        self.energypkg_N2 = 0.119  # kWh/kg
        self.rating_kgN2 = self.rating * self.dt / 3600 / self.energypkg_N2

    def step(self, P_in):
        P_in_kWh = P_in * self.dt / 3600
        N2 = np.min([P_in_kWh / self.energypkg_N2, self.rating_kgN2])
        reject = (P_in_kWh - N2 * self.energypkg_N2) * 3600 / self.dt
        return N2, reject

    def calc_financials(self, rating_elec, rating_kg_phr):
        self.rating_elec = rating_elec
        self.rating_N2 = rating_kg_phr

        rating_ton_pday = rating_kg_phr * 1e-3 * 24

        self.K = 9.2e5
        self.n = 0.49

        self.capex = self.K * rating_ton_pday**self.n
        self.opex = 0.05 * self.capex


class Control:
    # TODO
    # 1. implement logging

    def __init__(self):
        # self.initialize_logging()

        self.dt = 1
        self.t = 0  # time

        self.initialize_components()

        n_comps = len(self.ci)

        self.I = np.zeros([n_comps, n_comps])  # information exchange
        self.P = np.zeros([n_comps, n_comps])  # power exchange
        self.C = np.zeros([n_comps, n_comps])  # chemical exchange

        logging.info("Initialized components")

    def initialize_logging(self):
        logging.basicConfig(
            filename="Dynamic_Load/literature/Nayak_Luke_2018/example.log",
            encoding="utf-8",
            level=logging.DEBUG,
        )
        logging.debug(f"Log started at {datetime.datetime.now()}")
        # logging.debug("This message should go to the log file")
        # logging.info("So should this")
        # logging.warning("And this, too")
        # logging.error("And non-ASCII stuff, too, like Øresund and Malmö")

    def initialize_components(self):
        self.wt = Wind()
        self.pv = Solar()

        self.el = Electrolzyer()
        self.fc = FuelCell()
        self.hb = HaberBosch()
        self.asu = AirSeparationUnit()

        self.P2H2 = self.el.P2H2
        self.P2N2 = self.asu.P2N2
        self.P2NH3 = self.hb.P2NH3

        self.molmass_H2 = molmass.Formula("H2").mass  # g/mol
        self.molmass_N2 = molmass.Formula("N2").mass  # g/mol
        self.molmass_NH3 = molmass.Formula("NH3").mass  # g/mol

        self.comps = [self.el, self.fc, self.hb, self.asu]

        self.ci = {
            "atm": 0,
            "sun": 1,
            "pv": 2,
            "wt": 3,
            "paa": 4,
            "asu": 5,
            "el": 6,
            "hb": 7,
            "out": 8,
        }

    def set_component_matrices(self):
        for comp in self.comps:
            comp.I = self.I
            comp.P = self.P
            comp.C = self.C

    def get_component_matrices(self):
        for comp in self.comps:
            self.I = comp.I
            self.P = comp.P
            self.C = comp.C

    def update_time(self):
        self.wt.t = self.t
        self.pv.t = self.t
        self.el.t = self.t
        self.asu.t = self.t
        self.hb.t = self.t

    def step(self):
        # update time for all components
        self.update_time()

        # get wind and solar resource at time
        # self.wt.get_power()
        # self.pv.get_power()

        self.P[self.ci["paa"], self.ci["wt"]] = self.wt.get_power()
        self.P[self.ci["paa"], self.ci["pv"]] = self.pv.get_power()

        # allocate power
        self.PAA()

        # self.set_component_matrices()

        # calculate chemical
        self.C[self.ci["hb"], self.ci["el"]] = self.el.step(
            self.P[self.ci["el"], self.ci["paa"]]
        )
        self.C[self.ci["hb"], self.ci["asu"]] = self.asu.step(
            self.P[self.ci["asu"], self.ci["paa"]]
        )
        self.C[self.ci["out"], self.ci["hb"]] = self.hb.step(
            self.C[self.ci["hb"], self.ci["el"]], self.C[self.ci["hb"], self.ci["asu"]]
        )

        # self.get_component_matrices()

        # save outputs
        # logging.info(f"NH3 produced: { self.C[self.ci["hb"], self.ci["out"]] }")

        self.t += self.dt
        # print(self.t)

    def PAA(self):
        # power allocation algorithm

        # get wind and solar resource at time
        P = np.sum(self.P[self.ci["paa"], :])

        P_ratio = ((2) / (self.P2N2 * self.molmass_N2)) * (
            (3 * self.P2H2 * self.molmass_H2) / (2)
        )

        P_el = P * (P_ratio / (1 + P_ratio))
        P_asu = P * (1 / (P_ratio + 1))

        self.P[self.ci["asu"], self.ci["paa"]] = P_asu
        self.P[self.ci["el"], self.ci["paa"]] = P_el
