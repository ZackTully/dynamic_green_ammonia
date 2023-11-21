from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

x = XDSM(use_sfmath=True)

x.add_system("wind", FUNC, "wind")
x.add_system("pv", FUNC, "pv")
x.add_system("hybrid", FUNC, "hybrid")

x.add_system("EL", FUNC, "EL")
x.add_system("HB", FUNC, "HB")
x.add_system("ASU", FUNC, "ASU")
x.add_system("Batt", FUNC, "Batt")
x.add_system("H2_st", FUNC, "H2 st")


# power connections

x.connect("wind", "hybrid", "P_wind")
x.connect("pv", "hybrid", "P_pv")

x.connect("hybrid", "EL", "P_EL")
x.connect("hybrid", "ASU", "P_ASU")
x.connect("hybrid", "HB", "P_HB")
x.connect("hybrid", "Batt", "P_batt")
x.connect("Batt", "EL", "P_EL")
x.connect("Batt", "ASU", "P_ASU")
x.connect("Batt", "HB", "P_HB")

# chemical connections
x.connect("ASU", "HB", "N2")
x.connect("EL", "HB", "H2")
x.connect("EL", "H2_st", "H2")
x.connect("H2_st", "HB", "H2")


x.add_output("HB", "NH3", side=LEFT)


x.write("DL_org")
