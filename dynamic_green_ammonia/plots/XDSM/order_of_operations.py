from pyxdsm.XDSM import XDSM, OPT, SOLVER, DOE, FUNC, LEFT, GROUP

# optional_package = ["courier"]

x = XDSM(use_sfmath=True)  # , optional_latex_packages=optional_package)

# step 1: size generation
x.add_system("size_gen", DOE, ["1.\hspace{1mm}Size", "generation"])
x.add_input("size_gen", ["Wind cap.", "PV cap.", "EL cap."])


# step 2: simulate generation
x.add_system("sim_gen", DOE, ["2.\hspace{1mm}Simulate", "generation"])
x.add_input("sim_gen", ["Wind\hspace{1mm}resource", "Solar\hspace{1mm}resource"])
x.connect("size_gen", "sim_gen", "Capacity_{gen.}")


# step3: size end-use
x.add_system("size_end", DOE, ["3. Size", "ammonia"])
x.add_input("size_end", ["Turndown\hspace{1mm}f_T", "Ramping\hspace{1mm}f_R"])
x.connect("sim_gen", "size_end", "P, H_2")


# step 4: optimize end-use schedule
x.add_system("sched_opt", OPT, ["4.\hspace{1mm}Scheduling", "Optimization"])

x.connect("sim_gen", "sched_opt", "P, H_2")
x.connect("size_end", "sched_opt", "Capacity_{ammonia}")

# x.add_output("sched_opt", "Ammonia")
# x.connect("sched_opt", "nh3", "")

# step 5: size storage
x.add_system("size_storage", DOE, ["5.\hspace{1mm}Size", "storage"])

x.connect("sim_gen", "size_storage", "P, H_2")
x.connect("sched_opt", "size_storage", "H_2\hspace{1mm}demand")

# step 6: calculate costs
x.add_system("calc_cost", DOE, ["6.\hspace{1mm}Calculate", "costs"])

x.connect("size_gen", "calc_cost", "Capacity_{gen.}")
x.connect("size_end", "calc_cost", "Capacity_{ammonia}")
x.connect("size_storage", "calc_cost", "Capacity_{storage}")

# step 7: calculate LCOA
x.add_system("calc_LCOA", DOE, ["7.\hspace{1mm}Calculate", "LCOA"])

x.connect("sched_opt", "calc_LCOA", "Ammonia")
x.connect("calc_cost", "calc_LCOA", "Cost_{system}")

x.add_output("calc_LCOA", "LCOA", side=LEFT)

# x.write
x.write("order_of_ops")
# x.write("order_of_operations")
