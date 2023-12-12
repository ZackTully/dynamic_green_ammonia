import numpy as np
from scipy.optimize import linprog


class DemandOptimization:
    def __init__(self, H2_gen, ramp_lim, min_demand, max_demand, x0=None):
        self.optimization_version = 2
        self.H2_gen = H2_gen
        self.N = len(self.H2_gen)  # number of time steps in optimization
        self.ramp_lim = ramp_lim
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.x0 = x0

    def optimize(self):
        if self.min_demand >= self.max_demand:
            # if False:
            x, success = self.static_demand()
            res = None
        else:
            if self.optimization_version == 1:
                x, success, res = self.optimize_v1()
            elif self.optimization_version == 2:
                x, success, res = self.optimize_v2()

        self.demand = x[0 : self.N]
        self.state = x[self.N : 2 * self.N]
        self.capacity = x[-2] - x[-1]

        return x, success, res

    def static_demand(self):
        demand = self.max_demand * np.ones(len(self.H2_gen))
        storage_state = np.cumsum(self.H2_gen - demand)
        storage_state -= np.min(storage_state)
        state_max = np.max(storage_state)
        state_min = np.min(storage_state)
        x = np.concatenate([demand, storage_state, [state_max, state_min]])
        success = True
        return x, success

    def optimize_v1(self):
        n_steps = len(self.H2_gen)

        c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])

        A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
        b_ub = np.zeros([n_steps * 4])
        A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
        b_eq = np.zeros(n_steps + 1)

        A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]
        for i in range(n_steps):
            A_ub[i, [i + n_steps, -2]] = [1, -1]
            A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

            if i > 0:
                A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
                A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
            b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [
                self.ramp_lim,
                self.ramp_lim,
            ]

            b_eq[i] = self.H2_gen[i]
            if i == 0:
                A_eq[0, [0, n_steps]] = [1, 1]
                continue
            A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

        # bound_low = [self.min_demand] * n_steps + [None] * (n_steps + 2)
        bound_low = [self.min_demand] * n_steps + [0] * n_steps + [None] * 2
        bound_up = [self.max_demand] * n_steps + [None] * (n_steps + 2)
        bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
        x = res.x
        success = res.success

        return x, success, res

    def optimize_v2(self):
        N = len(self.H2_gen)

        # plant paramaters
        # td = 0.5
        # rl = 0.01 * (1 - td)
        # td = 0.3813675880432129
        # rl = 0.9

        rl = self.ramp_lim

        # center = np.interp(td, [0, 1], [np.max(self.H2_gen) / 2, np.mean(self.H2_gen)])
        # # center = np.mean(H2_gen)
        # max_demand = (2 / (td + 1)) * center
        # min_demand = td * max_demand
        min_demand = self.min_demand
        max_demand = self.max_demand

        R = rl * max_demand
        R = self.ramp_lim

        # H2_gen = np.linspace(0, 1.5 * max_demand, N) + 3 * np.random.random(N)

        # x = [u_0, ... , u_N, x_0, ... , x_N, x_max, x_min]
        u_min = min_demand
        u_max = max_demand

        # Cost vector
        C = np.zeros(N + N + 2)
        # C[N : N + N] = 1 * np.ones(N)
        C[2 * N + 0] = 1  # highest storage state
        C[2 * N + 1] = -1  # lowest storage state

        # Upper and lower bounds
        bound_l = np.concatenate(
            [
                [u_min] * N,  # demand lower bound
                [0] * N,  # storage state lower bound
                [None, 0],  # storage state max, min lower bound
            ]
        )

        bound_u = np.concatenate(
            [
                [u_max] * N,  # demand upper bound,
                [None] * N,  # storage state upper bound,
                [None, None],  # storage state max, min upper bound
            ]
        )

        # Positive demand ramp rate limit
        Aub_ramp_pos = np.zeros([N, N + N + 2])
        bub_ramp_pos = np.zeros(N)

        # u[k+1] - u[k] <= R
        # x[k+1] - x[k] <= R
        for k in range(N):
            if (k + 1) == N:
                break
            Aub_ramp_pos[k, k + 1] = 1
            Aub_ramp_pos[k, k] = -1
            bub_ramp_pos[k] = R

        # Negative demand ramp rate limit
        Aub_ramp_neg = np.zeros([N, N + N + 2])
        bub_ramp_neg = np.zeros(N)

        # -u[k+1] + u[k] <= R
        # -x[k+1] + x[k] <= R
        for k in range(N):
            if (k + 1) == N:
                break
            Aub_ramp_neg[k, k + 1] = -1
            Aub_ramp_neg[k, k] = 1
            bub_ramp_neg[k] = R

        # x_max
        Aub_xmax = np.zeros([N, N + N + 2])
        bub_xmax = np.zeros(N)

        # state[k] - state_max <= 0
        # x[N+k] - x[N+N] <= 0
        for k in range(N):
            Aub_xmax[k, N + k] = 1
            Aub_xmax[k, N + N] = -1
            bub_xmax[k] = 0

        # x_min
        Aub_xmin = np.zeros([N, N + N + 2])
        bub_xmin = np.zeros(N)

        # -state[k] + state_min <= 0
        # -x[N+k] + x[N+N+1] <= 0
        for k in range(N):
            Aub_xmin[k, N + k] = -1
            Aub_xmin[k, N + N + 1] = 1
            bub_xmin[k] = 0

        # Storage "dynamics"
        Aeq_dyn = np.zeros([N, N + N + 2])
        beq_dyn = np.zeros(N)

        # state[k+1] - state[k] + demand[k] = H2_gen[k]
        # x[N+k+1] - x[N+k] + x[k] = beq_dyn[k]
        for k in range(N):
            if (k + 1) == N:
                break
            Aeq_dyn[k, N + k + 1] = 1
            Aeq_dyn[k, N + k] = -1
            Aeq_dyn[k, k] = 1

            beq_dyn[k] = self.H2_gen[k]

        # state[0] = state[N]
        # -x[N+0] + x[N + N - 1] = 0
        Aeq_dyn[N - 1, N] = -1
        Aeq_dyn[N - 1, 2 * N - 1] = 1

        A_ub = np.concatenate([Aub_ramp_pos, Aub_ramp_neg, Aub_xmax, Aub_xmin])
        b_ub = np.concatenate([bub_ramp_pos, bub_ramp_neg, bub_xmax, bub_xmin])

        A_eq = Aeq_dyn
        b_eq = beq_dyn

        bounds = [(bound_l[i], bound_u[i]) for i, bl in enumerate(bound_l)]


        if self.x0 is not None:
            res = linprog(
                c=C,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                x0=self.x0
            )
        else: 
            res = linprog(
                c=C,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
            )

        return res.x, res.success, res

    def calc_proportional_demand(self, signal_ref, signal_des):
        """Return a demand profile proportional to the optimized demand profile by a
        factor of sum(signal_des) / sum(signal_ref)

        Args:
        signal_ref:
        signal_des:

        Returns:
        demand:
        state:
        capacity:
        """

        factor = np.sum(signal_des) / np.sum(signal_ref)
        return factor * self.demand, factor * self.state, factor * self.capacity
