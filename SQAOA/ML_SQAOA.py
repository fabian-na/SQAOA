import numpy as np
import networkx as nx
import cirq
import itertools
import time
import pickle
import os

from helpers.optimization import maximize_agree


class U_c(cirq.Gate):
    def __init__(self, G, current_n, gamma, const_gamma, q_map, use_const_costs=True):
        self.w = nx.adjacency_matrix(G)
        self.current_n = current_n
        self.gamma = gamma
        self.const_gamma = const_gamma
        self.q_map = q_map
        self.use_const_costs = use_const_costs

    def _num_qubits_(self):
        return self.current_n

    def _decompose_(self, qubits):
        if self.use_const_costs:
            for i in range(self.current_n):
                i_q = qubits[i]
                i_map = self.q_map[i]
                yield cirq.Rz(rads=2 * i_map * self.const_gamma).on(i_q)

        for i in range(self.current_n):
            for j in range(i + 1, self.current_n):
                i_q = qubits[i]
                j_q = qubits[j]
                i_map = self.q_map[i]
                j_map = self.q_map[j]

                if self.w[i_map, j_map] != 0:
                    yield cirq.CNOT(i_q, j_q)
                    yield cirq.Rz(rads=-2 * self.w[i_map, j_map] * self.gamma).on(j_q)
                    yield cirq.CNOT(i_q, j_q)

    def _circuit_diagram_info_(self, args):
        if self.use_const_costs:
            return [f"U_c({self.gamma}, {self.const_gamma})"] * self.current_n
        else:
            return [f"U_c({self.gamma})"] * self.current_n


class U_m(cirq.Gate):
    def __init__(self, current_n, beta):
        self.current_n = current_n
        self.beta = beta

    def _num_qubits_(self):
        return self.current_n

    def _decompose_(self, qubits):
        for i in range(self.current_n):
            i_q = qubits[i]

            yield cirq.Rx(rads=-2 * self.beta).on(i_q)

    def _circuit_diagram_info_(self, args):
        return [f"U_m({self.beta})"] * self.current_n


class Rh(cirq.Gate):
    def __init__(self, alpha):
        self.alpha = alpha

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        Rh_matrix = np.array(
            [
                [
                    np.cos(self.alpha) - np.sqrt(1 / 2) * np.sin(self.alpha) * (0 + 1j),
                    -np.sqrt(1 / 2) * np.sin(self.alpha) * (0 + 1j),
                ],
                [
                    -np.sqrt(1 / 2) * np.sin(self.alpha) * (0 + 1j),
                    np.cos(self.alpha) + np.sqrt(1 / 2) * np.sin(self.alpha) * (0 + 1j),
                ],
            ]
        )
        return Rh_matrix

    def _circuit_diagram_info_(self, args):
        return [f"Rh({self.alpha})"]


class U_t(cirq.Gate):
    def __init__(self, current_n, alpha):
        self.current_n = current_n
        self.alpha = alpha

    def _num_qubits_(self):
        return self.current_n

    def _decompose_(self, qubits):
        for i in range(self.current_n):
            i_q = qubits[i]

            yield Rh(alpha=self.alpha).on(i_q)

    def _circuit_diagram_info_(self, args):
        return [f"U_t({self.alpha})"] * self.current_n


def sqaoa(
    G,
    length,
    p,
    t,
    alpha,
    beta,
    gamma,
    same_beta,
    same_gamma,
    const_gamma,
    use_alpha,
    use_const_gamma,
    simulator,
    shots,
    old_probabilities,
    simulation,
):
    n_init = len(G.nodes)
    w = nx.adjacency_matrix(G)

    simulation_time = 0.0

    lookup_table = [{} for _ in range(length)]
    cbit_list = ["1" * n_init]

    current_probabilities = {}

    if length == 0:
        early_costs = 0.0
        for i in range(n_init):
            for j in range(i + 1, n_init):
                if w[i, j] > 0:
                    early_costs += w[i, j]

        return early_costs, {}

    prob_dicts = [{} for _ in range(length)]

    for current_iteration in range(length):
        new_cbit_list = []
        for cbit_str in cbit_list:
            used_params = []
            if use_alpha:
                if same_beta:
                    used_params.append(alpha[0])
                else:
                    used_params.append(alpha[current_iteration])
            else:
                used_params.append(np.pi / 2)

            if use_const_gamma:
                for j in range(p):
                    if same_gamma:
                        used_params.append((gamma[j], const_gamma[j]))
                    else:
                        used_params.append(
                            (
                                gamma[current_iteration * p + j],
                                const_gamma[current_iteration * p + j],
                            )
                        )
            else:
                for j in range(p):
                    if same_gamma:
                        used_params.append((gamma[j], 0))
                    else:
                        used_params.append((gamma[current_iteration * p + j], 0))

            if same_beta:
                for j in range(p):
                    used_params.append(beta[j])
            else:
                for j in range(p):
                    used_params.append(beta[current_iteration * p + j])

            used_params = tuple(used_params)

            if (used_params, cbit_str) in old_probabilities:
                cbit = np.array([int(c) for c in cbit_str])

                n_new = np.count_nonzero(cbit)

                q_map = {}
                for i, i_old in enumerate(np.nonzero(cbit)[0]):
                    q_map[i] = i_old

                if n_new == 1 or n_new == 0:
                    lookup_table[current_iteration][cbit_str] = 0.0
                    continue

                current_prob_list = old_probabilities[(used_params, cbit_str)]

            else:
                cbit = np.array([int(c) for c in cbit_str])

                n_new = np.count_nonzero(cbit)

                q_map = {}
                for i, i_old in enumerate(np.nonzero(cbit)[0]):
                    q_map[i] = i_old

                if n_new == 1 or n_new == 0:
                    lookup_table[current_iteration][cbit_str] = 0.0
                    continue

                qubits = cirq.LineQubit(0).range(n_new)
                cq = cirq.Circuit()

                if use_alpha:
                    if same_beta:
                        cq.append(U_t(n_new, alpha=alpha[0]).on(*qubits))
                    else:
                        cq.append(
                            U_t(n_new, alpha=alpha[current_iteration]).on(*qubits)
                        )
                else:
                    cq.append(
                        U_t(n_new, alpha=np.pi / 2).on(*qubits)
                    )  # alpha=0 --> identity, alpha=np.pi/2 --> iH

                if use_const_gamma:
                    for j in range(p):
                        if same_gamma:
                            cq.append(
                                U_c(
                                    G,
                                    n_new,
                                    gamma[j],
                                    const_gamma[j],
                                    q_map,
                                    use_const_gamma,
                                ).on(*qubits)
                            )
                        else:
                            cq.append(
                                U_c(
                                    G,
                                    n_new,
                                    gamma[current_iteration * p + j],
                                    const_gamma[current_iteration * p + j],
                                    q_map,
                                    use_const_gamma,
                                ).on(*qubits)
                            )

                        if same_beta:
                            cq.append(U_m(n_new, beta=beta[j]).on(*qubits))
                        else:
                            cq.append(
                                U_m(n_new, beta=beta[current_iteration * p + j]).on(
                                    *qubits
                                )
                            )

                else:
                    for j in range(p):
                        if same_gamma:
                            cq.append(
                                U_c(
                                    G, n_new, gamma[j], None, q_map, use_const_gamma
                                ).on(*qubits)
                            )
                        else:
                            cq.append(
                                U_c(
                                    G,
                                    n_new,
                                    gamma[current_iteration * p + j],
                                    None,
                                    q_map,
                                    use_const_gamma,
                                ).on(*qubits)
                            )

                        if same_beta:
                            cq.append(U_m(n_new, beta=beta[j]).on(*qubits))
                        else:
                            cq.append(
                                U_m(n_new, beta=beta[current_iteration * p + j]).on(
                                    *qubits
                                )
                            )

                start = time.time()
                if simulation == "sampling":
                    cq.append(cirq.measure(*qubits, key="m"))
                    result = simulator.run(cq, repetitions=shots)
                    measurements = result.measurements["m"]
                    measurements_str = [
                        "".join(str(e) for e in list(x)) for x in measurements
                    ]

                    max_keys = set()
                    count_max_keys = 0

                    while count_max_keys < len(measurements_str) * t:
                        max_key = max(
                            set(measurements_str) - max_keys, key=measurements_str.count
                        )
                        max_keys.add(max_key)
                        count_max_keys += measurements_str.count(max_key)

                    current_prob_list = [
                        (
                            np.array([int(c) for c in max_key]),
                            float(measurements_str.count(max_key) / count_max_keys),
                        )
                        for max_key in max_keys
                    ]

                elif simulation == "statevector":
                    result = simulator.simulate(cq)
                    current_prob_list = []
                    m_bits = ["".join(i) for i in itertools.product("01", repeat=n_new)]
                    m_bits.sort(key=lambda x: np.count_nonzero(x))

                    for i in range(len(m_bits)):
                        prob = np.square(np.abs(result.state_vector()[i]))
                        if prob > 0:
                            current_prob_list.append(
                                (np.array([int(c) for c in m_bits[i]]), prob)
                            )
                end = time.time()

                simulation_time += end - start

            step_costs = 0.0
            for sub_cbit, prob in current_prob_list:
                new_costs = 0.0

                for i in range(len(sub_cbit)):
                    for j in range(i + 1, len(sub_cbit)):
                        if w[q_map[i], q_map[j]] > 0:
                            new_costs += (
                                (1 - sub_cbit[i])
                                * (1 - sub_cbit[j])
                                * w[q_map[i], q_map[j]]
                            )
                            if current_iteration == length - 1:
                                new_costs += (
                                    sub_cbit[i] * sub_cbit[j] * w[q_map[i], q_map[j]]
                                )
                        if w[q_map[i], q_map[j]] < 0:
                            new_costs -= (
                                (1 - sub_cbit[i])
                                * (sub_cbit[j])
                                * w[q_map[i], q_map[j]]
                            )
                            new_costs -= (
                                (sub_cbit[i])
                                * (1 - sub_cbit[j])
                                * w[q_map[i], q_map[j]]
                            )

                step_costs += prob * new_costs

            current_probabilities[(used_params, cbit_str)] = current_prob_list
            prob_dicts[current_iteration][cbit_str] = current_prob_list
            lookup_table[current_iteration][cbit_str] = step_costs

            for sub_cbit, prob in current_prob_list:
                sub_cbit_str = [str(x) for x in sub_cbit]
                complete_cbit_str = ["0"] * n_init
                for i in range(len(sub_cbit)):
                    complete_cbit_str[q_map[i]] = sub_cbit_str[i]
                complete_cbit_str = "".join(complete_cbit_str)
                new_cbit_list.append(complete_cbit_str)

        cbit_list = list(set(new_cbit_list))

    for current_iteration in reversed(range(length)):
        for cbit_str in prob_dicts[current_iteration].keys():
            cbit = np.array([int(c) for c in cbit_str])
            q_map = {}
            for i, i_old in enumerate(np.nonzero(cbit)[0]):
                q_map[i] = i_old

            if current_iteration != length - 1:
                for sub_cbit, prob in prob_dicts[current_iteration][cbit_str]:
                    sub_cbit_str = [str(x) for x in sub_cbit]

                    complete_cbit_str = ["0"] * n_init
                    for i in range(len(sub_cbit_str)):
                        complete_cbit_str[q_map[i]] = sub_cbit_str[i]
                    complete_cbit_str = "".join(complete_cbit_str)
                    lookup_table[current_iteration][cbit_str] += (
                        prob * lookup_table[current_iteration + 1][complete_cbit_str]
                    )

    # 1 undecided, 0 decided
    return lookup_table[0]["1" * n_init], current_probabilities, simulation_time


class BlackBoxObjective:
    def __init__(
        self,
        G,
        length,
        p,
        t,
        same_parameters=False,
        use_alpha=False,
        use_const_gamma=True,
        shots=1000,
        simulation="sampling",
    ):
        self.G = G
        self.p = p
        self.t = t
        self.same_parameters = same_parameters
        self.use_alpha = use_alpha
        self.use_const_gamma = use_const_gamma
        self.shots = shots
        self.simulation = simulation

        self.n_parameters = 0

        self.length = 0
        self.set_length(length)

        self.old_lookup_table = {}
        self.simulation_time = 0.0

    def set_length(self, length):
        self.length = length - 1
        self.n_parameters = 0

        if self.same_parameters:
            if self.use_alpha:
                self.n_parameters += 1
            if self.use_const_gamma:
                self.n_parameters += self.p
            self.n_parameters += 2 * self.p
        else:
            if self.use_alpha:
                self.n_parameters += self.length + (self.length == 0)
            if self.use_const_gamma:
                self.n_parameters += self.length * self.p + (self.length == 0)
            self.n_parameters += 2 * self.length * self.p + 2 * (self.length == 0)

        self.old_lookup_table = {}
        self.simulation_time = 0.0

    def __call__(self, theta):
        constant_subproblems = np.ones(self.length, dtype=bool)
        constant_params = np.zeros(len(theta), dtype=bool)

        current_parameter = 0
        if self.same_parameters:
            if self.use_alpha:
                alpha = theta[:1]

                constant_subproblems[0] *= constant_params[0]
            else:
                alpha = []
            current_parameter += len(alpha)

            beta = theta[current_parameter : current_parameter + self.p]
            constant_subproblems *= np.all(
                constant_params[current_parameter : current_parameter + self.p]
            )
            current_parameter += len(beta)

            gamma = theta[current_parameter : current_parameter + self.p]
            constant_subproblems *= np.all(
                constant_params[current_parameter : current_parameter + self.p]
            )
            current_parameter += len(gamma)

            if self.use_const_gamma:
                const_gamma = theta[current_parameter : current_parameter + self.p]
                constant_subproblems *= np.all(
                    constant_params[current_parameter : current_parameter + self.p]
                )
            else:
                const_gamma = []

        else:
            if self.use_alpha:
                alpha = theta[: self.length]

                constant_subproblems *= constant_params[: self.length]
            else:
                alpha = []
            current_parameter += len(alpha)

            beta = theta[current_parameter : current_parameter + self.length * self.p]
            constant_subproblems *= np.all(
                constant_params[
                    current_parameter : current_parameter + self.length * self.p
                ].reshape(self.length, self.p),
                axis=1,
            )
            current_parameter += len(beta)

            gamma = theta[current_parameter : current_parameter + self.length * self.p]
            constant_subproblems *= np.all(
                constant_params[
                    current_parameter : current_parameter + self.length * self.p
                ].reshape(self.length, self.p),
                axis=1,
            )
            current_parameter += len(gamma)

            if self.use_const_gamma:
                const_gamma = theta[
                    current_parameter : current_parameter + self.length * self.p
                ]
                constant_subproblems *= np.all(
                    constant_params[
                        current_parameter : current_parameter + self.length * self.p
                    ].reshape(self.length, self.p),
                    axis=1,
                )
            else:
                const_gamma = []

        simulator = cirq.Simulator()
        approx_ratio, lookup_table, simulation_time = sqaoa(
            self.G,
            self.length,
            self.p,
            self.t,
            alpha,
            beta,
            gamma,
            self.same_parameters,
            self.same_parameters,
            const_gamma,
            self.use_alpha,
            self.use_const_gamma,
            simulator,
            self.shots,
            self.old_lookup_table,
            self.simulation,
        )

        self.old_theta = theta
        self.old_lookup_table = lookup_table
        self.simulation_time += simulation_time

        return -approx_ratio

    def bounds(self):
        return np.full((self.n_parameters, 2), [0, 2 * np.pi], dtype=float)

    def random_init_point(self):
        lb = np.zeros(self.n_parameters, dtype=float)
        ub = np.full(self.n_parameters, 2 * np.pi, dtype=float)

        return np.random.uniform(lb, ub, self.n_parameters)

    def optimal_init_point(
        self, G, method, budget=500, restarts=0, looping=False, use_bounds=False
    ):
        N = len(G.nodes)

        if os.path.isfile(f"init_points/init_SQAOA_{N}_{self.n_parameters}.p"):
            return np.array(
                pickle.load(
                    open(f"init_points/init_SQAOA_{N}_{self.n_parameters}.p", "rb")
                )
            )
        else:
            old_use_const_gamma = self.use_const_gamma
            self.use_const_gamma = False
            self.set_length(self.length + 1)

            _, init_points, _ = maximize_agree(
                self,
                G,
                method=method,
                budget=budget,
                looping=looping,
                restarts=restarts,
                init_point="random",
                use_bounds=use_bounds,
            )

            self.use_const_gamma = old_use_const_gamma
            self.set_length(self.length + 1)

        if self.use_const_gamma:
            if self.same_parameters:
                init_points = init_points + [0.0] * self.p
            else:
                init_points = init_points + [0.0] * (self.p * self.length)

        file = open(f"init_points/init_SQAOA_{N}_{self.n_parameters}.p", "ba")
        pickle.dump(init_points, file)
        file.close()

        return np.array(init_points)
