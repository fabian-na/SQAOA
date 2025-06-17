import numpy as np
import networkx as nx

from scipy.optimize import minimize as minimize_scipy
from skquant.opt import minimize as minimize_skquant


def max_agree(G):
    N = len(G.nodes)
    w = nx.adjacency_matrix(G)
    optimal_value = -np.inf
    optimal_solutions = []

    coloring = np.array(
        np.meshgrid(*[np.arange(x + 1) for x in np.arange(N)])
    ).T.reshape((-1, N))

    for c in coloring:
        current_value = 0

        for u, v in G.edges():
            if c[u] != c[v] and w[u, v] == -1:
                current_value += 1
            elif c[u] == c[v] and w[u, v] == 1:
                current_value += 1

        if current_value > optimal_value:
            optimal_value = current_value
            optimal_solutions = [c]
        elif current_value == optimal_value:
            optimal_solutions.append(c)

    return optimal_value, optimal_solutions


def _minimize_skquant(f, init_point, bounds, method, budget, use_bounds):
    if use_bounds:
        result, history = minimize_skquant(
            f, init_point, bounds=bounds, method=method, budget=budget
        )
    else:
        result, history = minimize_skquant(f, init_point, method=method, budget=budget)

    return -result.optval, result.optpar, len(history)


def _minimize_scipy(f, init_point, bounds, method, budget, use_bounds):
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        l = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        u = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)

    if use_bounds:
        res = minimize_scipy(
            f,
            init_point,
            bounds=bounds,
            method=method,
            options={"maxiter": budget},
        )
    else:
        res = minimize_scipy(
            f,
            init_point,
            method=method,
            options={"maxiter": budget},
        )

    x1 = []
    for i in res["x"]:
        x1.append(float(i))

    return -res["fun"], x1, res["nfev"]


def maximize_agree(
    objective,
    G,
    method,
    budget=500,
    restarts=0,
    init_point="random",
    looping=False,
    use_bounds=True,
):
    if looping is False:
        dim_list = [len(G.nodes)]
    else:
        dim_list = [i for i in range(1, len(G.nodes) + 1)]

    scipy_methods = ["Powell", "COBYLA"]
    skquant_methods = ["Bobyqa", "ImFil", "SnobFit"]

    best_val_dim = -np.inf
    best_res_dim = None

    for dim in dim_list:
        objective.set_length(dim)

        if isinstance(init_point, np.ndarray):
            init_point_current = init_point
        elif init_point == "random":
            init_point_current = objective.random_init_point()
        elif init_point == "optimal":
            init_point_current = objective.optimal_init_point(
                G,
                method,
                budget=2 * budget,
                restarts=2 * restarts,
                looping=False,
                use_bounds=use_bounds,
            )
        else:
            raise Exception(f"Cannot handle init_point {init_point}")

        bounds = objective.bounds()

        best_val_restart = -np.inf
        best_res_restart = None

        for _ in range(restarts + 1):
            if method in scipy_methods:
                res = _minimize_scipy(
                    objective,
                    init_point_current,
                    bounds=bounds,
                    method=method,
                    budget=budget,
                    use_bounds=use_bounds,
                )
            elif method in skquant_methods:
                res = _minimize_skquant(
                    objective,
                    init_point_current,
                    bounds=bounds,
                    method=method,
                    budget=budget,
                    use_bounds=use_bounds,
                )
            else:
                raise Exception(
                    f"Method {method} not in 'scipy_methods' or 'skquant_methods'."
                )

            if res[0] > best_val_restart:
                best_val_restart = res[0]
                best_res_restart = res

        if best_res_restart[0] > best_val_dim:
            best_val_dim = best_res_restart[0]
            best_res_dim = best_res_restart

    return best_res_dim
