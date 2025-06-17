import numpy as np
import networkx as nx
import pickle
import os


def create_random_ER_graph(N, p_edge, p_cost):
    G = nx.generators.random_graphs.erdos_renyi_graph(N, p_edge)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = np.random.choice([1, -1], p=[p_cost, 1 - p_cost])
    return G


def data_set_generation(N, number_of_graphs):
    if os.path.isfile(f"datasets/complete_{N}_{number_of_graphs}.p"):
        return

    data = []
    for i in range(number_of_graphs):
        p_cost = i / (number_of_graphs - 1)
        G = create_random_ER_graph(N, 1.0, p_cost)
        data.append(nx.to_numpy_array(G))

    file = open(f"datasets/complete_{N}_{number_of_graphs}.p", "ba")
    pickle.dump(data, file)


def data_set_generation_ER(N, p_edge, number_of_graphs):
    if os.path.isfile(f"datasets/ER_{N}_{number_of_graphs}_{int(p_edge * 100)}.p"):
        return

    data = []
    for i in range(number_of_graphs):
        p_cost = i / (number_of_graphs - 1)
        done = False
        while not done:
            G = create_random_ER_graph(N, p_edge, p_cost)
            if len(G.edges()) > 0:
                done = True
        data.append(nx.to_numpy_array(G))

    file = open(f"datasets/ER_{N}_{number_of_graphs}_{int(p_edge * 100)}.p", "ba")
    pickle.dump(data, file)
