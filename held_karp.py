import numpy as np
from itertools import combinations, permutations

def held_karp(dist_matrix):
    n = len(dist_matrix)

    memo = {(frozenset([i]), i): (cost, [0, i]) for i, cost in enumerate(dist_matrix[0][1:], 1)}

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            frozen_subset = frozenset(subset)

            for k in subset:
                min_cost = float('inf')
                min_prev_vertex = None
                current_subset = frozen_subset - {k}

                for m in subset:
                    if m == k:
                        continue

                    cost = memo[current_subset, m][0] + dist_matrix[m][k]
                    if cost < min_cost:
                        min_cost = cost
                        min_prev_vertex = m

                memo[frozen_subset, k] = (min_cost, memo[current_subset, min_prev_vertex][1] + [k])

    full_subset = frozenset(range(1, n))
    min_cost = float('inf')
    min_prev_vertex = None

    for k in range(1, n):
        cost = memo[full_subset, k][0] + dist_matrix[k][0]
        if cost < min_cost:
            min_cost = cost
            min_prev_vertex = k

    opt_path = memo[full_subset, min_prev_vertex][1] + [0]

    return min_cost, opt_path

dist_matrix = np.array([
    [0, 20, 30, 10, 20],
    [20, 0, 15, 35, 25],
    [30, 15, 0, 25, 20],
    [10, 35, 25, 0, 15],
    [20, 25, 20, 15, 0]
])

min_cost, opt_path = held_karp(dist_matrix)
print(f"Minimum cost: {min_cost}")
print(f"Optimal path: {opt_path}")
