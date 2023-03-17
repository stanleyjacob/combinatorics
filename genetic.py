import numpy as np
import random
import matplotlib.pyplot as plt

def distance_matrix(cities):
    n = len(cities)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cities[i] - cities[j])
            matrix[i][j] = matrix[j][i] = d
    return matrix

def fitness(route, dist_matrix):
    total_distance = 0
    for i in range(1, len(route)):
        total_distance += dist_matrix[route[i - 1]][route[i]]
    total_distance += dist_matrix[route[-1]][route[0]]
    return -total_distance

def selection(population, fitnesses, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitnesses == np.max(fitnesses))
        parents[i] = population[max_fitness_idx[0][0]]
        fitnesses[max_fitness_idx[0][0]] = -np.inf
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint32(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring):
    for i in range(offspring.shape[0]):
        swap_indices = random.sample(range(offspring.shape[1]), 2)
        offspring[i][swap_indices] = offspring[i][swap_indices[::-1]]
    return offspring

def genetic_algorithm(cities, num_generations, population_size, num_parents):
    dist_matrix = distance_matrix(cities)
    population = [np.random.permutation(len(cities)) for _ in range(population_size)]

    best_fitness_history = []
    for generation in range(num_generations):
        fitnesses = [fitness(ind, dist_matrix) for ind in population]
        parents = selection(np.array(population), np.array(fitnesses), num_parents)
        offspring = crossover(parents, offspring_size=(population_size - num_parents, len(cities)))
        offspring = mutation(offspring)

        population[:num_parents, :] = parents
        population[num_parents:, :] = offspring

        best_fitness = np.max(fitnesses)
        best_fitness_history.append(best_fitness)
        print(f'Generation {generation}, best fitness: {best_fitness}')

    best_solution_idx = np.where(fitnesses == np.max(fitnesses))
    best_solution = population[best_solution_idx[0][0]]
    return best_solution, best_fitness_history
  
