import numpy as np
import random 

def read_data(filename):
    with open(filename, 'r') as f:
        for line in f:
            iwp = line.strip().split()
            if len(iwp) >= 5 and iwp[2] == 'cities.':
                num_of_cities = int(iwp[1])
                break
                
        distance_matrix = []
        for line in f:
            distances = line.strip().split(',')
            if len(distances) == 1: continue
            distance_matrix.append([float(i) for i in distances])
    return num_of_cities, np.array(distance_matrix)


def fitness_function(route, distance_matrix):
    total_distance = 0
    for i in range(len(route)-1):
        total_distance += distance_matrix[route[i]][route[i+1]]
    return total_distance


def initialize(num_of_cities, num_of_population):
    population = []
    for _ in range(num_of_population):
        population.append(random.sample(list(range(num_of_cities)), num_of_cities))
    return np.array(population)