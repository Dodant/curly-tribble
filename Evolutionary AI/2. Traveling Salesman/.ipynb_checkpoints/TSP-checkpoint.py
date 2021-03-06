import random
import copy

import numpy as np
import matplotlib.pyplot as plt

import source as src


CROSS_OVER_P = 0.1
MUTATION_P = 0.01
GEN_ITERATION = 6000
POPULATION_SIZE = 100


def tournament(fitness, population):
    new_population = []
    for individual in population:
        x = random.randint(0, POPULATION_SIZE-1)
        if fitness(individual, DISTANCE_MATRIX) > fitness(population[x], DISTANCE_MATRIX):
            new_population.append(copy.deepcopy(population[x]))
        else:
            new_population.append(copy.deepcopy(individual))
    return new_population


def roulette(fitness, population):
    fitness_list = [fitness(i, DISTANCE_MATRIX) for i in population]
    max_fit = max(fitness_list)
    reverse_fitness = [max_fit - i for i in fitness_list]
    total = sum(reverse_fitness)
    fitness_acc_list, fitness_acc = [], 0
    new_population = []

    for i in reverse_fitness:
        fitness_acc += i / total
        fitness_acc_list.append(fitness_acc)

    for _ in range(POPULATION_SIZE):
        x = random.random()
        for i in range(len(fitness_acc_list)):
            if x < fitness_acc_list[i]:
                new_population.append(population[i])
                break
    return new_population


def crossover(population):
    random.shuffle(population)
    for i in range(len(population)):
        if random.random() < CROSS_OVER_P:
            pos = random.randint(0, len(population[i])-1) + 1
            tmp_i_l = copy.deepcopy(population[i][:pos])
            tmp_i_u = copy.deepcopy(population[i][pos:])
            population[i] = np.append(tmp_i_u, tmp_i_l)
    return population


def elitism(population, T):
    fitness_scores = [src.fitness_function(i, DISTANCE_MATRIX) for i in population]
    min_index, max_index = np.argsort(fitness_scores)[:T], np.argsort(fitness_scores)[-T:]
    for i in range(T):
        population[max_index[i]] = copy.deepcopy(population[min_index[i]])
    return population


def mutation(population):
    for index, individual in enumerate(population):
        temp = copy.deepcopy(individual)
        for i in range(len(individual)):
            if random.random() < MUTATION_P:
                x = random.randint(0, len(individual)-1)
                temp[i], temp[x] = temp[x], temp[i]
        population[index] = temp
    return population


def evalution(population):
    fitness_scores = [src.fitness_function(i, DISTANCE_MATRIX) for i in population]
    average, best_sample = np.mean(fitness_scores), np.min(fitness_scores)
    return average, best_sample


for i in range(1, 31):
    num_of_cities, DISTANCE_MATRIX = src.read_data(f'data(TSP)/data-{i}.txt')
    
    population = src.initialize(num_of_cities, POPULATION_SIZE)
    aver, best = [], []

    for generation in range(GEN_ITERATION):
        temp_p = elitism(population, int(POPULATION_SIZE * 0.2))
        temp_p = tournament(src.fitness_function, temp_p)
        temp_p = roulette(src.fitness_function, temp_p)
        # temp_p = crossover(temp_p)
        population = mutation(temp_p)
        

        average_val, best_sample = evalution(population)
        aver.append((average_val, generation))
        best.append((best_sample, generation))
        
        if generation % 500 == 0:
            print(f'Gen {generation:>4d}: avg - {int(average_val)}  /  best - {int(best_sample)}')
        
    
    print(min(aver))
    print(min(best))
    plt.title("TSP fitness trace")
    plt.plot([i[0] for i in aver], label="Fitness, average")
    plt.plot([i[0] for i in best], label="Fitness, best")
    plt.legend()
    plt.savefig(f"output/trace-{i}.png")
    plt.close()
    
    txt = ""
    for route in population:
        fitness = src.fitness_function(route, DISTANCE_MATRIX)
        txt += f"{'-'.join([str(i) for i in route])}, {fitness}\n"
    with open(f"output/fitness-{i}.txt", "w") as f:
        f.write(txt)
    print(f'Data - {i}')