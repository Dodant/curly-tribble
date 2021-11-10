import random
import copy

import numpy as np


N = 25
CROSSOVER_P = 0.9
MUTATION_P = 0.01
POPULATION_SIZE = 100
GENERATION_ITER = 50
OVERLAP = 5


def fitness(indi):
    return max(indi[0:N].count('0'), indi[0:N].count('1')) + max(indi[N:2*N].count('0'), indi[N:2*N].count('1'))


def crossover_overlap(fitness, population):
    fitness_list = [fitness(i) for i in population]
    overlap_samples = [population[i] for i in np.argsort(fitness_list)[::-1][:OVERLAP]]

    half = len(population)//2
    for i in range(half):
        if random.random() < CROSSOVER_P:
            for _ in range(3):  # 3 points
                pos = random.randint(0, len(population[i])-1) + 1
                tmp_i_l = copy.deepcopy(population[i][:pos])
                tmp_n_l = copy.deepcopy(population[i+half][:pos])
                tmp_i_u = copy.deepcopy(population[i][pos:])
                tmp_n_u = copy.deepcopy(population[i+half][pos:])

                population[i] = tmp_i_l + tmp_n_u
                population[i+half] = tmp_n_l + tmp_i_u
    
    for i in range(len(overlap_samples)):
        population[i] = copy.deepcopy(overlap_samples[i])

    random.shuffle(population)
    return population


def tournament(fitness, population):
    new_population = []
    for index, individual in enumerate(population):
        x = random.randint(0, POPULATION_SIZE-1)
        if fitness(individual) < fitness(population[x]):
            new_population.append(copy.deepcopy(population[x]))
        else:
            new_population.append(copy.deepcopy(population[index]))
    return new_population


# def crossover(population):
#     random.shuffle(population)
#     half = len(population)//2
#     for i in range(half):
#         if random.random() < CROSSOVER_P:
#             for _ in range(3):  # 3 points
#                 pos = random.randint(0, len(population[i])-1) + 1
#                 tmp_i_l = copy.deepcopy(population[i][:pos])
#                 tmp_n_l = copy.deepcopy(population[i+half][:pos])
#                 tmp_i_u = copy.deepcopy(population[i][pos:])
#                 tmp_n_u = copy.deepcopy(population[i+half][pos:])

#                 population[i] = tmp_i_l + tmp_n_u
#                 population[i+half] = tmp_n_l + tmp_i_u
#     return population


def mutation(population):
    for index, individual in enumerate(population):
        temp = copy.deepcopy(individual)
        for i in range(len(individual)):
            if random.random() < MUTATION_P:
                x = '0' if temp[i] == '1' else '1'
                temp = temp[:i] + x + temp[i+1:]
        population[index] = temp
    return population


def evaluation(population):
    return sum([fitness(i) for i in population]) / POPULATION_SIZE


def best_sample(population):
    return max([fitness(i) for i in population])


for n in range(10):
    random.seed(n)
    population = []
    for i in range(POPULATION_SIZE):
        txt = ''
        for j in range(2*N):
            txt += str(random.randint(0,1))
        population.append(txt)

    for i in range(GENERATION_ITER):
        temp_p = tournament(fitness, population)
        temp_p = mutation(temp_p)
        population = crossover_overlap(fitness, temp_p)
        print(f'Gen {i+1}, avg = {evaluation(population)}, best = {best_sample(population)}')

    txt = ""
    for indi in population:
        txt += indi + '\n'
    with open(f'output/fourmax_{n+1}.txt', "w") as f:
        f.write(txt)