import matplotlib.pyplot as plt
import source as src
import copy
import random

cross_over_p = 0.9
mutation_p = 0.01
generation_iter = 100
population_size = 100


def tournament(fitness, population, capacity, weights, profits):
    for index, individual in enumerate(population):
        x = random.randint(0, population_size - 1)
        if fitness(individual, capacity, weights, profits) < fitness(population[x], capacity, weights, profits):
            population[index] = copy.deepcopy(population[x])
    return population


def roulette(fitness, population, capacity, weights, profits):
    fitness_list = [fitness(i, capacity, weights, profits) for i in population]
    total, min_fit = sum(fitness_list), min(fitness_list)
    fitness_acc_list, fitness_acc = [], 0
    new_population = []

    # Standard
    # for i in fitness_list:
    #     fitness_acc += i / total
    #     fitness_acc_list.append(fitness_acc)

    # Subtract minimum
    for i in fitness_list:
        fitness_acc += (i - min_fit) / (total - min_fit * population_size)
        fitness_acc_list.append(fitness_acc)

    for _ in range(population_size):
        x = random.random()
        for i in range(len(fitness_acc_list)):
            if x < fitness_acc_list[i]:
                new_population.append(population[i])
                break

    return new_population


def crossover(population):
    random.shuffle(population)
    half = len(population) // 2
    for i in range(half):
        if random.random() < cross_over_p:
            for _ in range(3):  # 3 points
                pos = random.randint(0, len(population[i]) - 1) + 1
                tmp_i_l = copy.deepcopy(population[i][:pos])
                tmp_n_l = copy.deepcopy(population[i + half][:pos])
                tmp_i_u = copy.deepcopy(population[i][pos:])
                tmp_n_u = copy.deepcopy(population[i + half][pos:])

                population[i] = tmp_i_l + tmp_n_u
                population[i + half] = tmp_n_l + tmp_i_u
    return population


def mutation(population):
    for index, individual in enumerate(population):
        temp = copy.deepcopy(individual)
        for i in range(len(individual)):
            if random.random() < mutation_p:
                x = '0' if temp[i] == '1' else '1'
                temp = temp[:i] + x + temp[i + 1:]
        population[index] = temp
    return population


def evaluation(population, capacity, weights, profits):
    return sum([src.fitness_function(i, capacity, weights, profits) for i in population]) / population_size


def best_sample(population, capacity, weights, profits):
    return max([src.fitness_function(i, capacity, weights, profits) for i in population])


# population = src.initialize()
# for generation in range(generation_iter):
#     temp_p = roulette(src.fitness_function, population)
#     temp_p = crossover(temp_p)
#     population = mutation(temp_p)
#     print(f'Gen {generation+1:>3d}: avg - {int(evaluation(population))}  /  best - {int(best_sample(population))}')
#
# population = src.initialize()
# for generation in range(generation_iter):
#     temp_p = tournament(src.fitness_function, population)
#     temp_p = crossover(temp_p)
#     population = mutation(temp_p)
#     print(f'Gen {generation+1:>3d}: avg - {int(evaluation(population))}  /  best - {int(best_sample(population))}')


def generate_example(data, figure, tournament_txt, roulette_txt):
    """Generate example outputs"""
    spec = src.read_data(data)
    pop1 = src.initialize()
    pop2 = copy.deepcopy(pop1)
    pop1_evaluation, pop2_evaluation = [], []

    # tournament
    for _ in range(generation_iter):
        temp_p = tournament(src.fitness_function, pop1, *spec)
        temp_p = crossover(temp_p)
        pop1 = mutation(temp_p)
        pop1_evaluation.append(evaluation(pop1, *spec))

    # roulette wheel
    for _ in range(generation_iter):
        temp_p = roulette(src.fitness_function, pop2, *spec)
        temp_p = crossover(temp_p)
        pop2 = mutation(temp_p)
        pop2_evaluation.append(evaluation(pop2, *spec))

    plt.title("0/1 Knapsack fitness value trace")
    plt.plot(range(100), pop1_evaluation, label="Pairwise Tournament Selection")
    plt.plot(range(100), pop2_evaluation, label="Roulette Wheel Selection")
    plt.legend()
    plt.savefig(figure)
    plt.show()

    txt = ""
    for ind1 in pop1:
        fit1 = src.fitness_function(ind1, *spec)
        txt += "{},{:.6f}\n".format(ind1, fit1)
    with open(tournament_txt, "w") as f:
        f.write(txt)

    txt = ""
    for ind2 in pop2:
        fit2 = src.fitness_function(ind2, *spec)
        txt += "{},{:.6f}\n".format(ind2, fit2)
    with open(roulette_txt, "w") as f:
        f.write(txt)


if __name__ == '__main__':
    generate_example("Data(0-1Knapsack).txt", "trace.png", "tournament.txt", "roulette.txt")
    print("Done!")
