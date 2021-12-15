import argparse
import copy
import math
import random
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from anytree import Node, RenderTree

THRESHOLD = 40
CROSSOVER_P = 0.9
MUTATION_P = 0.01
POPULATION_SIZE = 500
GENERATION_ITER = 100
BI_OPERATOR = ['+', '*', '-', '+', '*', '-']
UNI_OPERATOR = ['sin', 'cos', 'sqr', 'sin', 'cos']
OPERATORS = BI_OPERATOR + UNI_OPERATOR
TERMINAL = ['x'] * 30 + [i / 10 for i in range(-20, 21)]


def init_tree():
    name = random.choice(OPERATORS)
    root = Node(name)
    if name in UNI_OPERATOR:
        operand = Node(random.choice(TERMINAL), parent=root)
    if name in BI_OPERATOR:
        left = Node(random.choice(TERMINAL), parent=root)
        right = Node(random.choice(TERMINAL), parent=root)
    return root


def print_tree(node):
    for pre, fill, node in RenderTree(node):
        print(f'{pre}{node.name}')


def save_tree(node, dataset, trial):
    with open(f'{dataset}_{trial}.txt', 'w') as f:
        for pre, fill, node in RenderTree(node):
            f.write(f'{pre}{node.name}\n')
        f.close()


def grow_tree(node):
    if random.random() < 0.9:
        if node.is_leaf:
            node.name = random.choice(OPERATORS)
            if node.name in UNI_OPERATOR:
                operand = Node(random.choice(TERMINAL), parent=node)
            if node.name in BI_OPERATOR:
                left = Node(random.choice(TERMINAL), parent=node)
                right = Node(random.choice(TERMINAL), parent=node)
        else:
            if node.name in UNI_OPERATOR:
                grow_tree(node.children[0])
            else:
                grow_tree(node.children[0])
                grow_tree(node.children[1])


def tournament(population, fitness, xy_pair):
    new_p = []
    for index, individual in enumerate(population):
        rand_tree = random.choice(population)
        if fitness(individual, xy_pair) > fitness(rand_tree, xy_pair):
            new_p.append(copy.deepcopy(rand_tree))
        else:
            new_p.append(copy.deepcopy(individual))
    del population
    return new_p


def crossover(first, second):
    try:
        if first != second:
            first_subtree = random.choice(first.descendants)
            first_subtree_parent = first_subtree.parent

            second_subtree = random.choice(second.descendants)
            second_subtree_parent = second_subtree.parent

            first_subtree.parent = None
            second_subtree.parent = None
            second_subtree.parent = first_subtree_parent
            first_subtree.parent = second_subtree_parent
    except:
        print(first == second)
        print_tree(first)
        print_tree(second)


def crossover_pop(population):
    for _ in range(POPULATION_SIZE):
        if random.random() < CROSSOVER_P:
            first = random.choice(population)
            second = random.choice(population)
            crossover(first, second)
    return population


def mutation(node):
    if random.random() < MUTATION_P:
        if node.name in BI_OPERATOR:
            node.name = random.choice(BI_OPERATOR)
            mutation(node.children[0])
            mutation(node.children[1])
        if node.name in UNI_OPERATOR:
            node.name = random.choice(UNI_OPERATOR)
            mutation(node.children[0])
        if node.name in TERMINAL:
            node.name = random.choice(TERMINAL)


def mutation_pop(population):
    for i in population:
        mutation(i)
    return population


def initialize_population(population_size):
    init_population = []
    for _ in range(population_size):
        tree = init_tree()
        grow_tree(tree)
        grow_tree(tree)
        init_population.append(tree)
    return init_population


def compute(node, x):
    if node.is_leaf:
        if node.name == 'x':
            return x
        return float(node.name)
    if node.name in BI_OPERATOR:
        return eval(f'{compute(node.children[0], x)}{node.name}{compute(node.children[1], x)}')
    if node.name in UNI_OPERATOR:
        if node.name == 'sqr':
            return eval(f'{compute(node.children[0], x)}**2')
        return eval(f'math.{node.name}({x})')


def fitness_fn(tree, xy_pair):
    return mean([(compute(tree, x) - y) ** 2 for (x, y) in xy_pair]) ** 0.5


def evaluation(fitness, population_, xy_pair):
    fitness_score = [fitness(i, xy_pair) for i in population_]
    return mean(fitness_score), min(fitness_score)


def elitism(population, fitness, xy_pair, T):
    fitness_scores = [fitness(i, xy_pair) for i in population]
    min_index, max_index = np.argsort(fitness_scores)[:T], np.argsort(fitness_scores)[-T:]
    for i in range(T):
        population[max_index[i]] = copy.deepcopy(population[min_index[i]])
    return population


def train(dataset, trial):
    xy_pair = []
    x_list, y_list = [], []
    filename = f'data/data-gp{dataset}.txt'

    with open(filename, 'r') as f:
        for line in f:
            x, y = line.split(',')
            x_list.append(float(x))
            y_list.append(float(y))
            xy_pair.append((float(x), float(y)))

    best_fit = 1000
    best_tree = init_tree()
    population = initialize_population(POPULATION_SIZE)

    for i in range(500):
        temp_p = tournament(population, fitness_fn, xy_pair)
        temp_p = crossover_pop(temp_p)
        temp_p = elitism(temp_p, fitness_fn, xy_pair, THRESHOLD)
        population = mutation_pop(temp_p)

        avg, bst = evaluation(fitness_fn, population, xy_pair)
        cand_tree = population[np.argmin([fitness_fn(i, xy_pair) for i in population])]
        cand_fit = fitness_fn(cand_tree, xy_pair)

        if best_fit > cand_fit:
            best_tree = cand_tree
            best_fit = cand_fit

            print_tree(best_tree)
            plt.scatter(x_list, y_list)
            plt.scatter(np.arange(-3, 3, 0.01), [compute(best_tree, x) for x in np.arange(-3, 3, 0.01)])
            plt.savefig(f'images/gp{dataset}/{trial}/gen{i+1}_{best_fit:.1f}_{len(best_tree.descendants)+1}.png')
            plt.close()

        print(f'Gen {i + 1:>3d}: avg - {avg:.1f}, best - {bst:.1f}, best_fit - {best_fit:.1f}')
        save_tree(best_tree, dataset, trial)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=int)
    parser.add_argument('-trial', type=int)

    args = parser.parse_args()
    train(args.dataset, args.trial)
