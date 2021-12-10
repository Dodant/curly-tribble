import copy
import math
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def sqr(a, b): return a ** 2
def sin(a, b): return math.sin(a)
def cos(a, b): return math.cos(a)
# def sqr(a, b): return math.sqrt(a)
# def pw(a, b): return math.pow(a, b)

CROSSOVER_P = 0.8
MUTATION_P = 0.01
POPULATION_SIZE = 500
GENERATION_ITER = 100
OPERATOR = [add, sub, mul, sqr, sin, cos]
TERMINAL = ['x', 'x', 'x', 'x', 'x', 'x', -2, -1, -.5, .5, 1, 2, math.e]
MIN_DEPTH, MAX_DEPTH = 2, 3


class GPtree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def label_name(self):
        if self.data in OPERATOR: return self.data.__name__
        else: return str(self.data)

    def disp_tree(self, pre):
        print(f'{pre}{self.label_name()}')
        if self.left: self.left.disp_tree(pre=pre+"_")
        if self.right: self.right.disp_tree(pre=pre+"_")

    def calculate_x(self, x):
        if self.data in OPERATOR:
            if self.data in [sqr, sin, cos]:
                self.right = None
                return self.data(self.left.calculate_x(x), None)
            return self.data(self.left.calculate_x(x), self.right.calculate_x(x))
        elif self.data == 'x': return x
        else: return self.data

    def build_tree(self, grow, max_depth, depth=0):
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = OPERATOR[random.randint(0, len(OPERATOR)-1)]
        elif depth >= max_depth:
            self.data = TERMINAL[random.randint(0, len(TERMINAL)-1)]
        else:
            if random.random() > 0.5: self.data = TERMINAL[random.randint(0, len(TERMINAL)-1)]
            else: self.data = OPERATOR[random.randint(0, len(OPERATOR)-1)]
        if self.data in OPERATOR:
            self.left = GPtree()
            self.left.build_tree(grow, max_depth, depth=depth+1)
            self.right = GPtree()
            self.right.build_tree(grow, max_depth, depth=depth+1)

    def size(self):
        if self.data in TERMINAL: return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def mutation(self):
        if random.random() < MUTATION_P: self.build_tree(grow=True, max_depth=2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()

    def build_subtree(self):
        t = GPtree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def cross(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second: return self.build_subtree()
            else:
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.cross(count, second)
            if self.right and count[0] > 1: ret = self.right.cross(count, second)
            return ret


def initialize():
    init_population = []
    for _ in range(POPULATION_SIZE // 2):
        t = GPtree()
        t.build_tree(grow=True, max_depth=MAX_DEPTH)  # grow
        init_population.append(t)
    for _ in range(POPULATION_SIZE // 2):
        t = GPtree()
        t.build_tree(grow=False, max_depth=MAX_DEPTH)  # full
        init_population.append(t)
    return init_population


def fitness_fn(individual, data_xy):
    return mean([abs(individual.calculate_x(x) - y) for (x, y) in data_xy])


def tournament(population_, fitness, data_xy):
    new_p = []
    for index, individual in enumerate(population_):
        x = random.randint(0, POPULATION_SIZE-1)
        if fitness(individual, data_xy) > fitness(population_[x], data_xy): new_p.append(copy.deepcopy(population_[x]))
        else: new_p.append(copy.deepcopy(individual))
    del population_
    return new_p


def crossover(population_):
    new_p = []
    for _ in range(POPULATION_SIZE//2):
        frst = population_[random.randint(0, POPULATION_SIZE-1)]
        scnd = population_[random.randint(0, POPULATION_SIZE-1)]
        if random.random() < CROSSOVER_P:
            first = frst.cross([random.randint(1, scnd.size())], None)
            second = scnd.cross([random.randint(1, scnd.size())], None)
            frst.cross([random.randint(1, frst.size())], second)
            scnd.cross([random.randint(1, scnd.size())], first)
            new_p.append(copy.deepcopy(frst))
            new_p.append(copy.deepcopy(scnd))
        else:
            new_p.append(copy.deepcopy(frst))
            new_p.append(copy.deepcopy(scnd))
    del population_
    return new_p


def mutation_pop(population_):
    for individual in population_: individual.mutation()
    return population_


def constraint(population_):
    new_p = []
    for indi in population_:
        if indi.size() >= 1000 or fitness_fn(indi, xy_pair) > 50000:
            new_indi = GPtree()
            new_indi.build_tree(grow=True, max_depth=4)
            new_p.append(new_indi)
            pass
        else: new_p.append(indi)
    del population_
    return new_p


def evaluation(fitness, population_, data_xy):
    fitness_score = [fitness(i, data_xy) for i in population_]
    return mean(fitness_score), min(fitness_score)


for n in range(2):
    xy_pair = []
    filename = f'data/data-gp2.txt'

    with open(filename, 'r') as f:
        for line in f:
            x, y = line.split(',')
            xy_pair.append((float(x), float(y)))

    population = initialize()
    best_tree = 0
    best_fit = 100

    for i in range(40):
        temp_p = tournament(population, fitness_fn, xy_pair)
        temp_p = crossover(temp_p)
        temp_p = constraint(temp_p)
        population = mutation_pop(temp_p)

        avg, bst = evaluation(fitness_fn, population, xy_pair)
        cand_tree = population[np.argmin([fitness_fn(i, xy_pair) for i in population])]
        cand_fit = fitness_fn(cand_tree, xy_pair)
        if best_fit > cand_fit:
            best_tree = cand_tree
            best_fit = cand_fit
        print(f'Gen {i+1:>3d}: avg - {avg:.1f}, best - {bst:.1f}, max size - {max([i.size() for i in population])}')

    best_tree.disp_tree(pre="")

    x_list, y_list = [], []

    with open(filename, 'r') as f:
        for line in f:
            x, y = line.split(',')
            x_list.append(float(x))
            y_list.append(float(y))

    plt.scatter(x_list, y_list)
    plt.scatter(np.arange(-3, 3, 0.01), [best_tree.calculate_x(x) for x in np.arange(-3, 3, 0.01)])
    plt.savefig(f'result4.png')
    plt.close()


