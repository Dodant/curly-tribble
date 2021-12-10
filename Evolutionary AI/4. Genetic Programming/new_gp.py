import copy
import math
import random
from statistics import mean

import numpy as np

# bi
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b

# uni
def sqr(a): return a ** 2
def sin(a): return math.sin(a)


CROSSOVER_P = 0.9
MUTATION_P = 0.01
POPULATION_SIZE = 50
GENERATION_ITER = 10
BI_OPERATOR = [add, sub, mul]
UNI_OPERATOR = [sqr, sin]
TERMINAL = ['x', 'x', 'x', -2, -1, - 0, 1, 2]
MIN_DEPTH, MAX_DEPTH = 2, 4

xy_pair = []
filename = 'data/data-gp1.txt'
# filename = 'data/data-gp2.txt'

with open(filename, 'r') as f:
    for line in f:
        x, y = line.split(',')
        xy_pair.append((float(x), float(y)))


class GPtree:
    def __init__(self, data=None, left=None, right=None, root=False):
        self.data = data
        self.left = left
        self.right = right
        self.root = root

    def size(self):
        if self.data in TERMINAL: return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def label_name(self):
        if self.data in BI_OPERATOR or self.data in UNI_OPERATOR: return self.data.__name__
        else: return str(self.data)

    def disp_tree(self, pre):
        print(f'{pre}{self.label_name()}')
        if self.left: self.left.disp_tree(pre=pre+"~")
        if self.right: self.right.disp_tree(pre=pre+"~")

    def compute(self, x):
        if self.data in BI_OPERATOR: return self.data(self.left.calculate_x(x), self.right.calculate_x(x))
        elif self.data in UNI_OPERATOR: return self.data(self.left.calculate_x(x))
        elif self.data == 'x': return x
        else: return self.data

    def mutation(self):
        if random.random() < MUTATION_P: self.random_tree(grow=True, max_depth=2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()

    def build_tree(self, depth):
        if random.random() < 0.5:
            self.data = BI_OPERATOR[random.randint(0, len(BI_OPERATOR)-1)]
        else:
            self.data = UNI_OPERATOR[random.randint(0, len(UNI_OPERATOR)-1)]
        self.root = True

        if self.data in BI_OPERATOR:
            self.left = GPtree()
            self.left.build_subtree(depth-1)
            self.right = GPtree()
            self.right.build_subtree(depth-1)
        elif self.data in UNI_OPERATOR:
            self.left = GPtree()
            self.left.build_subtree(depth-1)

    def build_subtree(self, depth):
        if depth <= 0:
            self.data = TERMINAL[random.randint(0, len(TERMINAL)-1)]
        else:
            factor = random.random()
            if factor < 0.3:
                self.data = BI_OPERATOR[random.randint(0, len(BI_OPERATOR)-1)]
            elif 0.3 <= factor <= 0.6:
                self.data = UNI_OPERATOR[random.randint(0, len(UNI_OPERATOR)-1)]
            else:
                self.data = TERMINAL[random.randint(0, len(TERMINAL)-1)]

            if self.data in BI_OPERATOR:
                self.left = GPtree()
                self.left.build_subtree(depth-1)
                self.right = GPtree()
                self.right.build_subtree(depth-1)
            elif self.data in UNI_OPERATOR:
                self.left = GPtree()
                self.left.build_subtree(depth-1)


    # def build_subtree(self):
    #
    #
    #
    # def random_tree(self, grow, max_depth, depth=0):
    #     if depth < MIN_DEPTH or (depth < max_depth and not grow):
    #         self.data = OPERATOR[random.randint(0, len(OPERATOR) - 1)]
    #     elif depth >= max_depth:
    #         self.data = TERMINAL[random.randint(0, len(TERMINAL) - 1)]
    #     else:
    #         if random.random() > 0.5:
    #             self.data = TERMINAL[random.randint(0, len(TERMINAL) - 1)]
    #         else:
    #             self.data = OPERATOR[random.randint(0, len(OPERATOR) - 1)]
    #     if self.data in OPERATOR:
    #         self.left = GPtree()
    #         self.left.random_tree(grow, max_depth, depth=depth + 1)
    #         self.right = GPtree()
    #         self.right.random_tree(grow, max_depth, depth=depth + 1)
    #
    #
    # def build_subtree(self):  # count is list in order to pass "by reference"
    #     t = GPtree()
    #     t.data = self.data
    #     if self.left:  t.left = self.left.build_subtree()
    #     if self.right: t.right = self.right.build_subtree()
    #     return t
    #
    # def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
    #     count[0] -= 1
    #     if count[0] <= 1:
    #         if not second:  # return subtree rooted here
    #             return self.build_subtree()
    #         else:  # glue subtree here
    #             self.data = second.data
    #             self.left = second.left
    #             self.right = second.right
    #     else:
    #         ret = None
    #         if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
    #         if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
    #         return ret


def initialize():
    init_population = []
    for _ in range(POPULATION_SIZE):
        t = GPtree()
        t.build_tree(10)
        init_population.append(t)
    return init_population


def fitness_fn(individual, data_xy):
    return mean([abs(individual.calculate_x(x) - y) for (x, y) in data_xy])


def tournament(population_, fitness, data_xy):
    new_population = []
    for index, individual in enumerate(population_):
        x = random.randint(0, POPULATION_SIZE-1)
        if fitness(individual, data_xy) > fitness(population_[x], data_xy):
            new_population.append(copy.deepcopy(population_[x]))
        else:
            new_population.append(copy.deepcopy(individual))
    return new_population


def crossover(population_):
    new_population = []
    for _ in range(POPULATION_SIZE//2):
        frst = population_[random.randint(0, POPULATION_SIZE-1)]
        scnd = population_[random.randint(0, POPULATION_SIZE-1)]
        if random.random() < CROSSOVER_P:
            first = frst.cross([random.randint(1, scnd.size())], None)
            second = scnd.cross([random.randint(1, scnd.size())], None)
            frst.cross([random.randint(1, frst.size())], second)
            scnd.cross([random.randint(1, scnd.size())], first)
            new_population.append(copy.deepcopy(frst))
            new_population.append(copy.deepcopy(scnd))
        else:
            new_population.append(copy.deepcopy(frst))
            new_population.append(copy.deepcopy(scnd))
    return new_population


def mutation_pop(population_):
    for individual in population_:
        individual.mutation()
    return population_


def evaluation(fitness, population_, data_xy):
    fitness_score = [fitness(i, data_xy) for i in population_]
    return mean(fitness_score), min(fitness_score)


population = initialize()
print([i.size() for i in population])
population[0].disp_tree("")
print(population[0].compute(1))
# for i in range(GENERATION_ITER):
#     temp_p = tournament(population, fitness_fn, xy_pair)
#     temp_p = crossover(temp_p)
#     population = mutation_pop(temp_p)
#
#     avg, bst = evaluation(fitness_fn, population, xy_pair)
#     print(f'Gen {i + 1:>3d}: avg - {avg:.1f}, best - {bst:.1f}')
#     print(max([i.size() for i in population]))

    # fitness_score = [fitness(i, xy_pair) for i in population]
    # population[np.argmin(fitness_score)].disp_tree(pre="")
    # break
