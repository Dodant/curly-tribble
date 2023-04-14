import random


def read_data(filename):
    """Parse problem specifications from the data file."""
    with open(filename, "r") as f:
        # header
        for line in f:
            iwp = line.strip().split()
            if len(iwp) >= 4 and iwp[2] == "capacity":
                capacity = float(iwp[3])
            elif iwp == ["item_index", "weight", "profit"]:
                table = True
                break
        if not table:
            raise ValueError("table not found.")
        # body
        weights = []
        profits = []
        for line in f:
            i, w, p = line.strip().split()
            weights.append(float(w))
            profits.append(float(p))
    return capacity, weights, profits


def fitness_function(individual, capacity, weights, profits):
    """Calculate fitness value of an individual."""
    sum_weight = 0
    sum_profit = 0
    for bit, weight, profit in zip(individual, weights, profits):
        if bit == "1":
            sum_weight += weight
            sum_profit += profit

    fitness = sum_profit if sum_weight <= capacity else 0
    return fitness


def initialize():
    """Initialize 100 individuals, each of which consists of 10000 bits"""
    population = []
    for _ in range(100):
        individual = ""
        for _ in range(10000):
            individual += "1" if random.random() < 0.5 else "0"
        population.append(individual)
    return population
