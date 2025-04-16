import time
import random
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt


def traversal(poz, chromosome):

    if chromosome.gen[poz] in chromosome.terminal_set:
        return poz + 1
    elif chromosome.gen[poz] in chromosome.func_set[1]:
        return traversal(poz + 1, chromosome)
    else:
        new_poz = traversal(poz + 1, chromosome)
        return traversal(new_poz, chromosome)

def mutate(chromosome):

    poz = np.random.randint(len(chromosome.gen))
    if chromosome.gen[poz] in chromosome.func_set[1] + chromosome.func_set[2]:
        if chromosome.gen[poz] in chromosome.func_set[1]:
            chromosome.gen[poz] = random.choice(chromosome.func_set[1])
        else:
            chromosome.gen[poz] = random.choice(chromosome.func_set[2])
    else:
        chromosome.gen[poz] = random.choice(chromosome.terminal_set)
    return chromosome

def selection(population, num_sel):

    sample = random.sample(population.list, num_sel)
    best = sample[0]
    for i in range(1, len(sample)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]
    
    return best

def cross_over(mother, father, max_depth):
    #combine 2 chromosome into new childs
    child = Chromosome(mother.terminal_set, mother.func_set, mother.depth, None)
    start_m = np.random.randint(len(mother.gen))
    start_f = np.random.randint(len(father.gen))
    end_m = traversal(start_m, mother)
    end_f = traversal(start_f, father)
    child.gen = mother.gen[:start_m] + father.gen[start_f : end_f] + mother.gen[end_m :]
    if child.get_depth() > max_depth and random.random() > 0.2:
        child = Chromosome(mother.terminal_set, mother.func_set, mother.depth)
    return child


def get_best(population):

    best = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]
    
    return best

def get_worst(population):

    worst = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness > worst.fitness:
            worst = population.list[i]
    
    return worst

def replace_worst(population, chromosome):

    worst = get_worst(population)
    if chromosome.fitness < worst.fitness:
        for i in range(len(population.list)):
            if population.list[i].fitness == worst.fitness:
                population.list[i] = chromosome
                break
    return population

def roulette_selecion(population):

    fitness = [chrom.fitness for chrom in population.list]
    order = [x for x in range(len(fitness))]
    order = sorted(order, key=lambda x: fitness[x])
    fs = [fitness[order[i]] for i in range(len(fitness))]
    sum_fs = sum(fs)
    max_fs = max(fs)
    min_fs = min(fs)
    p = random.random()*sum_fs
    t = max_fs + min_fs
    choosen = order[0]
    for i in range(len(fitness)):
        p -= (t - fitness[order[i]])
        if p < 0:
            choosen = order[i]
            break
    return population.list[choosen]

class Population:

    def __init__(self, size, depth, max_depth, num_selected, functions, terminals):

        self.size = size
        self.num_selected = num_selected
        self.max_depth = max_depth
        self.list = self.create_population(self.size, functions, terminals, depth)
        
    def create_population(self, number, functions, terminals, depth):
        population_list = []
        for i in range(number):
            if random.random() < 0.5:
                population_list.append(Chromosome(terminals, functions, depth, 'full'))  
            else:
                population_list.append(Chromosome(terminals, functions, depth, 'grow'))
        return population_list



class Chromosome:

    def __init__(self, terminals, functions, depth, method='full'):

        self.gen = []
        self.depth = depth
        self.func_set = functions
        self.terminal_set = terminals
        self.fitness = None
        if method == 'grow':
            self.grow()
        elif method == 'full':
            self.full()

    def grow(self, level = 0):
        if level == self.depth:
            self.gen.append(random.choice(self.terminal_set))
        else:
            if random.random() > 0.3:
                val = random.choice(self.func_set[2] + self.func_set[1])
                if val in self.func_set[2]:
                    self.gen.append(val)
                    self.grow(level + 1)
                    self.grow(level + 1)
                else:
                    self.gen.append(val)
                    self.grow(level + 1)
            else:
                val = random.choice(self.terminal_set)
                self.gen.append(val)

    def full(self, level = 0):

        if level == self.depth:
            self.gen.append(random.choice(self.terminal_set))
        else:
            val = random.choice(self.func_set[1] + self.func_set[2])
            if val in self.func_set[2]:
                self.gen.append(random.choice(self.func_set[2]))
                self.full(level + 1)
                self.full(level + 1)
            else:
                self.gen.append(random.choice(self.func_set[1]))
                self.full(level + 1)
        
        
    def eval(self, input, poz = 0):
      
        #Function to evaluate current chromosome with given input
        if self.gen[poz] in self.terminal_set:
            return input[int(self.gen[poz][1:])], poz
        elif self.gen[poz] in self.func_set[2]:
            poz_op = poz
            left, poz = self.eval(input, poz + 1)
            right, poz = self.eval(input, poz + 1)
            if self.gen[poz_op] == '+':
                return left + right, poz
            elif self.gen[poz_op] == '-':
                return left - right, poz
            elif self.gen[poz_op] == '^':
                return left ** right, poz
            elif self.gen[poz_op] == '/':
                return left / right, poz
            elif self.gen[poz_op] == '*':
                return left * right, poz
        else:
            poz_op = poz
            left, poz = self.eval(input, poz + 1)
            if self.gen[poz_op] == 'sin':
                return np.sin(left), poz
            elif self.gen[poz_op] == 'cos':
                return np.cos(left), poz
            elif self.gen[poz_op] == 'abs':
                return abs(left), poz
            elif self.gen[poz_op] == 'sqrt':
                return np.sqrt(left), poz
            elif self.gen[poz_op] == 'ln':
                return np.log(left), poz
            elif self.gen[poz_op] == 'tg':
                return np.tan(left), poz
            elif self.gen[poz_op] == 'ctg':
                return 1/np.tan(left), poz

    def evaluate_arg(self, input):
        return self.eval(input)[0]

    def calculate_fitness(self, inputs, outputs):
        diff = 0
        for i in range(len(inputs)):
            try:
                diff += (self.eval(inputs[i])[0] - outputs[i][0])**2
            except RuntimeWarning:
                self.gen = []
                if random.random() > 0.5:
                    self.grow()
                else:
                    self.full()
                self.calculate_fitness(inputs, outputs)
        
        if len(inputs) == 0:
            return 1e9
        self.fitness = diff/(len(inputs))
        return self.fitness

    def __get_depth_aux(self, poz = 0):

        elem = self.gen[poz]
        if elem in self.func_set[2]:
            left, poz = self.__get_depth_aux(poz + 1)
            right, poz = self.__get_depth_aux(poz)

            return 1 + max(left, right), poz
        elif elem in self.func_set[1]:
            left, poz = self.__get_depth_aux(poz + 1)
            return left + 1, poz
        else:
            return 1, poz + 1

    def get_depth(self):
        return self.__get_depth_aux()[0] - 1




class Algorithm:

    def __init__(self, population, iterations, inputs, outputs, feedback = 500):

        self.population = population
        self.iterations = iterations
        self.inputs = inputs
        self.outputs = outputs
        self.epoch_feedback = feedback
    
    def __one_step(self):

        # mother = selection(self.population, self.population.num_selected)
        # father = selection(self.population, self.population.num_selected)
        mother = roulette_selecion(self.population)
        father = roulette_selecion(self.population)
        child = cross_over(mother, father, self.population.max_depth)
        child = mutate(child)
        child.calculate_fitness(self.inputs, self.outputs)
        self.population = replace_worst(self.population, child)

    def train(self):
        for i in range(len(self.population.list)):
            self.population.list[i].calculate_fitness(self.inputs, self.outputs)
        for i in range(self.iterations):
            if i % self.epoch_feedback == 0:
                best_so_far = get_best(self.population)
                print("Predicted function: {0}".format(best_so_far.gen))
                print("Fitness: {0}".format(best_so_far.fitness))
            self.__one_step()
        return get_best(self.population)




VAR_MAX_SIZE = 1
MAX_DEPTH = 15
Terminals = ['x'+str(i) for i in range(VAR_MAX_SIZE)]
Functions = {1: ['sin','cos','ln','tg','sqrt'], 2:['-', '+', '*', '/','^']}

# original Function to be predicted:
def f(x):
    return -(x**3)

# retrieve 1000 inputs and outputs of original function 
X = [[x] for x in np.arange(0, 10, 0.01)]
y = [[f(x[0])] for x in X]

start_time = time.time()
pop = Population(8000, 1, MAX_DEPTH, 800, Functions, Terminals)
algorithm = Algorithm(pop, 8000, X, y, feedback=500)
Optimal = algorithm.train()
preorder_res = Optimal.gen;
print("preorder traverse of predicted function: ")
print(preorder_res);
y_pred = [[Optimal.evaluate_arg(x)] for x in X]

print("--- %s ExecutionSeconds ---" % (time.time() - start_time))
plt.plot(X, y_pred, color='r', dashes=[7, 2], label='Predicted')
plt.plot(X, y, color='b', dashes=[7, 3], label='Expected')
plt.show()
