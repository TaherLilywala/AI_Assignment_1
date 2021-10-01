from CNF_Creator import *
import time
import numpy as np
import math
import matplotlib.pyplot as plt

def pop_fitness(sentence, population):
    fitness = np.zeros(len(population))

    for i in range(len(population)):
        model = population[i]
        fit = model_fitness(sentence, model)
        fitness[i] = fit
        
    sorted_population = [x for _, x in sorted(zip(fitness, population), key = lambda x:x[0], reverse = True)]
    fitness = sorted(fitness, reverse=True)

    return [sorted_population, fitness]

def model_fitness(sentence, model):
    satisfied_clauses = 0
    
    for clause in sentence:
        for literal in clause:                
            if ((literal < 0) ^ (model[abs(literal)-1])):
                satisfied_clauses += 1
                break
    
    return ((satisfied_clauses/len(sentence))*100)

def mating(parent1, parent2):
    i = np.random.randint(0,len(parent1))
    return (np.append(parent1[:i], parent2[i:]), np.append(parent2[:i], parent1[i:]))

def mutate(child, mutation_probabilty):
    i = np.random.randint(0,len(child))
    rand_num = np.random.rand()
    
    if mutation_probabilty > rand_num:
        child[i] = 1-child[i]   # Flipping a random bit

    return child

def gen_algo(sentence, population_size, model_size, verbose):

    start_time = time.time()
    elapsed_time = 0

    accepted = False
    generation = 1
    population = [(np.random.randint(low = 0, high=2, size=model_size)) for i in range(population_size)]
    mutation_probabilty = 1

    early_stop_count = 1
    max_fitness = 0
    
    while(True):

        [population, fitness] = pop_fitness(sentence, population)
        best_fitness = fitness[0]

        if(best_fitness == max_fitness):
            if(early_stop_count==1):
                early_stop_time = time.time()
            early_stop_count += 1
            if(time.time()-early_stop_time >= 30):
                print("Fitness Plateaued, Early Stopping")
                print("Sentence: {}  \nObtained String: {}  \nFitness Achieved: {}  \nGenerations:{}".format(sentence, population[0], max_fitness, generation))
                print("Time Elapsed: {}".format(time.time()-start_time))
                elapsed_time = time.time()-start_time
                return (max_fitness, elapsed_time)
        else:
            early_stop_count = 1

        if(best_fitness > max_fitness):
            max_fitness = best_fitness
        
        if(verbose==1):
            print("Generation: {}   Fitness Achieved: {}".format(generation, best_fitness))

        if(best_fitness == 100):
            print("Maximum Fitness Reached")
            print("Sentence: {}  \nObtained String: {}  \nFitness Achieved: {}  \nGenerations:{}".format(sentence, population[0], max_fitness, generation))
            print("Time Elapsed: {}".format(time.time()-start_time))
            elapsed_time = time.time()-start_time
            return (max_fitness, elapsed_time)

        new_generation = []

        if (population_size < 15):
            if((generation % 100) == 0):
                population_size += 1

        num_elite = int(0.2*population_size)


        new_generation.extend(population[:num_elite])   # Sending top 20% to next generation right away

        num_mating = int(0.5*population_size)
        elite_ratio = int(0.9*population_size)

        for _ in range(num_mating):
            parents = random.choices(population= population[:elite_ratio], weights= fitness[:elite_ratio], k= 2)
            parent1 = parents[0]
            parent2 = parents[1]
            (child1, child2) = mating(parent1, parent2)
            mutation_probabilty = math.exp(-generation/(100*early_stop_count))
            child1 = mutate(child1, mutation_probabilty)
            child2 = mutate(child2, mutation_probabilty)
            new_generation.append(child1)
            new_generation.append(child2)

        population = new_generation
        generation = generation+1

        elapsed_time = time.time()-start_time

        if(elapsed_time>=45):
            print("Timed Out")
            print("Sentence: {}  \nObtained String: {}  \nFitness Achieved: {}  \nGenerations:{}".format(sentence, population[0], max_fitness, generation))
            print("Time Elapsed: {}".format(time.time()-start_time))
            return (max_fitness, elapsed_time)

def main():

    n = 50
    cnfC = CNF_Creator(n) # n is number of symbols in the 3-CNF sentence

    sentence = cnfC.CreateRandomSentence(m) # m is number of clauses in the 3-CNF sentence
    # print('Random sentence : ',sentence)

    # sentence = cnfC.ReadCNFfromCSVfile()
    # print('\nSentence from CSV file : ',sentence)

    model_size = n  # Number of symbols in 3-CNF sentence
    population_size = 5 # Number of members in the model
    max_generations = 10000

    return gen_algo(sentence, population_size, model_size, verbose)

if __name__=='__main__':
    
    m = 100

    fitness_all = []
    time_all = []
    m_all = []

    verbose = 0 # Set to 1 to print values

    for _ in range(11):
        fitness_val = []
        time_val = []
        for j in range(10):
            print("Run: \tm={} \tIteration={}".format(m,j))
            (fitness_iter, time_iter) = main()
            print("\n")
            fitness_val.append(fitness_iter)
            time_val.append(time_iter)
        fitness_all.append(np.average(fitness_val))
        time_all.append(np.average(time_val))
        m_all.append(m)
        m+=20


    plt.plot(m_all, fitness_all)

    plt.xlabel('Value of m')
    plt.ylabel('Average Fitness')
    
    plt.title('Average Fitness vs m')
    
    plt.show()


    plt.plot(m_all, time_all)

    plt.xlabel('Value of m')
    plt.ylabel('Average Time')
    
    plt.title('Average Time vs m')
    
    plt.show()