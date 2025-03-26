import random
import networkx as nx

def genetic_algorithm(graph, misinformation_spreaders, num_seeds, population_size, generations, mutation_rate):
    def fitness(solution):
        # Fitness function to evaluate the spread of truth vs. misinformation
        return sum([1 for node in solution if node not in misinformation_spreaders])

    def initialize_population():
        population = []
        for _ in range(population_size):
            individual = random.sample(list(graph.nodes()), num_seeds)
            population.append(individual)
        return population

    def selection(population):
        population_fitness = [(individual, fitness(individual)) for individual in population]
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        return [individual for individual, fit in population_fitness[:population_size//2]]

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, num_seeds-1)
        child1 = parent1[:crossover_point] + [node for node in parent2 if node not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [node for node in parent1 if node not in parent2[:crossover_point]]
        return child1[:num_seeds], child2[:num_seeds]

    def mutate(individual):
        if random.random() < mutation_rate:
            mutate_point = random.randint(0, num_seeds-1)
            new_gene = random.choice(list(graph.nodes()))
            while new_gene in individual:
                new_gene = random.choice(list(graph.nodes()))
            individual[mutate_point] = new_gene
        return individual

    # Initialize population
    population = initialize_population()

    for generation in range(generations):
        # Selection
        population = selection(population)
        # Crossover
        new_population = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                child1, child2 = crossover(population[i], population[i+1])
                new_population.append(child1)
                new_population.append(child2)
        # Mutation
        population = [mutate(individual) for individual in new_population]

    # Final selection of the best individual
    best_individual = max(population, key=fitness)
    return best_individual

def community_detection(graph, misinformation_spreaders, num_seeds):
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    seed_nodes = []

    for community in communities:
        if len(seed_nodes) >= num_seeds:
            break
        community_nodes = list(community)
        for node in community_nodes:
            if node not in misinformation_spreaders and len(seed_nodes) < num_seeds:
                seed_nodes.append(node)
    
    return seed_nodes
