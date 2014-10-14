import random
import numpy as np
from bisect import bisect


# Mating function with crossover point
def mate_point_cross(chrom1, chrom2):
	""" 
	Mating function with crossover point
	Parameters:
	-----------
	chrom1: array-like (list, tuple)
			First chromosome
	chrom2: array-like (list, tuple)
			Second chromosome
	Returns:
	-------
	offspring1: tuple
			First offspring
	offspring2: tuple
			Second offspring
	"""
	l = len(chrom1)

	offspring1 = [True] * l
	crossover_point = np.random.randint(1, l-1) + 1

	offspring1[0: crossover_point] = chrom1[0: crossover_point]
	offspring1[crossover_point:] = chrom2[crossover_point :]

	offspring2 = [True] * l
	crossover_point = np.random.randint(1, l-1) + 1

	offspring2[0: crossover_point] = chrom1[0: crossover_point]
	offspring2[crossover_point:] = chrom2[crossover_point :]

	return tuple(offspring1), tuple(offspring2)


def mate_random_cross(chrom1, chrom2):
	""" 
	Mating function with random gene exchange
	Parameters:
	-----------
	chrom1: array-like (list, tuple)
			First chromosome
	chrom2: array-like (list, tuple)
			Second chromosome
	Returns:
	-------
	offspring1: tuple
			First offspring
	offspring2: tuple
			Second offspring
	"""

	l = len(chrom1)

	offspring1 = [True] * l
	offspring2 = [True] * l

	for i in range(l):
		if random.getrandbits(1) == 0:
			offspring1[i] = chrom1[i]
		else:
			offspring1[i] = chrom2[i]

		if random.getrandbits(1) == 0:
			offspring2[i] = chrom1[i]
		else:
			offspring2[i] = chrom2[i]
	return tuple(offspring1), tuple(offspring2)

def mutate_chrom(chrom, mutate_prob, mutate_gene_func):
	"""
	Mutates a chromosome

	Parameters
	----------
	chrom: array-like (tuple)
		Chromosome to be mutated
	mutate_prob: float [0, 1]
		Probability of mutating each gene
	mutate_gene_func: function, takes two arguments: gene and position of gene
		The function that mutates a gene. Genes in different locations can
		have different mutation functions

	Returns:
	--------
	mutated_chrom: tuple
		The mutated chromosome
	"""
	mutated_chrom = np.array(chrom)
	for pos, gene in enumerate(chrom):
		if random.random() < mutate_prob:
			new_gene = mutate_gene_func(gene, pos)
			mutated_chrom[pos] = new_gene
	return tuple(mutated_chrom)

def mutate_bool(gene, pos):
	""" Boolean mutation """
	gene = bool(gene)
	return np.invert(gene)






# Sample from a list according to a probability weights
# See http://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability

def cdf(weights):
	total=sum(weights) * 1.
	result=[]
	cumsum=0
	for w in weights:
		cumsum += w 
		result.append(cumsum/total)
	return result

def choice(population, cdf_vals):
	"""
	Returns a random element of population sampled according
	to the weights cdf_vals (produced by the func cdf)
	Inputs
	------
	population: list, a list with objects to be sampled from
	cdf_vals: list/array with cdfs (produced by the func cdf)
	Returns
	-------
	An element from the list population
	"""
	assert len(population) == len(cdf_vals)
	x = random.random()
	idx = bisect(cdf_vals,x)
	return population[idx]


class OptimizeGA(object):
	def __init__(self, initial_pop, fit_func, mutate_prob, mutate_gene_func, retain_rate):
		self.populations = [initial_pop]
		self.fit_func = fit_func
		self.mutate_prob = mutate_prob
		self.mutate_gene_func = mutate_gene_func
		self.retain_rate = retain_rate
	def compute_pop_fitness(self, pop):
		fit = []
		for chrom in pop:
			fit.append(self.fit_func(chrom))	
		return tuple(fit)

	def produce_new_population(self, parent_pop):
		new_pop = []
		fitness = self.compute_pop_fitness(parent_pop)
		cdf_vals = cdf(fitness)
		indx = np.argsort(fitness)
		cutoff = int(len(parent_pop) * self.retain_rate)
		best_indx = indx[-cutoff:]

		for i in best_indx:
			new_pop.append(parent_pop[i])
		
		while len(new_pop) < len(parent_pop):
			chrom1 = choice(parent_pop, cdf_vals)
			chrom2 = choice(parent_pop, cdf_vals)
			off1, off2 = mate_point_cross(chrom1, chrom2)
			off1 = mutate_chrom(off1, self.mutate_prob, self.mutate_gene_func)
			off2 = mutate_chrom(off2, self.mutate_prob, self.mutate_gene_func)
			new_pop.append(off1)
			new_pop.append(off2)
		return tuple(new_pop)

	def produce_generations(self, num_gens):
		for i in range(num_gens):
			old_pop = self.populations[-1]
			new_pop = self.produce_new_population(old_pop)
			self.populations.append(new_pop)

