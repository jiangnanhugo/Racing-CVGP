import time
import numpy as np
from operator import attrgetter
from program import Program
from utils import create_uniform_generations, create_geometric_generations


class ExpandingGeneticProgram(object):
    """
    populations: the current list of programs
    hofs: list of the best programs
    """

    # static variables
    library = None
    gp_helper = None

    def __init__(self, cxpb, mutpb, maxdepth, population_size, tour_size, hof_size, n_generations, nvar):
        """
        cxpb: probability of mate
        mutpb: probability of mutations
        maxdepth: the maxdepth of the tree during mutation
        population_size: the size of the selected populations (at the end of each generation)
        tour_size: the size of the tournament for selection
        hof_size: the size of the best programs retained
        n_generations: the number of generations to be applied over each pool
        nvar: number of variables
        """
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size

        self.n_generations = create_uniform_generations(n_generations, nvar + 1)
        # self.n_generations = create_geometric_generations(n_generations, nvar + 1)

        self.timer_log = []

        self.nvar = nvar

        assert self.library != None
        assert Program.task != None

    def create_init_population(self):
        """
           create the initial population; for every variable, generate a set of generate random equations.
           save them to self.populations, self.hofs.
           look for every token in library, fill in the leaves with constants or inputs.
        """
        self.populations = []
        self.hofs = []

        for vari in range(self.nvar):
            vf = [vari, ]
            free_input_tokens = np.zeros(self.nvar, dtype=np.int32)
            free_input_tokens[vari] = 1
            self.library.set_allowed_input_tokens(free_input_tokens)
            for i, t in enumerate(self.library.tokens):
                if self.library.allowed_tokens[i]:
                    tree = [i]
                    for j in range(t.arity):
                        t_idx = np.random.choice(self.library.tokens_of_arity[0])
                        while self.library.allowed_tokens[t_idx] == 0:
                            t_idx = np.random.choice(self.library.tokens_of_arity[0])
                        tree.append(t_idx)
                    tree = np.array(tree)
                    pr = Program(tree, np.ones(tree.size, dtype=np.int32))
                    pr.vf = vf
                    self.populations.append(pr)
                    new_pr = pr.clone()
                    self.hofs.append(new_pr)

    def run_with_randomized_variable_ordering(self):
        # 1. generate all single variable equations
        self.create_init_population()
        # 2. apply GP
        for round_idx in range(self.nvar + 1):
            for pr in self.populations:
                # 2.1 set the free variables and controlled variables
                # 2.2 re-evaluate the constants and goodness-of-fit.
                # self._set_dataX_allowed_input_tokens(pr.cur_node.total_vf, verbose=True)
                pr.remove_r_evaluate()
                # finds the best constants on control variable data, and then computes  goodness-of-fit.
                _ = pr.r
            for pr in self.hofs:
                pr.remove_r_evaluate()
                _ = pr.r

            # 2.3  do n generation of GP,
            for it in range(self.n_generations[round_idx]):
                print('++++++++++++ ROUND {} ITERATION {} ++++++++++++'.format(round_idx, it))
                self.one_generation(verbose=True)
            # 2.4 find the best set of fitted expressions

            self.scheduling_vf(min(self.nvar, round_idx + 1))
            self.update_population()
            print_prs(self.populations)
            # 2.5 freeze tokens in the expressions
            for i, pr in enumerate(self.populations):
                # evaluate r again, just incase it has not been evaluated.
                this_r = pr.r
                if len(pr.const_pos) == 0 or pr.num_changing_consts == 0:
                    # only expand at those constant node. if there are no constant node,then we are done
                    # if we do not want num_changing_consts, then we also quit.
                    print('there are no constant node. we are done...')
                else:
                    pr.remove_r_evaluate()
                    this_r = pr.r
                # whether you get very different value for different constant.
                pr.freeze_equation()

    def scheduling_vf(self, num_of_free_variables):
        print("scheduling step.....")
        for pr in self.populations:
            while len(pr.vf) < num_of_free_variables:
                print("{} from: {}".format(pr, pr.vf), end="\t")
                pr.pick_new_random_vf()
                print("to {}".format(pr.vf))

    def one_generation(self, verbose=False):
        """
        One step of the genetic algorithm.
        This wraps selection, mutation, crossover and hall of fame computation
        over all the individuals in the population for this epoch/step.
        """
        t1 = time.perf_counter()

        # Select the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size)

        # Vary the pool of individuals
        offspring = self._var_and(offspring)

        # Replace the current population by the offspring
        self.populations = offspring + self.hofs

        # Update hall of fame
        self.update_hof()
        if verbose:
            print("after update hof=")
            print_prs(self.hofs)

        timer = time.perf_counter() - t1
        self.timer_log.append(timer)

    def selectTournament(self, population_size, tour_size, randomized=False):
        """evaluate on full data for tournament"""
        offspring = []
        for pp in range(population_size):
            spr = np.random.choice(self.populations, tour_size)
            maxspr = max(spr, key=attrgetter('r'))
            if randomized:
                maxspr = np.random.choice(spr)
            maxspri = maxspr.clone()
            offspring.append(maxspri)
        return offspring

    def _var_and(self, offspring, crosscover_with_same_vf=0.5):
        """Apply mutation AND crossover to each individual in a population, given a constant probability."""
        # Apply mutation on the offspring
        # sorted(offspring, reverse=True, key=attrgetter('r'))
        # print_prs(offspring)
        for i in range(len(offspring)):
            if np.random.random() < self.mutpb:
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)
        # sorted(offspring, reverse=True, key=attrgetter('r'))
        # print_prs(offspring)
        # Apply crossover on the offspring
        np.random.shuffle(offspring)
        if np.random.random() < crosscover_with_same_vf:

            for i in range(1, len(offspring), 2):
                if np.random.random() < self.cxpb:
                    self.gp_helper.mate(offspring[i - 1], offspring[i])
        else:
            used = set()
            for i in range(len(offspring)):
                selected = None
                used.add(i)
                for j in range(i + 1, len(offspring)):
                    if offspring[i].vf == offspring[j].vf and j not in used:
                        selected = j
                        used.add(j)
                        break
                if selected is None:
                    if len(used) < len(offspring):
                        selected = np.random.choice([i for i in range(len(offspring)) if i not in used])

                if np.random.random() < self.cxpb and selected is not None:
                    # print(offspring[i], offspring[i].vf, " ||||||||||| ", offspring[selected], offspring[selected].vf)
                    self.gp_helper.mate(offspring[i], offspring[selected])

        # sorted(offspring, reverse=True, key=attrgetter('r'))
        # print_prs(offspring)
        return offspring

    def update_hof(self):
        """update the set of Hall of Fame"""
        new_hof = sorted(self.populations, reverse=True, key=attrgetter('r'))
        self.hofs = [new_hof[i].clone() for i in range(self.hof_size)]

    def update_population(self):
        """update the population. sort by fitness score and cut by population_size."""
        new_population = sorted(self.populations, reverse=True, key=attrgetter('r'))
        self.populations = [new_population[i].clone() for i in range(self.population_size)]

    def print_final_hofs(self):
        Program.task.set_allowed_inputs(np.ones(self.nvar, dtype=np.int32))
        for pr in self.hofs:
            print("\t", pr.__getstate__())
            pr.task.rand_draw_X_non_fixed()
            pr.print_expression()
            print('\tvalidate r=', pr.task.reward_function(pr))
            pr.task.print_reward_function_all_metrics(pr)


def print_prs(prs):
    new_prs = sorted(prs, reverse=True, key=attrgetter('vf', 'r'))
    for pr in new_prs:
        print('        ' + str(pr.__getstate__()), end="\t")
        pr.print_expression()
    print("")
