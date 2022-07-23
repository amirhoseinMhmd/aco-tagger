import random
import numpy as np


class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = self.calc_pheromone(rank, cost_matrix)

    def calc_pheromone(self, rank, cost):
        pheromone = []
        for m in range(len(cost)):
            matrix = np.array(cost[m])
            shape = matrix.shape
            # if m == 0:
            #     pheromone.append([1 / (rank * rank) for j in range(shape[0])])
            # else:
            pheromone.append([[1 / (rank * rank + 1) for j in range(shape[1])] for i in range(shape[0])])
        return pheromone


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] = [p * self.rho for p in graph.pheromone[i][j]]
                for ant in ants:
                    graph.pheromone[ant.state - 1][i][j] += ant.pheromone_delta[ant.state - 1][i][j]

    def solve(self, graph: Graph):
        best_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(len(graph.matrix[ant.state])):
                    ant._select_next()
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.rank)]
        self.eta = self.calc_eta(graph)
        start = 0
        self.current = start
        self.state = 0

    def increase_state(self):
        self.state = self.state + 1

    def set_cost(self, cost):
        self.total_cost += cost

    def calc_eta(self, graph):
        eta = []
        for j in range(len(graph.matrix)):
            shape = np.array(graph.matrix[j]).shape
            temp = []
            if len(shape) == 1:
                temp.append([1 / m for m in graph.matrix[j]])
            else:
                for i in graph.matrix[j]:
                    temp.append([1 / m for m in i])

            eta.append(temp)
        return eta

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.state][self.current][i] ** self.colony.alpha * \
                           self.eta[self.state][self.current][
                               i] ** self.colony.beta
        probabilities = [0 for i in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)
                if denominator == 0:
                    probabilities[i] = 0
                else:
                    probabilities[i] = self.graph.pheromone[self.state][self.current][i] ** self.colony.alpha * \
                                       self.eta[self.state][self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass

        selected = 0
        # if self.state >= 0:
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.state][self.current][selected]
        self.current = selected
        self.allowed = list(range(3))
        self.increase_state()

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)] for k in
                                range(len(self.graph.pheromone))]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:
                self.pheromone_delta[i][j] = [self.colony.Q / i for i in self.graph.matrix[i][j]]
            else:
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
