import time

from train import train
import numpy as np

from aco import ACO, Graph

pos_dict = {}
TAG = 32


class Model(object):
    def __init__(self, lexicon, bigram, phi, tag):
        self.lexicon = lexicon
        self.bigram = bigram
        self.phi = phi
        self.tag = tag


def calc_cost(model: Model, words, tagg):
    weight = []
    for i, pos in enumerate(model.tag):
        pos_dict[i] = pos

    for i in range(len(words)):
        temp = []
        if i == 0:
            a = []
            for j in range(len(model.phi)):
                if model.lexicon[words[i]][j] != 0 and model.phi[j] != 0:
                    a.append(round((1 / model.lexicon[words[i]][j]) ** (1 / model.phi[j]), 4))
                else:
                    a.append(200000)
            temp.append(a)
            for i in range(1, tagg):
                temp.append([float('inf') for j in range(tagg)])
        else:
            for j in range(tagg):
                temp.append(mult_list(model.lexicon[words[i]], model.bigram[j]))

        weight.append(temp)

    return weight


def mult_list(a, b):
    res = []
    for i in range(len(a)):
        if a[i] != 0 and b[i] != 0:
            res.append(round((1 / a[i]) ** (1 / b[i]), 4))
        else:
            res.append(200000)
    return res


# در آینده به همکاری با همسایگان شتاب بیشتر خواهیم داد .
def main():
    pi, emission, transition, tag = train()
    model = Model(emission, transition, pi, tag)
    while True:
        text = input()
        text = text.strip()
        text = text.lower()
        words = text.split(' ')
        rank = len(words)
        cost_matrix = calc_cost(model, words, TAG)
        aco = ACO(ant_count=150, generations=3, alpha=.9, beta=.55, rho=1, q=10, strategy=0)
        graph = Graph(cost_matrix, rank)
        t = time.time()
        print('start solving graph...')
        path, cost = aco.solve(graph)
        print(time.time() - t)
        print('cost: {} \npath: {}'.format(cost, translate_path(words, path)))


def translate_path(words, path):
    res = {}
    for i in range(len(path)):
        print(words[i] + ' : ' + pos_dict[path[i]])
        res[words[i]] = pos_dict[path[i]]
    return res


if __name__ == '__main__':
    main()
