import train
import numpy as np

from aco import ACO, Graph

pos_dict = {}
TAG = 32

def calc_cost(words, tagg):
    phi, lexicon, bigram, tag = train.train()
    weight = []

    for i, pos in enumerate(tag):
        pos_dict[i] = pos

    for i in range(len(words)):
        temp = []
        if i == 0:
            a = []
            for j in range(len(phi)):
                if lexicon[words[i]][j] != 0 and phi[j] != 0:
                    a.append(round(1/lexicon[words[i]][j] ** 1/phi[j], 4))
                else:
                    a.append(2000)
            temp.append(a)
            for i in range(1, tagg):
                temp.append([float('inf') for j in range(tagg)])
        else:
            for j in range(tagg):
                temp.append(mult_list(lexicon[words[i]], bigram[j]))

        weight.append(temp)

    return weight


def mult_list(a, b):
    res = []
    for i in range(len(a)):
        if a[i] != 0 and b[i] != 0:
            res.append(round(1/a[i] ** 1/b[i], 4))
        else:
            res.append(2000)
    return res


def main():
    text = 'در آینده به همکاری با همسایگان شتاب بیشتر خواهیم داد'
    text = text.strip()
    text = text.lower()
    words = text.split(' ')
    rank = len(words)
    cost_matrix = calc_cost(words, TAG)
    aco = ACO(ant_count=100, generations=100, alpha=.9, beta=.8, rho=1, q=10, strategy=0)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    print('cost: {} \npath: {}'.format(cost, translate_path(words, path)))


def translate_path(words, path):
    res = {}
    for i in range(len(path)):
        print(words[i] + ' : ' + pos_dict[path[i]])
        res[words[i]] = pos_dict[path[i]]
    return res


if __name__ == '__main__':
    main()