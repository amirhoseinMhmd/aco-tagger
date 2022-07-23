from math import log10 as lg
import pandas as pd

from aco import ACO, Graph


def load_lexicon():
    df = pd.read_csv('lexicon')
    d = {}

    for index, row in df.iterrows():
        d[row['word']] = row.values.tolist()[1:]
    return d


def load_bigram():
    return pd.read_csv('bigram').to_numpy()


def load_pi():
    return pd.read_csv('pi').values.tolist()[0]


def calc_cost(words, tagg):
    lexicon = load_lexicon()
    bigram = load_bigram()
    phi = load_pi()
    weight = []

    for i in range(len(words)):
        if i >= len(words):
            break
        temp = []
        if i == 0:
            a = []
            for j in range(len(phi)):
                a.append(round(-lg(lexicon[words[i]][j] * phi[j]), 4))
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
        res.append(round(-lg(a[i] * b[i]), 4))
    return res


def main():
    # text = 'ali is dead'
    text = 'is ali dead'
    words = text.split(' ')
    rank = len(words)
    cost_matrix = calc_cost(words, 3)
    aco = ACO(10, 100, 0.90, 0.5, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    print('cost: {} \npath: {}'.format(cost, translate_path(words, path)))

pos_dict={
    0 : ' NOUN',
    1 : 'VERB',
    2 : 'ARTICLE',
}

def translate_path(words, path):
    res = {}
    for i in range(len(path)):
        res[words[i]] = pos_dict[path[i]]
    return res

if __name__ == '__main__':
    main()

