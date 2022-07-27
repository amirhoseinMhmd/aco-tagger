from math import log10 as lg
import pandas as pd
import train
import numpy as np

from aco import ACO, Graph

pos_dict={}
TAG = 32

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
    # lexicon = load_lexicon()
    # bigram = load_bigram()
    # phi = load_pi()
    # tag = ['NOUN','VERB', 'ART']
    phi, lexicon, bigram ,tag = train.train()
    weight = []

    for i, pos in enumerate(tag):
        pos_dict[i] = pos

    for i in range(len(words)):
        # if i >= len(words):
        #     break
        temp = []
        if i == 0:
            a = []
            for j in range(len(phi)):
                x = lexicon[words[i]][j] * phi[j]
                if x!=0:
                    a.append(round(-lg(x), 4))
                else:
                    a.append(20)
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
        x = a[i] * b[i]
        if x!= 0:
            res.append(round(-lg(a[i] * b[i]), 4))
        else:
            res.append(20)
    return res

# کلیه/DET معادن/N_PL کشور/N_SING قابل/ADJ بهره‌برداری/N_SING نبوده/V_PP و/CON استخراج/N_SING آن‌ها/PRO مقرون‌به‌صرفه/ADJ نیست/ ./DELM
def main():
    # text = 'ali is dead'
    text = 'کلیه معادن کشور قابل بهره‌برداری نبوده و استخراج آن‌ها مقرون‌به‌صرفه نیست'
    text = text.strip()
    text = text.lower()
    words = text.split(' ')
    rank = len(words)
    cost_matrix = calc_cost(words, TAG)
    aco =  ACO(ant_count=1000, generations=10, alpha=.9, beta=.5, rho=.95, q=5, strategy=2)
    #      ACO(10, 100, 1.0, 10.0, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    print('cost: {} \npath: {}'.format(cost, translate_path(words, path)))


def translate_path(words, path):
    res = {}
    for i in range(len(path)):
        res[words[i]] = pos_dict[path[i]]
    return res

if __name__ == '__main__':
    main()