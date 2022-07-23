from math import log10 as lg
import numpy as np
import pandas as pa


def load_lexicon():
    df = pa.read_csv('lexicon')
    d = {}

    for index, row in df.iterrows():
        d[row['word']] = row.values.tolist()[1:]
    return d

def load_bigram():
    return pa.read_csv('bigram').to_numpy()

def load_pi():
    return pa.read_csv('pi').values.tolist()[0]

def calc_cost(words):

    tagg = 3
    lexicon = load_lexicon()
    bigram = load_bigram()
    phi = load_pi()
    weight = []

    for i in range(len(words)):
        temp = []
        if i == 0:
            for j in range(len(phi)):
                temp.append(round(-lg(lexicon[words[i]][j] * phi[j]), 4))

        else:
            for j in range(tagg):
                a = list(lexicon[words[i]][j] * bigram[j])
                a = [round(-lg(x), 4) for x in a]
                temp.append(a)

        weight.append(temp)

    return weight


if __name__ == '__main__':
    text = 'ali is dead'
    words = text.split(' ')

    weight = calc_cost(words)
    for w in weight:
        print(np.array(w))
