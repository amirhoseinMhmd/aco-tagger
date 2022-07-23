from math import log10 as lg
import pandas as pa

from aco import ACO, Graph
import time


# def distance(city1: dict, city2: dict):
#     return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


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


def calc_cost(words, tagg):
    lexicon = load_lexicon()
    bigram = load_bigram()
    phi = load_pi()
    weight = []

    for i in range(len(words)):
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
                a = list(lexicon[words[i]][j] * bigram[j])
                a = [round(-lg(x), 4) for x in a]
                temp.append(a)

        weight.append(temp)

    return weight


def main():
    # text = input('please enter your text : ')
    text = 'ali is dead'
    # text = 'is ali dead'
    words = text.split(' ')
    # cities = []
    # points = []
    # with open('zi929.tsp') as f:
    #     for line in f.readlines():
    #         city = line.split(' ')
    #         cities.append(dict(index=int(city[0]), x=float(city[1]), y=float(city[2])))
    #         points.append((float(city[1]), float(city[2])))
    # cost_matrix = []
    rank = len(words)
    # for i in range(rank):
    #     row = []
    #     for j in range(rank):
    #         row.append(distance(cities[i], cities[j]))
    #     cost_matrix.append(row)
    cost_matrix = calc_cost(words, 3)
    aco = ACO(10, 100, 0.90, 0.5, 0.5, 10, 2)
    graph = Graph(cost_matrix, rank)
    path, cost = aco.solve(graph)
    make_csv('{}, path: {}'.format(cost, path), 'result.csv')
    print('cost: {}, path: {}'.format(cost, path))


def make_csv(data, file_name):
    with open(file_name, 'a') as f:
        f.write(data)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    for i in range( 1):
        start_millis = int(round(time.time() * 1000))
        main()
        time.time()
        final_millis = int(round(time.time()) * 1000)
        t = final_millis - start_millis
        if t < 0:
            t *= -1
        fi = open("times.csv", 'a')
        fi.write(str(t))
        fi.write('\n')
        fi.close()
