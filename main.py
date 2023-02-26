from train import train
from aco import ACO, Graph
import datetime as dt
from math import log10
from viterbi import Viterbi

tag_dict = {}
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
        tag_dict[i] = pos

    for i in range(len(words)):
        if not words[i] in model.lexicon:
            raise Exception('The word {} not contained in the dictionary'.format(words[i]))
        temp = []
        if i == 0:
            a = []
            for j in range(len(model.phi)):
                if model.lexicon[words[i]][j] != 0 and model.phi[j] != 0:
                    # a.append(round((1 / model.lexicon[words[i]][j]) ** (1 / model.phi[j]), 4))
                    a.append(round((1 / model.lexicon[words[i]][j]) ** (-log10(model.phi[j])), 4))
                else:
                    a.append('inf')
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
            # res.append(round((1 / a[i]) ** (1 / b[i]), 4))
            res.append(round((1 / a[i]) ** (-log10(b[i])), 4))
        else:
            res.append('inf')
    return res


def create_text(sentences: str):
    sentences = sentences.split(' ')
    text = ''
    tag = []
    for s in sentences:
        part = s.split('/')
        text += part[0] + ' '
        tag.append(part[1])
    return text, tag


def bleu(a, b):
    e = 0
    for i in range(len(a)):
        if a[i] == tag_dict[b[i]]:
            e += 1
    return e / len(a)


def main():

    ant_count = 3
    generations = 20
    alpha = .9
    beta = .9
    rho = .95
    q = 10
    strategy = 2
    output = str(dt.datetime.now()) + str(ant_count) + "-" + str(generations) + "-" + str(alpha) + "-" + str(
        beta) + "-" + str(rho) + "-" + str(
        q) + "-" + str(strategy)
    pi, emission, transition, tags, tests = train()
    model = Model(emission, transition, pi, tags)
    aco_total_error = 0.0
    viterbi_total_error = 0.0
    index = 1
    for test in tests:
        test = test.strip()
        text, tag = create_text(test)
        text = text.strip()
        text = text.lower()

        words = text.split(' ')
        rank = len(words)

        try:
            cost_matrix = calc_cost(model, words, TAG)
            aco = ACO(ant_count=ant_count, generations=generations, alpha=alpha, beta=beta, rho=rho, q=q,
                      strategy=strategy)
            graph = Graph(cost_matrix, rank)

            viterbi = Viterbi(pi, emission, transition, tags)

            aco_path, cost = aco.solve(graph)
            aco_error = bleu(tag, aco_path)
            aco_total_error += aco_error

            viterbi_path = viterbi.solve(text)
            viterbi_error = bleu(tag, viterbi_path)
            viterbi_total_error += viterbi_error

            # f = open(output, 'a', encoding='utf-8')
            print('---------------------------------------------------------')
            print(rank)
            print(str(tests.index(test)) + '/' + str(len(tests)) )
            print('aco accuracy percentage : {}'.format(aco_error * 100))
            print('viterbi accuracy percentage : {}'.format(viterbi_error * 100) )
            print('')
            print('aco average accuracy {} '.format(100 * aco_total_error / index) )
            print('viterbi average accuracy {} '.format(100 * viterbi_total_error / index) )
            # f.write('duration time : {}'.format(time.time() - t) + '\n')
            # f.close()
            index += 1
        except Exception as e:
            print(e)

    # f = open(output, 'a', encoding='utf-8')
    # f.write('---------------------------------------------------------\n')
    # f.write('average accuracy {} '.format(100 * aco_total_error / len(tests)) + '\n')
    # f.write('average accuracy {} '.format(100 * aco_total_error / len(tests)) + '\n')
    # f.close()


def translate_path(words, path):
    for i in range(len(path)):
        print(words[i] + ' : ' + tag_dict[path[i]])


if __name__ == '__main__':
    main()
