from model import Model
from train import train


class Viterbi:
    def __init__(self, model : Model):
        # self.pi = pi
        # self.lexicon = lexicon
        # self.bigram = bigram
        # self.tag = tag
        self.model = model

    def second_score(self, text):
        index_res = []
        score_res = []
        prev_score = []

        for i in range(len(self.model.pi)):
            prev_score.append(self.model.pi[i] * self.model.lexicon.get(text[0])[i])
            score_res.append(prev_score)

        for word in text[1:]:
            score2 = []
            index = []
            for j in range(len(self.model.tag)):
                score1 = []
                for i in range(len(self.model.tag)):
                    score1.append(prev_score[i] * self.model.bigram[j][i])
                m = max(score1)
                index.append(score1.index(m))
                score2.append(m * self.model.lexicon.get(word)[j])
            prev_score = score2
            index_res.append(index)
            score_res.append(score2)
        return score_res, index_res

    def backward(self, score, index1):
        res = []
        i = self.argmax(score[-1])
        res.append(i)
        for j in range(len(index1) - 1, -1, -1):
            i = index1[j][i]
            res.append(i)
        return list(reversed(res))

    def argmax(self, l):
        m = max(l)
        return l.index(m)

    def solve(self, text):
        text = text.split(' ')
        score1, index = self.second_score(text)
        return self.backward(score1, index)


def translate_path(words, path):
    for i in range(len(path)):
        print(words[i] + ' : ' + tag[path[i]])


if __name__ == '__main__':
    pi, emission, transition, tag, tests = train()
    model = Model(emission, transition, pi, tag)
    viterbi = Viterbi(model)
    while True:
        text = input()
        path = viterbi.solve(text)
        translate_path(text.split(' '), path)
