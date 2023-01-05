class Viterbi :
    def __init__(self, pi, lexicon, bigram, tag):
        self.pi = pi
        self.lexicon = lexicon
        self.bigram = bigram
        self.tag = tag
    def first_score(self,  word):
        score = []
        for i in range(len(self.pi)):
            score.append(self.pi[i] * self.lexicon.get(word)[i])
        return score


    def second_score(self,score, text):
        index_res = []
        score_res = []
        for word in text:
            if text.index(word) == 0:
                continue
            score2 = []
            index = []
            for j in range(len(self.bigram)):
                score1 = []
                for i in range(len(self.bigram)):
                    score1.append(score[i] * self.bigram[j][i])
                m = max(score1)
                index.append(score1.index(m))
                score2.append(round(m * self.lexicon.get(word)[j], 5))

            score = score2
            index_res.append(index)
            score_res.append(score2)
        return score_res, index_res

    def backward(self, score, index1):
        # a = max(score[len(score)-1])
        i = self.argmax(score[-1])
        print(self.tag[i])
        for j in range(len(score) - 1, -1, -1):
            i = index1[j][i]
            print(self.tag[i])

    def argmax(self , l):
        m = max(l)
        return l.index(m)

# هوا امروز برفی است .



    def solve(self , text):
        text = text.split(' ')
        # pi, lexicon, bigram, tag = train()
        score = self.first_score(text[0])
        # print(score)
        score1, index = self.second_score(score, text)
        # print(score1)
        # print(index)
        self.backward(score1, index)

