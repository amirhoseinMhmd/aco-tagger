from collections import defaultdict
import numpy as np

TAG = 32

def train():
    f = open('output.txt', 'r', encoding='utf-8')
    text = f.read()
    f.close()
    tag = []
    word_count = defaultdict(lambda: np.ones(shape=(TAG)))
    transition = np.zeros(shape=(TAG, TAG))
    unigram = np.zeros(shape=(TAG))
    start_count = np.zeros(shape=(TAG))
    lines = text.split('\n')

    for l in lines:
        l = l.strip()
        words = l.split(' ')
        if len(words) == 1:
            continue
        start = words[0].split('/')
        if not tag.__contains__(start[1]):
            tag.append(start[1])
        start_count[tag.index(start[1])] += 1

        for i in range(len(words)):
            w = words[i]
            part = w.split('/')
            if len(part) != 2:
                continue
            part[0] = part[0].lower()
            if not tag.__contains__(part[1]):
                tag.append(part[1])
            word_count[part[0]][tag.index(part[1])] += 1
            unigram[tag.index(part[1])] += 1
            if i != 0:
                prev = words[i - 1]
                prev_part = prev.split('/')
                transition[tag.index(part[1])][tag.index(prev_part[1])] += 1
    start_count /= len(lines)
    for i in range(len(tag)):
        transition[i] = transition[i] / unigram[i]

    emission = defaultdict()
    for k in word_count:
        s = sum(word_count[k])
        emission[k] = word_count[k] / s

    return start_count, emission, transition, tag


if __name__ == '__main__':
    train()
