
import numpy as np
from random import shuffle

from nltk.translate.bleu_score import corpus_bleu

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from progressbar import progressbar

class cuboidDataset():
    def __init__(self, matrices_file_path):
        dataset = np.load(matrices_file_path)


        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples_index = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__matrix_length = self.__matrices.shape[1]
        self.__matrix_size = self.__matrices.shape[1]*self.__matrices.shape[2]
        assert(self.__matrix_size == 13*13)

        self.__length = len(self.__analogies_index)

        self.__analogies_lengths = dataset["lengths"]


    def __len__(self):
        return self.__length

    def __getitem__(self, index):

            matrices_indices = self.__analogies_index[index]
            return self.__matrices[matrices_indices], self.__sentence_couples_index[matrices_indices]



def extractPath(matrix1, matrix2, sentences1, sentences2, bilingual=False):
    paths = list()
    n, m = matrix1.shape
    m2, l = matrix2.shape
    assert (m == m2)
    for i in range(n):
        for j in range(m):
            if matrix1[i, j] != 0:
                for k in range(l):
                    if matrix2[j, k] != 0:
                        proba = matrix1[i, j] * matrix2[j, k]
                        if bilingual:
                            proba = 0*proba
                        path = (proba, sentences1[1][j], sentences1[0][i],
                                sentences2[1][k], i - j + k)
                        data.append(path)
    return paths

def decodePath(paths):
    result = list()
    for path in paths:
        proba, w1, w2, w3, position = path
        if w1 == w2:
            result.append((w3, position, proba))
        elif w1 == w3:
            result.append((w2, position, proba))
        else:
            # TODO: ajouter la possibilité de décoder avec word embedding
            pass
    return result

dataset = cuboidDataset("en-fr.matrices.npz")

solved = 0
tried = 0
candidates = list()
references = list()

#for i in [150]:
#for i in progressbar(range(10000, len(dataset))):
ids = list(range(len(dataset)))
shuffle(ids)

for i in ids[:1000]:
    matrices, sentences = dataset[i]
    sentences = sentences.tolist()

    #print("target= ", sentences[6][1])
    for i in range(8):
        for j in range(2):
            sentences[i][j] = word_tokenize(sentences[i][j].lower())

    #print(sentences)

    # m6 <- m0123
    #m0, m1


    data = list()

    data.extend(extractPath(matrices[0], matrices[1], sentences[0], sentences[1], bilingual=True))
    data.extend(extractPath(matrices[2], matrices[3], sentences[2], sentences[3], bilingual=True))
    data.extend(extractPath(matrices[4], matrices[5].T, sentences[4], sentences[5][::-1]))

    weight_dict = dict()
    word_dict = dict()
    word_index = list()
    position_dict = dict()
    position_index = list()
    elements = decodePath(data)
    for word, position, weight in elements:
        try:
            weight_dict[(word, position)] = max(weight_dict[(word, position)], weight)
        except:
            weight_dict[(word, position)] = weight

        if word not in word_dict:
            word_dict[word] = len(word_index)
            word_index.append(word)

        if position not in position_dict:
            position_dict[position] = len(position_index)
            position_index.append(position)

    c = list()
    for i in range(len(word_index)):
        for j in range(len(position_index)):
            try:
                c.append(weight_dict[(word_index[i], position_index[j])])

            except:
                c.append(0)

    assert(len(c) == len(position_index)*len(word_index))

    word_constraint = list()
    for i in range(len(word_index)):
        constraint = list()
        for j in range(len(word_index)):
            for k in range(len(position_index)):
                if i==j:
                    constraint.append(1)
                else:
                    constraint.append(0)
        word_constraint.append(constraint)


    position_constraint = list()
    for i in range(len(position_index)):
        constraint = list()
        for j in range(len(word_index)):
            for k in range(len(position_index)):
                if i==k:
                    constraint.append(1)
                else:
                    constraint.append(0)
        position_constraint.append(constraint)

    all_constraint = word_constraint + position_constraint
    all_constraint_rh = np.array([1]*len(all_constraint))
    all_constraint = np.array(all_constraint)
    c = -np.array(c)

    result = linprog(c, A_ub=all_constraint, b_ub=all_constraint_rh).x

    result.shape = (len(word_index), len(position_index))
    result = np.argmax(result, axis=0)
    #print(result)

    sentence = list()
    for position_id, word_id in enumerate(result):
        try:
            proba = weight_dict[(word_index[word_id], position_index[position_id])]
            #print(word_index[word_id], proba)
            if proba > 0.1:
                sentence.append(word_index[word_id])
        except:
            pass
    solution = sentence
    target = sentences[6][1]

    tried += 1
    if target == solution:
        solved += 1
        print(solution, target)
        print((solved/tried)*100, "% success")
    candidates.append(solution)
    references.append([target])

print(corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25)))

