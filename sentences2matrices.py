#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gensim.models import KeyedVectors, keyedvectors
from nltk.tokenize import word_tokenize
import numpy as np
import argparse
from progressbar import progressbar



matrix_size = 32 # Define the shape of the matrices

def cossim2distance(x):
    """

    :param x: a scalar value or a np.array like (in this case, apply elementwise).
    :return: The distance base on x seen as the cosine similarity (return sqrt(2-2x))
    """
    # distance = sqrt(2(1-cossim(a,b)))
    return np.sqrt(2*(1-np.minimum(x, 1)))


def normalizeAndSymetrize(matrix):
    """
    Normalize (Norm L1) a matrix in the first and second axis (then we obtain two matrices).
    Then, the component wise geometric means of these two matrices are computed.
    :param matrix: A numpy ndarray like matrix
    :return: None, the computation is done in place;
    """
    ponderation = np.sqrt(np.sum(matrix, axis=1, keepdims=True).dot(np.sum(matrix, axis=0, keepdims=True)))
    assert(ponderation.shape == matrix.shape)
    np.divide(matrix, ponderation, out=matrix)


def reshaperMatrix(matrix, n):
    """
    Take a matrix and expend it to the nxn size.
    :param matrix: input ndarray like matrix.
    :param n: first axis size.
    :return: a nxn array.
    """
    output_matrix = (-1)*np.zeros((n, n))
    input_n, input_m = matrix.shape
    for x in range(input_n):
        for y in range(input_m):
            output_matrix[x, y] = matrix[x, y]
    return output_matrix





def monolingualMatrices(sentences_couples, model_path, dict_to_update):
    wv_model = KeyedVectors.load(model_path)
    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = word_tokenize(sentence1.lower())
        s2_words = word_tokenize(sentence2.lower())
        try:
            s1_vectors = np.array([wv_model.get_vector(word) for word in s1_words])
            s2_vectors = np.array([wv_model.get_vector(word) for word in s2_words])
            s1_vectors_normalized = s1_vectors/np.linalg.norm(s1_vectors, axis=1, keepdims=True)
            s2_vectors_normalized = s2_vectors/np.linalg.norm(s2_vectors, axis=1, keepdims=True)
            matrix = s1_vectors_normalized.dot(s2_vectors_normalized.T)
            dict_to_update[(sentence1, sentence2)] = matrix
            dict_to_update[(sentence2, sentence1)] = matrix.T
        except:
            pass

def bilingualMatrices(sentences_couples, model_path, dict_to_update):
    tt_model = dict()
    model_file = open(model_path)
    for line in model_file:
        w1, w2, score = line.rstrip('\n').split(" ")
        tt_model[(w1, w2)] = np.float32(score)
    model_file.close()


    for sentence1, sentence2 in progressbar(sentences_couples):
        s1_words = word_tokenize(sentence1.lower())
        s2_words = word_tokenize(sentence2.lower())
        n = len(s1_words)
        m = len(s2_words)
        matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                try:
                    matrix[i, j] = tt_model[(s1_words[i], s2_words[j])]
                except:
                    matrix[i, j] = 0
        matrix = (matrix * 2) - 1
        dict_to_update[(sentence1, sentence2)] = matrix


def couple_sort(couple):
    if couple[0] <= couple[1]:
        return couple
    else:
        return couple[1], couple[0]


if __name__ == '__main__':

    #Argz parsing
    parser = argparse.ArgumentParser(description='Computes alignment matrices.')
    parser.add_argument('--input', dest='input_path', action='store',
                        help='Cuboid analogies textfile to proceed.')
    parser.add_argument('--output', dest='output_path', action='store',
                        help='Output name to store matrices.')
    parser.add_argument('--first_language_model', dest='model1_path', action='store',
                        help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--second_language_model', dest='model2_path', action='store',
                        help='Word embeding model path. Will be open with gensim.open().')
    parser.add_argument('--bilingual_model', dest='bilingual_model_path', action='store',
                        help='Word translation model path. Should looks like a trained Hieraligne model.')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model1_path = args.model1_path
    model2_path = args.model2_path
    bilingual_model_path = args.bilingual_model_path


    #Load sentences
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()
    assert(len(lines) % 2 == 0)  # One analogy takes 2 lines. So there is an even number of lines.
    language1_couples = set()
    language2_couples = set()
    bilingual_couples = set()

    # for each analogy.
    analogies = list()
    for i in progressbar(range(0, len(lines), 2)):
        analogies.append([lines[i].split("\t"), lines[i + 1].split("\t")])
    for analogy in progressbar(analogies):
        language1_couples.add((analogy[0][0], analogy[0][1]))
        language1_couples.add((analogy[0][2], analogy[0][3]))
        language1_couples.add((analogy[0][0], analogy[0][2]))
        language1_couples.add((analogy[0][1], analogy[0][3]))

        language2_couples.add((analogy[1][0], analogy[1][1]))
        language2_couples.add((analogy[1][2], analogy[1][3]))
        language2_couples.add((analogy[1][0], analogy[1][2]))
        language2_couples.add((analogy[1][1], analogy[1][3]))

        for k in range(4):
            bilingual_couples.add((analogy[0][k], analogy[1][k]))

    matrices_dict = dict()
    monolingualMatrices(language1_couples, model1_path, matrices_dict)
    monolingualMatrices(language2_couples, model2_path, matrices_dict)
    bilingualMatrices(bilingual_couples, bilingual_model_path, matrices_dict)

    #  From there, matrices_dict contains all the necessary matrices
    equivalent_analogies = [
        [0, 1, 2, 3], [2, 3, 0, 1],
        [0, 2, 1, 3], [1, 3, 0, 2],
        [1, 0, 3, 2], [3, 2, 1, 0],
        [2, 0, 3, 1], [3, 1, 2, 0]
    ]


    keys, values = zip(*matrices_dict.items())
    keys_index = {key: index for index, key in enumerate(keys)}

    analogy_matrices_list = []
    analogies_lengths_list = []
    for analogy in progressbar(analogies):
        for i0, i1, i2, i3 in equivalent_analogies:
            try:
                analogy_matrices = [
                    keys_index[(analogy[0][i0], analogy[0][i1])],
                    keys_index[(analogy[0][i1], analogy[1][i1])],
                    keys_index[(analogy[0][i0], analogy[0][i2])],
                    keys_index[(analogy[0][i2], analogy[1][i2])],
                    keys_index[(analogy[1][i2], analogy[1][i3])],
                    keys_index[(analogy[1][i1], analogy[1][i3])],

                    keys_index[(analogy[0][i0], analogy[1][i0])],
                    keys_index[(analogy[1][i0], analogy[1][i2])],
                    keys_index[(analogy[1][i0], analogy[1][i1])],
                ]
                analogy_matrices_list.append(analogy_matrices)

                la, lb = values[analogy_matrices[0]].shape
                lc, lc2 = values[analogy_matrices[3]].shape
                la2, lb2 = values[analogy_matrices[8]].shape
                _, ld2 = values[analogy_matrices[4]].shape

                analogies_lengths_list.append((la, lb, lc, lb2, lc2, ld2, la2))

            except:
                pass




    analogies_lengths_dict = dict()
    for lengths in analogies_lengths_list:
        try:
            analogies_lengths_dict[lengths] += 1
        except:
            analogies_lengths_dict[lengths] = 1
    print(len(analogies_lengths_dict))

    matrices = [reshaperMatrix(value, matrix_size) for _, value in enumerate(values)]
    print(matrices[:3])
    matrices = np.stack(matrices, axis=0)
    np.savez(output_path, index=keys, analogies=analogy_matrices_list, matrices=matrices, lengths=analogies_lengths_list)







