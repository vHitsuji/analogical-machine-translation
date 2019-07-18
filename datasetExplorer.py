
import numpy as np
import matplotlib.pyplot as plt

class cuboidDataset():
    def __init__(self, matrices_file_path):
        dataset = np.load(matrices_file_path)


        self.__analogies_index = dataset["analogies"]
        self.__sentence_couples = dataset["index"]
        self.__matrices = dataset["matrices"]
        self.__matrix_length = self.__matrices.shape[1]
        self.__matrix_size = self.__matrices.shape[1]*self.__matrices.shape[2]

        self.__length = len(self.__analogies_index)

        self.__analogies_lengths = dataset["lengths"]

    def getallmatrices(self):
        return self.__matrices

    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        matrices_indices = self.__analogies_index[index]
        return Cuboid(self.__matrices[matrices_indices], self.__analogies_lengths[index], \
            self.__sentence_couples[matrices_indices])


class Cuboid():
    def __init__(self, matrices, lengths, sentences):
        self.__matrices = matrices
        self.__lengths = lengths
        self.__sentences = sentences

    def getAnalogy(self, id):
        """
        for A:B::C:D
        matrices = mAB, mCD, mAC, mBD
        lengths = lA, lB, lC, lD
        sentences = A, B, C, D

        """

        assert(id <= 2 and id >= 0)
        if id == 0:
            #  B:B'::A:A'
            sentences = self.__sentences[0][1], self.__sentences[0][0], self.__sentences[8][1], self.__sentences[8][0]
            lengths = self.__lengths[1], self.__lengths[0], self.__lengths[3], self.__lengths[6]
            matrices = self.__matrices[0].T, self.__matrices[8].T, self.__matrices[1], self.__matrices[6]
        elif id == 1:
            sentences = self.__sentences[2][1], self.__sentences[2][0], self.__sentences[7][1], self.__sentences[7][0]
            lengths = self.__lengths[2], self.__lengths[0], self.__lengths[4], self.__lengths[6]
            matrices = self.__matrices[2].T, self.__matrices[7].T, self.__matrices[3], self.__matrices[6]
        else:
            sentences = self.__sentences[4][1], self.__sentences[4][0], self.__sentences[8][1], self.__sentences[8][0]
            lengths = self.__lengths[3], self.__lengths[5], self.__lengths[4], self.__lengths[6]
            matrices = self.__matrices[4].T, self.__matrices[8].T, self.__matrices[5].T, self.__matrices[7].T

        return Analogy(matrices, lengths, sentences)


class Analogy():
    def __init__(self, matrices, lengths, sentences):
        self.__matrices = matrices
        self.__lengths = lengths
        self.__sentences = sentences

    def __str__(self):
        return " ".join((self.__sentences[0], ":", self.__sentences[1],
                        "::",
                        self.__sentences[2], ":", self.__sentences[3]))


    def __printMatrix(self, matrix, s1_words, s2_words):
        l1 = len(s1_words)
        l2 = len(s2_words)
        plt.imshow(matrix, cmap="gray_r")
        plt.xticks(range(len(s2_words)), s2_words)
        plt.yticks(range(len(s1_words)), s1_words)

    def showRatio(self, id):
        assert(id <= 3 and id >= 0)
        matrix = self.__matrices[id]

        if id == 0:
            sentences1 = self.__sentences[0]
            sentences2 = self.__sentences[1]
        elif id == 1:
            sentences1 = self.__sentences[2]
            sentences2 = self.__sentences[3]
        elif id == 2:
            sentences1 = self.__sentences[0]
            sentences2 = self.__sentences[3]
        else:
            sentences1 = self.__sentences[1]
            sentences2 = self.__sentences[3]
        s1_words = sentences1.split(" ")
        s2_words = sentences2.split(" ")
        self.__printMatrix(matrix[:len(s1_words), :len(s2_words)], s1_words, s2_words)
        plt.show()

    @staticmethod
    def __matrixToLatex(matrix):
        matrix = (matrix*100)
        n, m = matrix.shape
        rows = list()
        for i in range(n):
            row = list()
            for j in range(m):
                row.append(str(int(matrix[i, j])))
            row = ",".join(row)
            rows.append(row)
        rows = "\\\\".join(rows)
        return rows



    def latexAnalogy(self):

        c = self.__sentences[2].split(" ")[::-1]
        a = self.__sentences[0].split(" ")
        b = self.__sentences[1].split(" ")
        d = self.__sentences[3].split(" ")[::-1]
        #matrix1 = self.__matrices[2][:len(a), :len(c)][::-1, ::-1]
        matrix1 = self.__matrices[2][::-1, ::-1]
        #matrix2 = self.__matrices[0][:len(a), :len(b)][::-1, ::]
        matrix2 = self.__matrices[0][::-1, ::]
        #matrix3 = self.__matrices[1].T[:len(d), :len(c)][::, ::-1]
        matrix3 = self.__matrices[1].T[::, ::-1]
        #matrix4 = self.__matrices[3][:len(b), :len(d)].T
        matrix4 = self.__matrices[3].T

        a_latex = "".join(["{" + word + "}" for word in a])
        b_latex = "".join(["{" + word + "}" for word in b])
        c_latex = "".join(["{" + word + "}" for word in c])
        d_latex = "".join(["{" + word + "}" for word in d])
        matrix1_latex = self.__matrixToLatex(matrix1)
        matrix2_latex = self.__matrixToLatex(matrix2)
        matrix3_latex = self.__matrixToLatex(matrix3)
        matrix4_latex = self.__matrixToLatex(matrix4)

        string = "\\alignmatadv{" + \
                 "}{".join([a_latex, c_latex, b_latex, d_latex,
                            matrix1_latex, matrix2_latex, matrix3_latex, matrix4_latex]) + "}"
        return string


    def showAnalogy(self, padding=False):

        plt.subplot(2,2,1)
        s1_words = self.__sentences[2].split(" ")[::-1]
        s2_words = self.__sentences[0].split(" ")[::-1]

        matrix = self.__matrices[2][::-1, ::-1]
        #matrix = self.__matrices[2][:len(s2_words), :len(s1_words)][::-1, ::-1]
        self.__printMatrix(matrix, [], s1_words)


        plt.subplot(2, 2, 2)
        s1_words = self.__sentences[0].split(" ")[::-1]
        s2_words = self.__sentences[1].split(" ")

        matrix = self.__matrices[0][::-1, ::]
        #matrix = self.__matrices[0][:len(s1_words), :len(s2_words)][::-1, ::]
        self.__printMatrix(matrix, s1_words, s2_words)

        plt.subplot(2, 2, 3)
        s1_words = self.__sentences[3].split(" ")
        s2_words = self.__sentences[2].split(" ")[::-1]

        matrix = self.__matrices[1].T[::, ::-1]
        #matrix = self.__matrices[1].T[:len(s1_words), :len(s2_words)][::, ::-1]
        self.__printMatrix(matrix, [], [])


        s1_words = self.__sentences[1].split(" ")
        s2_words = self.__sentences[3].split(" ")
        plt.subplot(2, 2, 4)
        matrix = self.__matrices[3].T
        #matrix = self.__matrices[3][:len(s1_words), :len(s2_words)].T
        self.__printMatrix(matrix, s2_words, [])

        plt.show()


def printMatrix(matrix, s1_words=None, s2_words=None):
    """
    Print an alignment matrix with given string as axis scale.
    Do not forget to plt.show() when you want to show the matrix.
    :param matrix: Matrix to show.
    :param s1_words: x_axis sentence.
    :param s2_words: y_axis sentence.
    :return: None
    """
    plt.matshow(matrix, cmap="gray_r")
    if s1_words is not None and s1_words is not None:
        plt.xticks(range(len(s2_words)), s2_words)
        plt.yticks(range(len(s1_words)), s1_words)

dataset = cuboidDataset("en-fr.matrices.npz")
matrices = dataset.getallmatrices()

printMatrix(np.mean(matrices, axis=0))
printMatrix(np.std(matrices, axis=0))
plt.show()