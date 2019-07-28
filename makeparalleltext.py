#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from progressbar import progressbar




if __name__ == '__main__':

    #Argz parsing
    parser = argparse.ArgumentParser(description='Extract bitexts from cuboid analogies file.')
    parser.add_argument('--input', dest='input_path', action='store', help='Cuboid analogies textfile to proceed.')
    parser.add_argument('--output_l1_l2', dest='output12_path', action='store', help='Output name for the language1 to language2 bitext.')
    parser.add_argument('--output_l2_l1', dest='output21_path', action='store', help='Output name for the language2 to language1 bitext.')
    args = parser.parse_args()
    input_path = args.input_path
    output12_path = args.output12_path
    output21_path = args.output21_path


    #Load sentences
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()

    assert(len(lines) % 2 == 0)  # One analogy takes 2 lines. So there is an even number of lines.

    bilingual_couples = set()
    bitext12 = []
    bitext21 = []

    for i in progressbar(range(0, len(lines), 2)):
        sentences = [lines[i].split("\t"), lines[i+1].split("\t")]
        for k in range(4):
            bilingual_couples.add((sentences[0][k], sentences[1][k]))

    for s1, s2 in bilingual_couples:
        bitext12.append("\t".join((s1, s2)))
        bitext21.append("\t".join((s2, s1)))

    bitext12 = "\n".join(bitext12)
    bitext21 = "\n".join(bitext21)

    bitext12_file = open(output12_path, "w+")
    bitext21_file = open(output21_path, "w+")
    bitext12_file.write(bitext12)
    bitext21_file.write(bitext21)
    bitext12_file.close()
    bitext21_file.close()






