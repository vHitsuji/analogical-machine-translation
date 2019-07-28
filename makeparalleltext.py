#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __doc__ string
"""
The purpose of this little script is to extract a parallel corpus from a cuboids corpus.
"""

__author__ = "Taillandier Valentin"
__copyright__ = "Copyright (C) 2019, Taillandier Valentin"
__license__ = "GPL"
__version__ = "1.0"


import argparse
try:
    # If progressbar module is available, it will be used to show some progressbars.
    # To install it -> pip3 install progressbar2 --user
    from progressbar import progressbar
except ImportError:
    def progressbar(x):
        return x



if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', dest='input_path', action='store', help='Cuboids text file path.')
    parser.add_argument('--output_l1_l2', dest='first_to_second_path', action='store',
                        help='Output path for the first language to second language parallel corpus.')
    parser.add_argument('--output_l2_l1', dest='second_to_first_path', action='store',
                        help='Output path for the second language to first language parallel corpus.')
    args = parser.parse_args()
    input_path = args.input_path
    output12_path = args.first_to_second_path
    output21_path = args.second_to_first_path

    # Load sentences
    input_file = open(input_path)
    lines = input_file.read().splitlines()
    input_file.close()

    assert(len(lines) % 2 == 0)

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






