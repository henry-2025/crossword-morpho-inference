from logging import exception
import os
import argparse
import csv
import nltk
import re

# meant to be run on the big .tsv dataset, not the smaller two
def main(args):
    # choose to separate files into individual
    tagger = nltk.tag.PerceptronTagger()
    writer = csv.writer(args.out_file)

    i = 0
    args.in_file.readline()
    for line in args.in_file.readlines():
        words = line.split()
        if len(words) < 4: continue

        # make a bunch of features
        pubid = words[0]
        year = words[1]
        targ = words[2]
        clue = ' '.join(words[3:])
        targ_tag = tagger.tag([targ])[0][1]
        targ_len = len(targ)

        writer.writerow([pubid, year, clue, targ, targ_tag])
        if i % 10000 == 0 and i:
            print(f"Processed {i} clue/targ pairs")
        if args.limit:
            if i > args.limit: break
        i += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of crossword data stored in the xd format and output to a pickle file with ")
    parser.add_argument('in_file', help="parent directory for xd files", type=argparse.FileType('r'))
    parser.add_argument('out_file', help='parent directory for tagged pickle output files', type=argparse.FileType('w'))
    parser.add_argument('--separate', type=bool, help="separate aggregate data into files corresponding to directories")
    parser.add_argument('--limit', type=int, default=None, help="number of lines in the input file that should be processed")

    args = parser.parse_args()
    main(args)