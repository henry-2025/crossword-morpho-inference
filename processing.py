import os
import argparse
import pickle
import spacy
import re

nlp = spacy.load("en_core_web_sm")

POS_THRESHOLD = 0.3

def main(args):
    # choose to separate files into individual
    tagger = nlp.get_pipe("tagger")
    xdre = re.compile("^[A,D]\d+\.")
    targre = re.compile("~ (.*)$")
    cluere = re.compile("^[A,D]\d+\. (.*) ~")
    if (args.separate):
        os.makedirs(args.out_dir, exist_ok=True)
        for root, dirs, files in os.walk(args.in_dir):
            for fname in files:
                print(fname)
                in_file = open(os.path.join(root, fname), 'r')
                
                clue_pos_pairs = []
                for l in in_file.readlines():
                    if xdre.search(l):
                        targ = targre.search(l)[1]
                        clue = cluere.search(l)[1]

                        targ = nlp(targ)
                        # pick the parts of speech guesses with probabilities greater than POS_THRESHOLD
                        pos_guesses = [i for (i,_) in list(filter(lambda x: x[1] > POS_THRESHOLD, enumerate(tagger.model.predict([targ])[0][0])))]
                        clue_pos_pairs.append((clue, pos_guesses))
                out_file_path = os.path.join(args.out_dir, os.path.splitext(fname)[0] + ".pkl")
                out_file = open(out_file_path, 'wb')
                pickle.dump(clue_pos_pairs, out_file)   
                out_file.close()                     
    
    # create one large file containing all crossword data
    else:
        for _, _, files in os.walk(args.directory):
            pass

            

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of crossword data stored in the xd format and output to a pickle file with ")
    parser.add_argument('in_dir', help="parent directory for xd files")
    parser.add_argument('out_dir', help='parent directory for tagged pickle output files')
    parser.add_argument('--separate', type=bool, help="separate aggregate data into files corresponding to directories")

    args = parser.parse_args()
    main(args)