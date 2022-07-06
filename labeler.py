import logging
import os
import pickle
import random
import sys
import pandas as pd
import json
import multiprocessing as mp
from multiprocessing_logging import install_mp_handler

VERSION = 0.1
DIVIDER = "-".join("-" for i in range(50))

logging.basicConfig(handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
],
    level=logging.ERROR,
    format='%(processName)s - %(asctime)s - %(message)s')

MULTIPROCESS_DIVISOR = 4

install_mp_handler()


def get_labels(label_name: str) -> list:
    """
    Gets a list of acceptable labels from the user
    :param label_name: The column name that the labels relate to
    :return: The list of acceptable labels.
    """

    number_of_labels = input("Please enter the number of labels for " + label_name + ": ")

    if not number_of_labels.isdigit():
        logging.critical("Input is not an positive integer")
        return get_labels(label_name)

    print("What labels are valid for " + label_name + "?")

    labels = []

    for i in range(int(number_of_labels)):
        labels.append(input("Enter a label: "))

    logging.info("Valid Labels:" + str(labels))
    return labels


def assign_label(labels: list) -> str:
    label = input("Please label the above file " + str(labels) + ": ")

    if label not in labels:
        logging.critical("Label is not part of the existing label set")
        return assign_label(labels)

    print(DIVIDER)
    return label


def load_files(pickle_file: str) -> list:
    with open(pickle_file, "rb") as pf:
        return pickle.load(pf)


def main(pickle_file: str, output_file: str, sample_size: int, label_name: str) -> None:
    """
    Used for labelling the Blackbox Mini Source Dataset.
    :param pickle_file: The pickle file to read
    :param output_file: The CSV to write the raw data and labels to for use in ML.
    :param sample_size: The number of random samples to take from the dataset to label
    :param label_name: the label name, used as the column in the CSV file
    :return: None
    """
    if not os.path.exists(pickle_file) and pickle_file.endswith(".pickle"):
        logging.fatal("Pickle file is not valid")
        return

    if not output_file.endswith(".csv"):
        logging.fatal("Output file is not a CSV")
        return

    labels = get_labels(label_name)

    files = load_files(pickle_file)

    output_data = pd.DataFrame(columns=['file_name', 'source', 'compile_result', label_name])

    if sample_size >= len(files):
        sample_size = len(files)

    random_files = random.choices(files, k=sample_size)

    for file in random_files:
        print(file[0] + "\n")

        with open(file[1]) as f:
            # print meta data
            meta = json.loads(f.read())
            print(str(meta) + "\n")

        with open(file[0]) as f:
            # print source
            src = f.read()
            print(src + "\n")

        label = assign_label(labels)

        output_data = pd.concat([output_data,
                                 pd.DataFrame([{'file_name': file,
                                                'source': src, 'compile_result': meta['compile_result'],
                                                label_name: label}])],
                                ignore_index=True)

    output_data.to_csv(output_file)


if __name__ == '__main__':
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if "-v" in opts or "--verbose" in opts:
        logging.getLogger().setLevel(logging.INFO)
    if "-vv" in opts:
        logging.getLogger().setLevel(logging.DEBUG)
    if "--version" in opts:
        print(VERSION)

    if len(args) != 4:
        logging.critical("Please add the directory to label and the file to save the labels to")
        logging.critical("python3 labeler.py /data/minisrc /home/mmesser/readability_labels.csv 100 readable")
        logging.critical("Use -v to enable logging")
        logging.critical("Not enough arguments to start process")
    else:
        if not args[2].isdigit():
            logging.critical("Sample size is not a positive integer")
        else:
            main(pickle_file=args[0], output_file=args[1], sample_size=int(args[2]), label_name=args[3])
