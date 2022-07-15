import logging
import os
import pickle
import random
import sys
import pandas as pd
import json

VERSION = 0.1
DIVIDER = "-".join("-" for i in range(50))

logging.basicConfig(handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
],
    level=logging.ERROR,
    format='%(asctime)s - %(message)s')


def get_labels(label_name: str) -> list:
    """
    Gets a list of acceptable labels from the user
    :param label_name: The column name that the labels relate to
    :return: The list of acceptable labels.
    """

    number_of_labels = input("Please enter the number of labels for " + label_name + ": ")

    while not number_of_labels.isdigit():
        logging.critical("Input is not an positive integer")
        number_of_labels = input("Please enter the number of labels for " + label_name + ": ")

    print("What labels are valid for " + label_name + "?")

    labels = []

    for i in range(int(number_of_labels)):
        labels.append(input("Enter a label: "))

    logging.info("Valid Labels:" + str(labels))

    # Add exit to label list to allow for partial saving
    labels.append("exit")

    return labels


def assign_label(labels: list) -> str:
    new_label = input("Please label the above file " + str(labels) + ": ")

    while new_label not in labels:
        logging.critical("Label is not part of the existing label set")
        new_label = input("Please label the above file " + str(labels) + ": ")

    print(DIVIDER)
    return new_label


def load_files(pickle_file: str) -> list:
    with open(pickle_file, "rb") as pf:
        return pickle.load(pf)


def save_in_progress(random_files: list, output_data: pd.DataFrame, output_file: str, label_name: str) -> None:
    """
    Saves the in progress labelling to a CSV.
    :param random_files: The random sample of files to label
    :param output_data: The output dataframe to write to a CSV
    :param output_file: The path to save the partially labelled CSV
    :param label_name: The name of the label
    :return: None
    """
    
    for file in random_files:
        output_data = pd.concat([output_data,
                                 pd.DataFrame([{'file_name': file,
                                                'source': None, 'compile_result': None,
                                                label_name: None}])],
                                ignore_index=True)

    logging.debug("Writing to file")
    output_data.to_csv(output_file, index=False)
    logging.debug("Saved file")


def label(files: list, labels: list, output_data: pd.DataFrame, output_file, label_name):
    for file in list(files):
        print(file + "\n")

        with open(file) as f:
            # print meta data
            meta = json.loads(f.read())
            print(str(meta) + "\n")

        with open(meta['src_file']) as f:
            # print source
            src = f.read()
            print(src + "\n")

        new_label = assign_label(labels)

        if new_label == "exit":
            logging.debug("Saving current state...")
            save_in_progress(files, output_data, output_file, label_name)
            return None

        output_data = pd.concat([output_data,
                                 pd.DataFrame([{'file_name': file,
                                                'source': src, 'compile_result': meta['compile_result'],
                                                label_name: new_label}])],
                                ignore_index=True)

        files.remove(file)

    return output_data


def initial_labeller(pickle_file: str, output_file: str, sample_size: int, label_name: str) -> None:
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

    output_data = label(random_files, labels, output_data, output_file, label_name)

    if output_data is None:
        return

    output_data.to_csv(output_file, index=False)


def continue_labelling(csv_file: str) -> None:
    """
    Continue labelling from the point that was last left off.
    :param csv_file: The CSV that has been previously partially saved
    :return: None
    """

    if not os.path.exists(csv_file):
        logging.fatal("CSV file not found")
        return

    output_data = pd.read_csv(csv_file)

    label_name = output_data.columns[len(output_data.columns) - 1]
    files = output_data[output_data[label_name].isna()]['file_name'].tolist()

    output_data = output_data[~output_data[label_name].isna()]

    labels = get_labels(label_name)

    output_data = label(files, labels, output_data, csv_file, label_name)

    if output_data is None:
        return

    output_data.to_csv(csv_file, index=False)


if __name__ == '__main__':
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if "-v" in opts or "--verbose" in opts:
        logging.getLogger().setLevel(logging.INFO)
    if "-vv" in opts:
        logging.getLogger().setLevel(logging.DEBUG)
    if "--version" in opts:
        print(VERSION)


    if len(args) == 4:
        if not args[2].isdigit():
            logging.critical("Sample size is not a positive integer")
        else:
            initial_labeller(pickle_file=args[0], output_file=args[1], sample_size=int(args[2]), label_name=args[3])
    elif len(args) == 1:
        if not args[0].endswith(".csv"):
            logging.critical("CSV not supplied")
        else:
            continue_labelling(csv_file=args[0])
    else:
        logging.critical("Please use the following format to start labelling "
                         "`python3 labeler.py <raw_data_root_dir> <csv_output> <sample_size> <label_name>'")
        logging.critical("Or use the following format to continue labelling `python3 labeler.py <csv_file>'")
        logging.critical("Use -v to enable logging")
        logging.critical("Not enough arguments to start process")
