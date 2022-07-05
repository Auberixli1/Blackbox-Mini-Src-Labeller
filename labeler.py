import logging
import os
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


def process_src(path: str, filename: str, desired_size: int, line_threshold: int) -> [str, None]:
    """
    The java source file to process and check the line length for.
    :param path: The base path
    :param filename: The java file name
    :param desired_size: The desired number of lines
    :param line_threshold: The threshold to allow for approximate values
    :return: The source files path if it has the number of lines within the threshold; otherwise None
    """
    file_path = os.path.join(path, filename)
    with open(file_path) as f:
        file_length = len(f.readlines())
        logging.debug(file_path + ":" + str(file_length))
        if desired_size - line_threshold <= file_length <= desired_size + line_threshold:
            return file_path


def process_meta(src_file: str) -> str:
    """
    Converts .java extension to the .json extension used in the metadata files.
    :param src_file: The src file to convert
    :return: The metadata file
    """
    return src_file.replace(".java", ".json")


def get_all_files(dir_to_label: str, desired_size: int, line_threshold: int) -> list:
    """
    Creates a list of files to facilitate random sampling
    Adapted from: https://stackoverflow.com/questions/6411811/randomly-selecting-a-file-from-a-tree-of-directories-in-a-completely-fair-manner
    :param dir_to_label: The base directory to take the sample from.
    :param desired_size: The total number of lines that need to be in the source file (approximately)
    :param line_threshold: The +- threshold for the number of lines in a source file
    :return: The list of all files
    """

    src_files = []

    pool = mp.Pool(mp.cpu_count() // MULTIPROCESS_DIVISOR)

    for path, _, files in os.walk(dir_to_label):
        temp_data = pool.starmap(process_src, [(path, f, desired_size, line_threshold)
                                           for f in files if f.endswith(".java")])
        src_files.append(temp_data)

    src_files = list(filter(None, [s for src in src_files for s in src]))
    src_files.sort()

    meta_files = pool.map(process_meta, src_files)
    meta_files.sort()

    pool.close()

    return list(zip(src_files, meta_files))


def get_desired_file_length(line_threshold: int) -> int:
    """
    Asks the user for the desired length of the source file.
    :param line_threshold: The +- threshold for the number of lines in a source file
    :return: The desired length as an int
    """
    length = input("How long should the sampled files be? +- " + str(line_threshold) + " Lines\n")

    if not length.isdigit():
        logging.critical("Input is not an positive integer")
        return get_desired_file_length(line_threshold)

    return int(length)


def main(dir_to_label: str, output_file: str, sample_size: int, line_threshold: int, label_name: str) -> None:
    """
    Used for labelling the Blackbox Mini Source Dataset.
    :param dir_to_label: The directory to take a random sample from
    :param output_file: The CSV to write the raw data and labels to for use in ML.
    :param sample_size: The number of random samples to take from the dataset to label
    :param line_threshold: The +- threshold for the number of lines in a source file
    :param label_name: the label name, used as the column in the CSV file
    :return: None
    """
    if not os.path.isdir(dir_to_label):
        logging.fatal("Directory is not valid")
        return

    if not output_file.endswith(".csv"):
        logging.fatal("Output file is not a CSV")
        return

    labels = get_labels(label_name)

    file_length = get_desired_file_length(line_threshold)

    files = get_all_files(dir_to_label, file_length, line_threshold)

    if len(files) == 0:
        logging.fatal("No files found with desired length.")
        return

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

    if len(args) != 5:
        logging.critical("Please add the directory to label and the file to save the labels to")
        logging.critical("python3 labeler.py /data/minisrc /home/mmesser/readability_labels.csv 100 20 readable")
        logging.critical("Use -v to enable logging")
        logging.critical("Not enough arguments to start process")
    else:
        if not args[2].isdigit() or not args[3].isdigit():
            logging.critical("Sample size is not a positive integer")
        else:
            main(dir_to_label=args[0], output_file=args[1], sample_size=int(args[2]),
                 line_threshold=int(args[3]), label_name=args[4])
