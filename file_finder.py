import logging
import os
import sys
import pickle
import multiprocessing as mp
from datetime import datetime

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


desired_length = 100
line_threshold = 5


def process(path: str) -> [str, None]:
    """
    Process the files to get the correct length
    :param path: The root path to process
    :return: A list of files that are approx the correct length
    """

    src_path = path.replace("json", "java")
    if src_path == "/data/minisrc/srcml-2019-09/project-17094036/src-83472986.java":
        logging.info("Skipping extremely large file")
        return None

    with open(src_path) as src_file:
        file_length = len(src_file.readlines())
        logging.debug(src_path + ":" + str(file_length))
        if desired_length - line_threshold <= file_length <= desired_length + line_threshold:
            return path

    return None


def get_all_files(dir_to_label: str) -> list:
    """
    Creates a list of files to facilitate random sampling
    Adapted from: https://stackoverflow.com/questions/6411811/randomly-selecting-a-file-from-a-tree-of-directories-in-a-completely-fair-manner
    :param dir_to_label: The base directory to take the sample from.
    :param desired_size: The total number of lines that need to be in the source file (approximately)
    :param line_threshold: The +- threshold for the number of lines in a source file
    :return: The list of all files
    """

    pool = mp.Pool(mp.cpu_count() // MULTIPROCESS_DIVISOR)

    meta_files = []

    for path, _, files in os.walk(dir_to_label):
        meta_files = pool.map(process, [os.path.join(path, file)
                                        for file in files if file.endswith(".json")])

    meta_files = list(filter(None, meta_files))
    pool.close()

    return meta_files


def main(dir_to_label: str) -> None:
    """
    Used for labelling the Blackbox Mini Source Dataset.
    :param dir_to_label: The directory to take a random sample from
    :param file_length: The ideal length of the file to get
    :param line_threshold: The +- threshold for the number of lines in a source file
    :return: None
    """
    if not os.path.isdir(dir_to_label):
        logging.fatal("Directory is not valid")
        return

    files = get_all_files(dir_to_label)

    if len(files) == 0:
        logging.fatal("No files found with desired length.")
        return

    with open("found_files_" + datetime.now().isoformat(timespec="seconds") + ".pickle", "wb") as of:
        logging.info("Saving found files... (" + str(len(files)) + ")")
        logging.debug("Files: " + str(files))
        pickle.dump(files, of)


if __name__ == '__main__':
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if "-v" in opts or "--verbose" in opts:
        logging.getLogger().setLevel(logging.INFO)
    if "-vv" in opts:
        logging.getLogger().setLevel(logging.DEBUG)
    if "--version" in opts:
        print(VERSION)

    if len(args) != 3:
        logging.critical("Please add the directory to find all files, the desired file length and the line threshold")
        logging.critical("python3 labeler.py /data/minisrc 100 20")
        logging.critical("Use -v to enable logging")
        logging.critical("Not enough arguments to start process")
    else:
        if not args[1].isdigit() or not args[2].isdigit():
            logging.critical("Desired length of line threshold is not a positive integer")
        else:
            desired_length = int(args[1])
            line_threshold = int(args[2])

            main(dir_to_label=args[0])
