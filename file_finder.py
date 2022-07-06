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


def main(dir_to_label: str, file_length: int, line_threshold: int) -> None:
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

    files = get_all_files(dir_to_label, file_length, line_threshold)

    if len(files) == 0:
        logging.fatal("No files found with desired length.")
        return

    with open("found_files_" + datetime.now().strftime("%Y-%m-%d_%I:%M_%S") + ".pickle", "wb") as of:
        logging.info("Saving found files... (" + str(len(files)) + ")")
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
            main(dir_to_label=args[0], file_length=int(args[1]), line_threshold=int(args[2]))
