import hashlib
import os
from collections import Counter
def exact_deduplication(input_files: list[os.PathLike], output_dir: os.PathLike):
    counter = Counter()
    for file_path in input_files:
        with open(file_path) as file:
            lines = file.readlines()
            hashes = [hashlib.sha256(line.encode()).hexdigest() for line in lines]
            counter.update(hashes)
    for file_path in input_files:
        with open(file_path) as file:
            lines = file.readlines()
            hashes = [hashlib.sha256(line.encode()).hexdigest() for line in lines]
            unique_lines = [
                line for line, hash in zip(lines, hashes) if counter[hash] == 1
            ]
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(output_path, "w") as output_file:
                output_file.writelines(unique_lines)