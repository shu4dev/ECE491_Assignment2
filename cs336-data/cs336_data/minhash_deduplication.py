import os
import mmh3
import random
import string
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from pathlib import Path
def generate_ngrams(n:int, text:str) -> list[str]:
    normalized_text = normalize(text)
    n_grams = ngrams(word_tokenize(normalized_text), n)
    result = [''.join(grams) for grams in n_grams]
    return result

def normalize(text:str) -> str:
    lower_case = text.lower()
    remove_punc = lower_case.translate(str.maketrans('', '', string.punctuation))
    normalize_whitespace = " ".join(remove_punc.split())
    result = unicodedata.normalize('NFD', normalize_whitespace)
    return result

def MinHash(num_hashes: int, text: list[str]):
    random.seed(42)
    random_numbers = random.sample(range(1,num_hashes * 100), num_hashes)
    signature = []
    for num in random_numbers:
        hashes  = []
        for gram in text:
            hash_value = mmh3.hash(gram, num)
            hashes.append(hash_value)
        signature.append(min(hashes))
    return signature

def Jaccard(hash1: set, hash2: set):
    return len(hash1.intersection(hash2)) / len(hash1.union(hash2)) 

def LSHash(num_bands:int, minhash1:list[int], minhash2:list[int]):
    for i in range(0, len(minhash1), num_bands):
        band1 = minhash1[i:i+num_bands]
        band2 = minhash2[i:i+num_bands]
        if band1 == band2:
            return True
    return False

def fuzzy_deduplication(input_files: list[os.PathLike], num_hashes: int, num_bands: int, ngrams: int, jaccard_threshold: float, output_directory: os.PathLike):
    cluster, temp = 0, 0
    index = 1
    meet = 0
    bucket = input_files[:]
    while index < len(input_files):
        with open(input_files[cluster]) as f:
            text1 = f.read()
            text1 = normalize(text1)
        with open(input_files[index]) as f:
            text2 = f.read()
            text2 = normalize(text2)
        ngrams1 = generate_ngrams(ngrams, text1)
        ngrams2 = generate_ngrams(ngrams, text2)
        minhash1 = MinHash(num_hashes, ngrams1)
        minhash2 = MinHash(num_hashes, ngrams2)
        if LSHash(num_bands, minhash1, minhash2):
            jaccard = Jaccard(set(minhash1), set(minhash2))
            if jaccard > jaccard_threshold:
                bucket[index] = Path('/')
            else:
                if meet == 0:
                    temp = index
                    meet = 1
                else:
                    temp = min(temp, index) 
        else:
            if meet == 0:
                temp = index
                meet = 1
            else:
                temp = min(temp, index)
        if index == len(input_files) - 1 and (cluster == len(input_files) - 2 or temp == len(input_files) - 1):
            for file_path in bucket:
                if file_path != Path('/'):
                    with open(file_path) as file:
                        lines = file.readlines()
                        output_path = os.path.join(output_directory, os.path.basename(file_path))
                        with open(output_path, "w") as output_file:
                            output_file.writelines(lines)
            break
        elif index == len(input_files) - 1:
            cluster = temp
            meet = 0
            index = cluster + 1
        else:
            index += 1