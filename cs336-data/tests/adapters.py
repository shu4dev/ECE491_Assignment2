#!/usr/bin/env python3
from __future__ import annotations
from cs336_data.extract import extract_text
from cs336_data.language_identification import language_identification
from cs336_data.pii import pii
from cs336_data.toxicity import toxicity
from cs336_data.quality import gopher
import os
from typing import Any

qt = gopher()
tox = toxicity()
li = language_identification()
pid = pii()

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)

def run_identify_language(text: str) -> tuple[Any, float]:
    return li.predict(text)

def run_mask_emails(text: str) -> tuple[str, int]:
    return pid.mask_emails(text)

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return pid.mask_phone_numbers(text)

def run_mask_ips(text: str) -> tuple[str, int]:
    return pid.mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return tox.classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return tox.classify_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return qt.classify_quality(text)

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
