# UHM ECE 496B Spring 2025 Assignment 2: Data

This asignment is created from Assignment 4 of [CS336 at Stanford taught in Spring 2024](https://stanford-cs336.github.io/spring2024/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment4_data.pdf](./cs336_spring2024_assignment4_data.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2024-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `setup.py`. This module should contain your
  from-scratch language model from assignment 1. At this point, **you are free
  to optimize this model however you wish**---feel free to replace your
  hand-crafted components with PyTorch built-ins wherever you like.
- [`./cs336-data`](./cs336-data): directory containing a module
  `cs336_data` and its associated `setup.py`. In this module, you will
  implement code to filter and preprocess data.

Visually, it should look something like:

``` sh
.
├── cs336-basics # Files from assignment 1, feel free to make optimizations 
│   ├── cs336_basics # A python module named cs336_basics
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... other files in the cs336_basics module, taken from assignment 1 ...
│   ├── requirements.txt
│   └── setup.py (setup.py to install `cs336_basics`) 
├── cs336-data # TODO(you):code that you'll write for assignment 2
│   ├── cs336_data # A python module named cs336_data
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... TODO(you): other python files that you need for assignment 2 ...
│   ├── requirements.txt
│   ├── ... TODO(you): any other files or folders you need for assignment 2 ...
│   └── setup.py (setup.py to install `cs336_data`)
├── README.md
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

1. Set up a conda environment and install packages. In particular, the
   `cs336-basics` package (located at [`./cs336-basics`](./cs336-basics))
   installs the `cs336_basics` module, and the `cs336-data` package (located
   at [`./cs336-data`](./cs336-data)) installs the `cs336_data` module.

``` sh
conda create -n cs336_data python=3.10 --yes
conda activate cs336_data
pip install -e ./cs336-basics/ -e ./cs336-data/'[test]'
```

2. Activate the environment:

``` sh
conda activate cs336_data
```

## ECE491B Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2024_assignment4_data.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/s2025-assignment2-data.git
2. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://forms.gle/CSRweWjuBxvYbb9MA). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
    - You don't need to submit to leaderboard.
3. None of the data or tools are pre-downloaded or pre-installed for you. The handout describes the steps to get them (e.g. download specific WARC files or fastText library). You should follow the steps yourself.
4. Problems listed in Section 4 are not required and can be skipped. However, they can be submitted for extra credit even after the deadline for Assignment 2 (and before the last day of the class) 
    - You may leverage CPUs located on Koa cluster to solve the problems listed in Section 4
