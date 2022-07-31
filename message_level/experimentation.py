import numpy as np
import seaborn
import torch

if __name__ == '__main__':
    df = seaborn.load_dataset("penguins")
    print(df)