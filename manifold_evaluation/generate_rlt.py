import disentanglement.libs.disentanglement.gs as gs
import os
import numpy as np
import json
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pdb



def generate_rlt(preprocessed_barcode_path,name, load_preprocessed,preprocess=True,plot=True):
    filename = preprocessed_barcode_path
    if not os.path.exists(filename) or not load_preprocessed:
        print(f'mean barcodes dont exist or aren\'t being used')
        filename = "./results/barcode/"+name+"_results.json"
        mode = "bary"
    else:
        mode = "baryed"
    if plot:
        if not os.path.exists(f"figure_interps/{name}"):
            os.makedirs(f"figure_interps/{name}", exist_ok=True)
    with open(filename, "r") as f:
        results_dict = json.load(f)

    # factor_names = ["Position", "Velocity", "Agent Observations", "Target Observations", "Agent Number", "Other"]
    vis_type = "step"
    filetype = "png"
    for i, (key, value_map) in tqdm(enumerate(results_dict.items())):
        num_barcodes = len(value_map.items())
        cur_items = list(value_map.items())
        random.shuffle(cur_items)
        plt.style.use('seaborn')
        num_landmarks = 100  # 64
        fig, ax = plt.subplots()
        for val, barcode in cur_items:
            if mode == "bary":
                cur_code = gs.barymean(np.asarray(barcode))
            elif mode == "baryed":
                cur_code = barcode
            else:
                cur_code = np.mean(np.asarray(barcode), 0)
            if preprocess:
                if isinstance(cur_code, list):
                    results_dict[key][val] = cur_code
                else:
                    results_dict[key][val] = cur_code.tolist()
            if plot:
                if vis_type == "smooth":
                    sns.lineplot(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.2),
                                 color=f'C{i}')
                    plt.fill_between(np.arange(num_landmarks), cur_code[:num_landmarks],
                                     alpha=max(1 / num_barcodes, 0.1), color=f'C{i}')
                else:
                    sns.lineplot(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.2),
                                 color=f'C{i}', drawstyle='steps-pre')
                    plt.fill_between(np.arange(num_landmarks), cur_code[:num_landmarks],
                                     alpha=max(1 / num_barcodes, 0.1), color=f'C{i}', step="pre")
                plt.xlabel("Holes", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                # plt.title(factor_names[i])
        if plot:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
            plt.tight_layout()
            plt.xlim(0, num_landmarks)
            plt.savefig(f"figure_interps/{name}/{i}_{mode}_{vis_type}_barcode.{filetype}")
            plt.close()

    if preprocess:
        filename = "./results/mean_barcode/" + name + "_results.json"
        if not os.path.exists("./results/mean_barcode"):
            os.mkdir("./results/mean_barcode")
        with open(filename, "w+") as f:
            json.dump(results_dict, f)
