import os
import disentanglement.libs.disentanglement.gs as gs
import numpy as np
import torch
import json
from tqdm import tqdm
import time
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import pdb

from disentanglement.libs.disentanglement.utils import get_dataset_args, load_models, sample_noise, visualize


def covar(results_dict, name, gs_results_dir, verbose=False, real=False):
    diffs = []
    cur_results_dict = results_dict
    target_results_dict = results_dict
    if real:
        print("Cannot load real dataset")
    for cur_factor, cur_factor_dict in cur_results_dict.items():
        for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
            cur_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                for target_value, target_barcode in sorted(target_factor_dict.items()):
                    score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                    cur_diffs.append(score)
                    if verbose:
                        print(score, end=",")
            if verbose:
                print()
            diffs.append(cur_diffs)
    data = np.asarray(diffs)
    plt.matshow(data, cmap=plt.cm.Blues)
    plt.savefig(f"{gs_results_dir}/{name}.png")
    plt.close()
    return data


def agg_covar(results_dict, ones_only, name, gs_results_dir, real=False):
    agg_diffs = []
    cur_results_dict = results_dict
    if real:
        print("Cannot load real dataset")
    target_results_dict = results_dict
    if ones_only:
        for cur_factor, cur_factor_dict in cur_results_dict.items():
            cur_agg_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
                    for target_value, target_barcode in sorted(target_factor_dict.items()):
                        if int(cur_value) == 1 and int(target_value) == 1:
                            score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                            cur_agg_diffs.append(score)
            agg_diffs.append(cur_agg_diffs)
    else:
        for cur_factor, cur_factor_dict in cur_results_dict.items():
            cur_agg_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                factor_avg = 0
                for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
                    for target_value, target_barcode in sorted(target_factor_dict.items()):
                        score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                        factor_avg += score
                factor_avg /= len(list(target_factor_dict.items())) * len(list(cur_factor_dict.items()))
                cur_agg_diffs.append(factor_avg)
            agg_diffs.append(cur_agg_diffs)
    agg_diffs = np.asarray(agg_diffs)
    rev_diffs = 1 - agg_diffs
    plt.matshow(rev_diffs, cmap=plt.cm.Blues)
    plt.savefig(f"{gs_results_dir}/agg_{name}.png")
    plt.close()
    return agg_diffs, rev_diffs


def bicluster_mean(cocluster, data, n_clust, real=False):
    if real:
        print("Cannot load real dataset")

    else:
        sorted_idx_row = np.argsort(cocluster.row_labels_)
        sorted_idx_col = np.argsort(cocluster.column_labels_)
        sorted_data = data[sorted_idx_row]
        sorted_data = sorted_data[:, sorted_idx_col]

        prev = None
        bounds = []
        sorted_labels = cocluster.row_labels_[sorted_idx_row]
        for i, c in enumerate(sorted_labels):
            if prev != c:
                bounds.append(i)
            prev = c
        bounds.append(len(sorted_labels))

        # Collapse rows
        avg_row_means = []
        for i in range(n_clust):
            points = data[bounds[i]:bounds[i + 1]]
            mean = points.mean(axis=0)
            avg_row_means.append(mean)

        prev = None
        bounds = []
        sorted_labels = cocluster.column_labels_[sorted_idx_col]
        for i, c in enumerate(sorted_labels):
            if prev != c:
                bounds.append(i)
            prev = c
        bounds.append(len(sorted_labels))

        # Collapse cols
        avg_data = []
        avg_row_means = np.array(avg_row_means)
        for i in range(n_clust):
            points = avg_row_means[:, bounds[i]:bounds[i + 1]]
            mean = points.mean(axis=1)
            avg_data.append(mean)

        avg_data = np.array(avg_data)
        return avg_data


def total_variance(i, cocluster, data, real=False):
    rows, cols = cocluster.get_indices(i)
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]

    denom_in = len(rows) * len(cols)
    denom_out = len(col_complement) * len(rows)
    if denom_in == 0 or denom_out == 0:
        # Skip because no correspondence
        if real:
            return 0
        else:
            print('Denom should not be 0 in unsupervised case')
            pdb.set_trace()

    # Get sum of values inside of cluster
    in_sum = data[rows][:, cols].sum()

    # Get sum of values outside of cluster
    out_sum = data[rows][:, col_complement].sum()

    in_norm = in_sum / denom_in
    out_norm = out_sum / denom_out

    in_var = data[rows][:, cols].std() / denom_in
    out_var = data[rows][:, col_complement].std() / denom_out
    score = in_var ** 2 + out_var ** 2
    return score


def bicluster_score(i, cocluster, data, real=False):
    rows, cols = cocluster.get_indices(i)
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]

    denom_in = len(rows) * len(cols)
    denom_out = len(col_complement) * len(rows)
    if denom_in == 0 or denom_out == 0:
        # Skip because no correspondence
        if real:
            return 0
        else:
            print('Denom should not be 0 in unsupervised case')
            pdb.set_trace()

    # Get sum of values inside of cluster
    in_sum = data[rows][:, cols].sum()

    # Get sum of values outside of cluster
    out_sum = data[rows][:, col_complement].sum()

    in_norm = in_sum / denom_in
    out_norm = out_sum / denom_out

    score = (out_norm - in_norm)
    return score


def bicluster(data, n_clust, name, gs_results_dir):
    model = SpectralCoclustering(n_clusters=n_clust, random_state=0)
    cluster = model.fit(data)

    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title(f"{name} with {n_clust} coclusters")
    plt.savefig(f"{gs_results_dir}/cocluster/agg_{n_clust}_{name}.png")
    plt.close()

    print(f"Saved cocluster png to {gs_results_dir}/cocluster/agg_{n_clust}_{name}.png")
    return cluster


def interpolate(args, z2s):
    # Given z dim num, name for the corresponding factor (for filename), generate latent interpolations using decoder

    decoder_params = {'dataset_name': args.dataset_name}
    real_dataset, ns, image_shape, npix, nc, ncls, factor_id2name = get_dataset_args(args,
                                                                                     return_factor_name_map='celeba' in args.dataset_name)
    decoder = load_models(
        args, ns, npix, nc, ncls,
        model_types=["decoder"],
        model_params=[decoder_params],
        model_ckpts=[args.decoder_ckpt]
    )
    decoder.eval()

    args.viz_batch_size = 8
    fixed_zz = sample_noise(args.viz_batch_size, args.nz, args.device)
    fixed_zs = []
    n_view = 1
    for dim in range(args.nz):
        fixed_z = np.tile(np.random.randn(n_view, args.nz), (args.viz_batch_size, n_view)).astype(np.float32)
        fixed_z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, args.viz_batch_size))
        fixed_zs.append(torch.from_numpy(fixed_z))

    fakes = []
    for iz, fixed_z in enumerate(fixed_zs):
        with torch.no_grad():
            fake = decoder(fixed_z).detach().cpu()

            # Save individual ones with factor names - easier to inspect b/c can't write text and don't know groupings
            s_num = z2s[iz]
            if 'celeba' in args.dataset_name:
                factor_name = factor_id2name[s_num]
            else:
                dsprites_factor_map = ['shape', 'scale', 'orient', 'xpos', 'ypos']
                factor_name = dsprites_factor_map[s_num]
            i_save_path = Path(args.gs_results_dir) / 'interpolations' / f'{args.name}_{factor_name}_z{iz}_s{s_num}.png'
            visualize(fake, i_save_path)

            fakes.append(fake)

    for iviz in range(10):
        fixed_zz = sample_noise(args.viz_batch_size, args.nz, args.device)
        fixed_zs = []
        for dim in range(args.nz):
            fixed_z = np.tile(np.random.randn(1, args.nz), (args.viz_batch_size, 1)).astype(np.float32)
            fixed_z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, args.viz_batch_size))
            fixed_zs.append(torch.from_numpy(fixed_z))

        fakes = []
        for iz, fixed_z in enumerate(fixed_zs):
            with torch.no_grad():
                fake = decoder(fixed_z).detach().cpu()

                # Save individual ones with factor names - easier to inspect b/c can't write text and don't know groupings
                s_num = z2s[iz]
                if 'celeba' in args.dataset_name:
                    factor_name = factor_id2name[s_num]
                else:
                    dsprites_factor_map = ['shape', 'scale', 'orient', 'xpos', 'ypos']
                    factor_name = dsprites_factor_map[s_num]
                i_save_path = Path(
                    args.gs_results_dir) / 'interpolations' / f'{args.name}_{factor_name}_z{iz}_s{s_num}_{iviz}.png'
                i_save_path = f'{args.name}_{factor_name}_z{iz}_s{s_num}_{iviz}.png'
                visualize(fake, i_save_path)

                fakes.append(fake)

        # Save concatenated full one
        fakes_concat = torch.cat(fakes, 0)
        save_path = Path(args.gs_results_dir) / 'interpolations' / f'{args.name}_match2s_{iviz}.png'
        visualize(fakes_concat, save_path)


def preprocess_wbary(filename, name, plot=True):
    original_filename = filename.replace('mean_', '')
    print(f'Reading preprocessed RLTs at {original_filename}')

    with open(original_filename, "r") as f:
        results_dict = json.load(f)

    if plot:
        vis_folder = f"./results/wbary_vis/{name}"
        os.makedirs(vis_folder, exist_ok=True)
        print(f'Saving vis to {vis_folder}')
    print(len(results_dict))
    for i, (key, relevant) in tqdm(enumerate(results_dict.items())):
        for val, barcode in relevant.items():
            cur_code = gs.barymean(np.asarray(barcode))
            results_dict[key][val] = cur_code.tolist()
            if plot:
                plt.bar(np.arange(len(cur_code)), cur_code, alpha=0.2, color=f'C{i}')
                plt.savefig(f"{vis_folder}/{i}_barcode")
        if plot:
            plt.close()

    with open(filename, "w") as f:
        json.dump(results_dict, f)
    print(f'Saved Wbary barcodes to {filename}')
    return results_dict


def compare_barcodes(rlt_filename, name, results_dir, plot=True):
    ones_only = False
    do_bicluster = True
    search_n_clusters = True
    save_scores = True
    if os.path.exists(rlt_filename):
        with open(rlt_filename, "r") as f:
            results_dict = json.load(f)
    else:
        # Preprocess W-barycenter RLTs
        results_dict = preprocess_wbary(rlt_filename, name, plot)

    diffs = covar(results_dict, name, "results")
    agg_diffs, rev_diffs = agg_covar(results_dict, ones_only, name, results_dir, real=False)

    if do_bicluster:
        num_latents = len(results_dict)
        print('num_latents: ', num_latents)
        unsup_scores = {}
        if search_n_clusters:
            n_clusters = list(range(2, num_latents + 1))
        else:
            n_clusters = [min(5, num_latents)]

        overall_scores, overall_vars = [], []
        for n_clust in n_clusters:
            # Use reversed diffs: Higher the value the better for inside a factor, lower better for outside a factor
            cluster = bicluster(rev_diffs, n_clust, name, results_dir)
            avg_cluster = bicluster_mean(cluster, rev_diffs, n_clust, real=False)
            if plot:
                np.save(f'{results_dir}/avg_cluster_np/{name}.npy', avg_cluster)

            # Supervised
            row_matches, col_matches = linear_sum_assignment(avg_cluster)
            match_dists = np.array(avg_cluster)[row_matches, col_matches]
            score = match_dists.sum() / avg_cluster.shape[1]  # This is num real factors
            # sorted_match_dists = match_dists[:, col_matches]
            if plot:
                plt.matshow(avg_cluster[:, col_matches], cmap=plt.cm.Blues)
                plt.title(f"Averaged {name} {n_clust} coclusters")
                plt.gcf().text(.02, .01, f'{score}', fontsize=8)
                plt.savefig(f"{results_dir}/cocluster/average_{n_clust}_{name}.png")
                plt.close()

                print(
                    f"Saved averaged fake-real cocluster png to {results_dir}/cocluster/average_{n_clust}_{name}.png")

                # Save full thing
                plt.matshow(avg_cluster, cmap=plt.cm.Blues)
                real_n_clust = 40
                plt.title(f"Averaged {name} {n_clust}x{real_n_clust} coclusters")
                plt.gcf().text(.02, .01, f'{score}', fontsize=8)
                plt.savefig(f"{results_dir}/cocluster/full_average_{n_clust}_{name}.png")
                plt.close()

                print(
                    f"Saved averaged real cocluster png to {results_dir}/cocluster/full_average_{n_clust}_{name}.png")

            bicluster_scores = [bicluster_score(i, cluster, agg_diffs, real=False) for i in range(n_clust)]
            bicluster_vars = [total_variance(i, cluster, agg_diffs, real=False) for i in range(n_clust)]

            overall_score = np.mean(bicluster_scores) * 10000
            overall_var = np.mean(bicluster_vars) * 10000

            overall_vars.append(overall_var)
            overall_scores.append(overall_score)
            print(f"{n_clust} Bicluster sum: {overall_score} with var {overall_var}")
            unsup_scores[n_clust] = overall_score

        # Select minimum variance n_clust, and get the overall score
        final_score = overall_scores[np.argmin(overall_vars)]
        final_var = overall_vars[np.argmin(overall_vars)]
        final_n_clust = n_clusters[np.argmin(overall_vars)]
        print(f'Final score is: {final_score} with {final_n_clust} clusters and variance {final_var}')

        if save_scores:
            # Write scores to file
            timestamp = str(time.time()).replace('.', '')
            supervision = 'unsupervised'

            row = {
                'score': final_score,
                'n_clust': final_n_clust,
                'var': final_var,
                'type': supervision,
                'run_name': name,
                'timestamp': timestamp,
            }

            df = pd.DataFrame(row, index=[0])
            df_col_order = ['score', 'n_clust', 'var', 'type', 'timestamp', 'run_name','timestamp']
            df = df[df_col_order]

            scores_file = f'{results_dir}/scores/all.csv'

            if os.path.exists(scores_file):
                df.to_csv(scores_file, mode='a', header=False, index=False)
            else:
                print(f'Creating scores file {scores_file}...')
                df.to_csv(scores_file, header=True, index=False)


if __name__ == "__main__":
    name = "medium"
    rlt_filename = "./results/mean_barcode/" + name + "_results.json"
    results_dir = "results/covar"
    compare_barcodes(rlt_filename, name, results_dir)
