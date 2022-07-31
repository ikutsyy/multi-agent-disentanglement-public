from disentanglement.manifold_evaluation import generate_covariance, generate_barcodes
from disentanglement.manifold_evaluation import generate_rlt

if __name__ == '__main__':
    name = "large"
    params = {
        "obs_size": 16,
        "num_agents": 5,
        "message_size":10,
        "num_targets":3,
        "results_dir": "./results",
        "results_file": "./results/barcode/" + name + "_results.json",
    }
    z_size = 4
    print("Comparing Embedding Spaces...")
    # generate_barcodes.compare_embedding_spaces_fake(params, z_size, 100, 1 / 128, name,agent=True, plot=True)

    mean_results_dir = "./results/mean_barcode/" + name + "_results.json"

    print("Computing RLT...")
    generate_rlt.generate_rlt(preprocessed_barcode_path=mean_results_dir, name=name, load_preprocessed=True,
                              preprocess=True)

    covar_results_dir = "results/covar"
    mean_results_dir = "./results/mean_barcode/" + name + "_results.json"

    print("Comparing barcodes...")
    generate_covariance.compare_barcodes(mean_results_dir, name, covar_results_dir)
