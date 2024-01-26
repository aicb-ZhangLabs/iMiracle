import torch

import pickle
import numpy as np
import pandas as pd
import scanpy as sc

import warnings
from numba import NumbaDeprecationWarning

# Suppress specific deprecation warnings from Numba
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

def load_data_for_imputation(platform="10xVisium", sample_number=151507, split_number=0):

    if platform == "10xVisium":
        data_dir="/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC"
        lr_db_path="/home/zihend1/iMiracle/Data/LR_DB/Human/"
        pre_resolution=1
        adata = sc.read_h5ad(data_dir+"/Preprocessed/"+str(sample_number)+"/filtered_adata.h5ad")
        
        # # Run PCA
        sc.tl.pca(adata, svd_solver='arpack')
        # Compute the neighborhood graph
        sc.pp.neighbors(adata)
        # Run Louvain clustering
        sc.tl.leiden(adata, resolution=pre_resolution, key_added='leiden_label')
        # Convert 'leiden_label' to one-hot encoded feature
        initial_x = pd.get_dummies(adata.obs['leiden_label'], prefix='leiden')
        
        # initial_x = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/initial_x.npy")
        with open(data_dir+"/Preprocessed/"+str(sample_number)+"/sample_grn.pkl", 'rb') as f:
            grn = pickle.load(f)

        interaction_df = pd.read_csv(lr_db_path+"interaction_df.csv")
        complex_df = pd.read_csv(lr_db_path+"complex_df.csv")
        # cofactor_df = pd.read_csv(lr_db_path+"cofactor_df.csv")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.from_numpy(adata.X.todense().astype(np.float32))
        original_x = x.clone().to(device)  # Save the original features
        val_mask = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/split_"+str(split_number)+"_val_mask.npz")['arr_0']
        test_mask = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/split_"+str(split_number)+"_test_mask.npz")['arr_0']
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        x[val_mask] = 0
        x[test_mask] = 0
        
        adata.X = x
    
    elif platform == "Stereoseq":
        data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/Stereoseq/MouseOlfactoryBulb"
        lr_db_path="/home/zihend1/iMiracle/Data/LR_DB/Mouse/"
        pre_resolution=1
        adata = sc.read_h5ad(data_dir+"/Preprocessed/filtered_adata.h5ad")
        
        # # Run PCA
        sc.tl.pca(adata, svd_solver='arpack')
        # Compute the neighborhood graph
        sc.pp.neighbors(adata)
        # Run Louvain clustering
        sc.tl.leiden(adata, resolution=pre_resolution, key_added='leiden_label')
        # Convert 'leiden_label' to one-hot encoded feature
        initial_x = pd.get_dummies(adata.obs['leiden_label'], prefix='leiden')
        
        grn = None

        interaction_df = pd.read_csv(lr_db_path+"interaction_df.csv")
        complex_df = pd.read_csv(lr_db_path+"complex_df.csv")
        # cofactor_df = pd.read_csv(lr_db_path+"cofactor_df.csv")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.from_numpy(adata.X.astype(np.float32))
        original_x = x.clone().to(device)  # Save the original features
        val_mask = np.load(data_dir+"/Preprocessed/split_"+str(split_number)+"_val_mask.npz")['arr_0']
        test_mask = np.load(data_dir+"/Preprocessed/split_"+str(split_number)+"_test_mask.npz")['arr_0']
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        x[val_mask] = 0
        x[test_mask] = 0
        
        adata.X = x
        
    elif platform == "SlideseqV2":
        data_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/SlideseqV2/MouseOlfactoryBulb"
        lr_db_path="/home/zihend1/iMiracle/Data/LR_DB/Mouse/"
        pre_resolution=1
        adata = sc.read_h5ad(data_dir+"/Preprocessed/filtered_adata.h5ad")
        
        # # Run PCA
        sc.tl.pca(adata, svd_solver='arpack')
        # Compute the neighborhood graph
        sc.pp.neighbors(adata)
        # Run Louvain clustering
        sc.tl.leiden(adata, resolution=pre_resolution, key_added='leiden_label')
        # Convert 'leiden_label' to one-hot encoded feature
        initial_x = pd.get_dummies(adata.obs['leiden_label'], prefix='leiden')
        
        grn = None

        interaction_df = pd.read_csv(lr_db_path+"interaction_df.csv")
        complex_df = pd.read_csv(lr_db_path+"complex_df.csv")
        # cofactor_df = pd.read_csv(lr_db_path+"cofactor_df.csv")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.from_numpy(adata.X.astype(np.float32))
        original_x = x.clone().to(device)  # Save the original features
        val_mask = np.load(data_dir+"/Preprocessed/split_"+str(split_number)+"_val_mask.npz")['arr_0']
        test_mask = np.load(data_dir+"/Preprocessed/split_"+str(split_number)+"_test_mask.npz")['arr_0']
        val_mask = torch.tensor(val_mask)
        test_mask = torch.tensor(test_mask)
        x[val_mask] = 0
        x[test_mask] = 0
        
        adata.X = x 
    
    else:
        raise ValueError("Unsupported platform. Choose '10xVisium', 'Stereoseq' or 'SlideseqV2'.")
    
    return adata, interaction_df, complex_df, initial_x, grn, original_x, val_mask, test_mask
        
def load_data_for_clustering(platform="10xVisium", sample_number=151507):
    
    if platform == "10xVisium":
        data_dir="/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC"
        lr_db_path="/home/zihend1/iMiracle/Data/LR_DB/Human/"
        sample_dir = "/extra/zhanglab0/SpatialTranscriptomicsData/10XVisium/DLPFC/" + str(sample_number)
        adata = sc.read_visium(path=sample_dir, count_file=str(sample_number)+'_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False, max_value=10)
        
        Ann_df = pd.read_csv(f"{sample_dir}_truth.txt", sep='\t', header=None, index_col=0)
        # Ann_df.columns = ['Ground Truth']
        adata.obs['Ground Truth'] = Ann_df
        
        pre_resolution = 1
        # # Run PCA
        sc.tl.pca(adata, svd_solver='arpack')
        # Compute the neighborhood graph
        sc.pp.neighbors(adata)
        # Run Louvain clustering
        sc.tl.leiden(adata, resolution=pre_resolution, key_added='leiden_label')
        # Convert 'leiden_label' to one-hot encoded feature
        initial_x = pd.get_dummies(adata.obs['leiden_label'], prefix='leiden')
        
        # initial_x = np.load(data_dir+"/Preprocessed/"+str(sample_number)+"/initial_x.npy")
        with open(data_dir+"/Preprocessed/"+str(sample_number)+"/sample_grn.pkl", 'rb') as f:
            grn = pickle.load(f)

        interaction_df = pd.read_csv(lr_db_path+"interaction_df.csv")
        complex_df = pd.read_csv(lr_db_path+"complex_df.csv")
        # cofactor_df = pd.read_csv(lr_db_path+"cofactor_df.csv")
    
    else:
        raise ValueError("Unsupported platform. Choose '10xVisium'.")
    
    return adata, interaction_df, complex_df, initial_x, grn
    