import torch

import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import gmean
from scipy.spatial.distance import pdist, squareform

class MultiViewGraph:
    def __init__(self, interaction_db, complex_db, adata, expressed_prop=0.15, precision="half", alpha=0.5, n_neighbors=5, quality_control = False, preserved_views=50):
        """
        Initialize the MultiViewGraph class with interaction database, complex database, and annotated data.

        Parameters:
        interaction_db (DataFrame): DataFrame containing ligand-receptor interactions.
        complex_db (DataFrame): DataFrame containing complex information for ligands and receptors.
        adata (AnnData): Annotated data matrix with spatial coordinates and expression data.
        expressed_prop (float): Proportion threshold to consider ligand or receptor as expressed.
        """
        self.interaction_db = interaction_db
        self.complex_db = complex_db
        self.adata = adata
        self.expressed_prop = expressed_prop
        self.expressed_lr_df = None
        self.multi_view_graph = None
        self.multi_view_graph_edge_index = None
        self.previous_multi_view_graph = []
        self.previous_multi_view_graph_edge_index = []
        self.precision = precision
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        
        self.quality_control = quality_control
        self.preserved_views = preserved_views
        
        self.init_MultiViewGraph()

    def init_MultiViewGraph(self):
        """
        Initialize the multi-view graph by calculating expressed ligand-receptor pairs,
        adjacency matrices, and edge indices.
        """
        print("Initializing Multi-View Graph...")
        self.get_expressed_lr_df()
        self.compute_adjacency_matrices()
        self.adjacency_matrices_to_edge_indices()

    def lr_expr_calculate(self, interaction_term, l_or_r, mean='gmean'):
        # Calculate expression levels of ligands or receptors.
        lr_term = self.interaction_db[self.interaction_db.interaction_name == interaction_term][l_or_r].item()
        if str(lr_term) == 'nan':
            return np.zeros([self.adata.shape[0]])

        if self.complex_db is not None and lr_term in self.complex_db.index:
            lr_list = list(self.complex_db.loc[lr_term])
            lr_list = [x for x in lr_list if str(x) != 'nan']
        else:
            lr_list = [lr_term]

        if mean == 'gmean':
            if len(set(lr_list) - set(self.adata.var_names)) != 0:
                lr_expr = np.zeros([self.adata.shape[0]])
            else:
                lr_expr = gmean(np.array(self.adata[:, lr_list].to_df()).T)
        else:
            used_lr = list(set(lr_list).intersection(set(self.adata.var_names)))
            lr_expr = np.array(self.adata[:, used_lr].to_df()).T.mean(0)
        return lr_expr

    def get_expressed_lr_df(self):
        """
        Calculate and store the DataFrame of expressed ligand-receptor pairs based on the expression proportion.
        Then select the top pairs based on expression intensity, limiting the selection to a maximum of 50 views.
        """
        print("Calculating expressed ligand-receptor pairs...")
        receptor_series_list = []
        ligand_series_list = []
        interaction_strength = []

        for interaction_term in tqdm(self.interaction_db.interaction_name, desc="Processing interactions"):
            receptor_expr = self.lr_expr_calculate(interaction_term, 'receptor')
            ligand_expr = self.lr_expr_calculate(interaction_term, 'ligand')

            receptor_series = pd.Series(receptor_expr, name=interaction_term)
            ligand_series = pd.Series(ligand_expr, name=interaction_term)

            receptor_series_list.append(receptor_series)
            ligand_series_list.append(ligand_series)

            # Calculate mean expression level for each ligand-receptor pair
            interaction_strength.append((interaction_term, np.mean(receptor_expr + ligand_expr)))

        receptor_df = pd.concat(receptor_series_list, axis=1)
        ligand_df = pd.concat(ligand_series_list, axis=1)

        # Determine expressed ligand-receptor pairs based on expression proportion
        r_pass = np.where(np.sum(receptor_df > 0, axis=0) > self.adata.shape[0] * self.expressed_prop)[0]
        l_pass = np.where(np.sum(ligand_df > 0, axis=0) > self.adata.shape[0] * self.expressed_prop)[0]
        passed_indices = list(set(r_pass).intersection(l_pass))
        used_interaction_db = self.interaction_db.iloc[passed_indices]

        # If quality control is enabled, select top interactions based on expression strength
        if self.quality_control:
            # Filter the interaction strength list to only include passed interactions
            filtered_interaction_strength = [pair for pair in interaction_strength if pair[0] in used_interaction_db['interaction_name'].values]
            # Select top interactions based on expression strength
            top_interactions = sorted(filtered_interaction_strength, key=lambda x: x[1], reverse=True)[:self.preserved_views]
            top_interaction_terms = [term for term, strength in top_interactions]
            # Filter to only include top interactions
            used_interaction_db = used_interaction_db[used_interaction_db['interaction_name'].isin(top_interaction_terms)]

        self.expressed_lr_df = used_interaction_db

    def compute_adjacency_matrices(self):
        """
        Compute and store adjacency matrices for all expressed ligand-receptor pairs.
        """
        print("Computing adjacency matrices...")
        n_spots = self.adata.shape[0]
        spatial_coords = self.adata.obsm['spatial']
        dist_mat = squareform(pdist(spatial_coords, metric='euclidean'))

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist_mat = 1 / dist_mat
            np.fill_diagonal(inv_dist_mat, 0)

        adjacency_matrices = {}

        for idx, row in tqdm(self.expressed_lr_df.iterrows(), desc="Building matrices", total=self.expressed_lr_df.shape[0]):
            interaction_term = row['interaction_name']
            ligand_expr = self.lr_expr_calculate(interaction_term, 'ligand')
            receptor_expr = self.lr_expr_calculate(interaction_term, 'receptor')
            lr_adjacency = np.outer(ligand_expr, receptor_expr) * inv_dist_mat
            if self.precision == "half":
                adjacency_matrices[interaction_term] = torch.from_numpy(lr_adjacency).clone().detach().half()
            else:
                adjacency_matrices[interaction_term] = torch.from_numpy(lr_adjacency).clone().detach().float()
        
        self.multi_view_graph = adjacency_matrices

    def adjacency_matrices_to_edge_indices(self):
        n_neighbors=self.n_neighbors
        """
        Convert adjacency matrices to edge indices for use in graph neural networks.
        """
        print("Converting adjacency matrices to edge indices...")
        edge_indices_dict = {}

        for interaction_term, adj_matrix in tqdm(self.multi_view_graph.items(), desc="Processing edge indices"):
            n_spots = adj_matrix.shape[0]

            sorted_values, sorted_indices = torch.sort(adj_matrix, descending=True)
            top_indices = sorted_indices[:, :n_neighbors]

            source_indices = torch.arange(n_spots).unsqueeze(1).repeat(1, n_neighbors).flatten()
            target_indices = top_indices.flatten()
            
            edge_indices = torch.stack([source_indices, target_indices], dim=0)
            edge_indices_dict[interaction_term] = edge_indices
    
            # values = torch.ones(source_indices.size(0))
            # sparse_edge_index = torch.sparse_coo_tensor(torch.stack([source_indices, target_indices]), values, (n_spots, n_spots))

            # edge_indices_dict[interaction_term] = sparse_edge_index

        self.multi_view_graph_edge_index = edge_indices_dict

    def update_data(self, new_data, alpha=0.5):
        """
        Update the class with new spot by gene matrix data.

        Parameters:
        new_data (AnnData): New annotated data matrix with spot by gene information.
        alpha (float): Blending factor for updating adjacency matrices (0 <= alpha <= 1).
        """
        print("Updating Multi-View Graph with new data...")
        self.adata = new_data
        
        self.previous_multi_view_graph.append(copy.deepcopy(self.multi_view_graph))
        self.previous_multi_view_graph_edge_index.append(copy.deepcopy(self.multi_view_graph_edge_index))

        # Keep a copy of the old views before updating
        old_views = set(self.multi_view_graph.keys())

        self.get_expressed_lr_df()

        old_adjacency_matrices = self.multi_view_graph
        self.compute_adjacency_matrices()

        print("Blending old and new adjacency matrices...")
        for interaction_term, new_adj_matrix in self.multi_view_graph.items():
            if interaction_term in old_adjacency_matrices:
                old_adj_matrix = old_adjacency_matrices[interaction_term]
                blended_matrix = alpha * old_adj_matrix + (1 - alpha) * new_adj_matrix
                self.multi_view_graph[interaction_term] = blended_matrix
            else:
                self.multi_view_graph[interaction_term] = new_adj_matrix

        # Ensure that old views are preserved
        for old_view in old_views:
            if old_view not in self.multi_view_graph:
                self.multi_view_graph[old_view] = old_adjacency_matrices[old_view]

        self.adjacency_matrices_to_edge_indices()
        
    def backup_data(self):
        
        self.multi_view_graph = self.previous_multi_view_graph[-1]
        self.multi_view_graph_edge_index = self.previous_multi_view_graph_edge_index[-1]
        
        self.previous_multi_view_graph = self.previous_multi_view_graph[:-1]
        self.previous_multi_view_graph_edge_index = self.previous_multi_view_graph_edge_index[:-1]        
        
