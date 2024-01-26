import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
       
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type, num_layers=2, dropout=0.5):
        super(GNNStack, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            if gnn_type == 'GCN':
                self.layers.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.layers.append(GATConv(in_dim, hidden_dim))
            elif gnn_type == 'SAGE':
                self.layers.append(SAGEConv(in_dim, hidden_dim))
            elif gnn_type == 'Transformer':
                self.layers.append(TransformerConv(in_dim, hidden_dim))
            else:
                raise ValueError("Unsupported GNN type. Choose 'GCN', 'GAT', 'SAGE' or 'Transformer'.")

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x
    

class iMiracle(nn.Module):
    def __init__(self, mvG, input_dim, hidden_dim, output_dim, device, gnn_type='GCN', gnn_layers=2, share_gnn=False, share_decoder=False, iterative=True, GRN=None):
        super(iMiracle, self).__init__()
        self.mvG = mvG
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        self.share_gnn = share_gnn
        self.share_decoder = share_decoder
        self.iterative = iterative

        self.device = device

        # MLP for basic gene expression decoupling
        self.basic_mlp = self.create_mlp(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)

        if self.share_gnn:
            self.gnn = self.create_gnn().to(self.device)
        else:
            self.gnns = nn.ModuleDict({interaction_term: self.create_gnn()
                                       for interaction_term in mvG.multi_view_graph})

        if self.share_decoder:
            self.decoder = self.create_mlp(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        else:
            self.decoders = nn.ModuleDict({interaction_term: self.create_mlp(input_dim=self.hidden_dim,\
                                                                             hidden_dim=self.hidden_dim,\
                                                                             output_dim=self.output_dim)
                                       for interaction_term in mvG.multi_view_graph})
        
        self.GRN = GRN
        
        self.spatial_coords = torch.tensor(self.mvG.adata.obsm['spatial'], dtype=torch.float).to(self.device)
        self.spatial_distance = torch.cdist(self.spatial_coords, self.spatial_coords, p=2).float().to(self.device)

    def create_mlp(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )#.half()

    def create_gnn(self):
        # Create a two-layer GNN based on the specified type
        return GNNStack(self.input_dim, self.hidden_dim, self.gnn_type, self.gnn_layers)#.half()
    
    def normalize_distances(self, distances):
        min_dist = distances.min()
        max_dist = distances.max()
        normalized_dist = (distances - min_dist) / (max_dist - min_dist)
        return normalized_dist

    def spatial_loss(self, embeddings):
        pairwise_distance = torch.cdist(embeddings, embeddings, p=2)
        norm_pairwise_distance = self.normalize_distances(pairwise_distance)
        norm_spatial_distance = self.normalize_distances(self.spatial_distance)
        loss = F.mse_loss(norm_pairwise_distance, norm_spatial_distance)
        return loss


    def forward(self, x_initial, save_outputs=False, compute_spatial_loss=False):
        
        # x_initial = x_initial.to(dtype=torch.float16)
        x_basic = self.basic_mlp(x_initial)  # Basic gene expression
        x_total = x_basic.clone()
        # x_total = torch.zeros_like(x_basic)

        # Optionally save basic gene expression
        if save_outputs:
            self.saved_basic_gene_expression = x_basic.cpu().detach().numpy()

         # Apply GNN for each view and aggregate the results, saving outputs if requested
        self.saved_H_lr = {} if save_outputs else None
        self.saved_X_lr = {} if save_outputs else None
        self.saved_X_lr_GRN = {} if save_outputs else None
        
        spatial_loss_total = 0
    
        # Apply GNN for each view and aggregate the results
        for interaction_term, edge_index in self.mvG.multi_view_graph_edge_index.items():
        # for interaction_term, adj in self.mvG.multi_view_graph.items():
            
            # print(interaction_term)
            
            # edge_index = adj
            if self.share_gnn:
                gnn = self.gnn.to(self.device)
            else:
                gnn = self.gnns[interaction_term].to(self.device)
                
            if self.share_decoder:
                decoder = self.decoder.to(self.device)
            else:
                decoder = self.decoders[interaction_term].to(self.device)
                
            H_lr = gnn(x_initial, edge_index.to(self.device))#.half()
            
            if compute_spatial_loss:
                spatial_loss_total += self.spatial_loss(H_lr)
            
            X_lr = decoder(H_lr)#.half()
            
            if self.GRN is not None and interaction_term in self.GRN:
                X_lr_GRN = X_lr * torch.tensor(self.GRN[interaction_term]).to(self.device)
            else:
                X_lr_GRN = X_lr
                
            x_total += X_lr_GRN

             # Save the output of each view
            if save_outputs:
                self.saved_H_lr[interaction_term] = H_lr.cpu().detach().numpy()
                self.saved_X_lr[interaction_term] = X_lr.cpu().detach().numpy()
                self.saved_X_lr_GRN[interaction_term] = X_lr_GRN.cpu().detach().numpy()
    
        return x_total, spatial_loss_total
        
    def update_gnns(self):
        # Update or add GNNs for new views
        for interaction_term in self.mvG.multi_view_graph:
            if interaction_term not in self.gnns:
                self.gnns[interaction_term] = self.create_gnn()

    def update_decoders(self):
        # Update or add decoders for new views
        for interaction_term in self.mvG.multi_view_graph:
            if interaction_term not in self.decoders:
                self.decoders[interaction_term] = self.create_mlp(input_dim=self.hidden_dim,hidden_dim=self.hidden_dim,output_dim=self.output_dim)
