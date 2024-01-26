import torch
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np

class iMiracleTrainer:
    def __init__(self, mode, model, adata, initial_x, observed_x, device, precision="float"):
    
        assert mode in ["imputation", "clustering", "ranking"]
        self.mode = mode
        self.model = model
        self.adata = adata
        self.device = device
        self.precision = precision

        self.initial_x = self.convert_to_tensor(initial_x)
        self.observed_x = self.convert_to_tensor(observed_x)
        
        self.imputation_parameters_loaded = False 
        self.clustering_parameters_loaded = False 
        self.ranking_parameters_loaded = False
        
    def load_parameter_for_imputation(self, original_x, val_mask, test_mask, learning_rate=0.01, patience=10, omega=1, spatial_loss_epoch=1):
        
        assert self.mode == "imputation"

        self.original_x = self.convert_to_tensor(original_x)
        self.val_mask = self.convert_to_tensor(val_mask)
        self.test_mask = self.convert_to_tensor(test_mask)
        self.learning_rate = learning_rate
        self.patience = patience
        self.omega = omega
        self.spatial_loss_epoch = spatial_loss_epoch
        
        self.imputation_parameters_loaded = True
        
    def load_parameter_for_clustering(self, epoch = 100, learning_rate=0.01, patience=10, omega=1, spatial_loss_epoch=1):
        
        assert self.mode == "clustering"

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.patience = patience
        self.omega = omega
        self.spatial_loss_epoch = spatial_loss_epoch
        
        self.clustering_parameters_loaded = True 
        
    def load_parameter_for_ranking(self, epoch=1000, learning_rate=0.01, patience=10, threshold=0.001, omega=1, spatial_loss_epoch=1):
        
        assert self.mode == "ranking"

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.patience = patience
        self.threshold = threshold
        self.omega = omega
        self.spatial_loss_epoch = spatial_loss_epoch
        
        self.ranking_parameters_loaded = True 
        
    def convert_to_tensor(self, data):
        
        if self.precision == "half":
            dtype=torch.half
        else:
            dtype=torch.float
        
        if isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype).to(self.device)
        elif not isinstance(data, torch.Tensor):
            return torch.tensor(data.to_numpy(), dtype=dtype).to(self.device)
        
        return data.to(self.device)

    def train_model_for_imputation(self):
        
        assert self.mode == "imputation"
        if not self.imputation_parameters_loaded:
            raise RuntimeError("Parameters for imputation not loaded. Please run load_parameter_for_imputation first.")
        
        best_val_loss = float('inf')
        loss_stagnant_count = 0
        no_new_views = False

        # Create a mask for non-zero entries
        non_zero_mask = self.observed_x != 0
    
        print("Starting training...")
    
        while not no_new_views:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            epoch = 0
            while(1):
                # Switch back to training mode
                self.model.train()
                
                epoch += 1
                optimizer.zero_grad()
                
                if epoch%self.spatial_loss_epoch == 0:
                    x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x, compute_spatial_loss=True)
                else:
                    x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x)
    
                # Apply mask to both reconstructed and original data
                masked_reconstructed = x_reconstructed[non_zero_mask]
                masked_original = self.original_x[non_zero_mask]
    
                # Compute loss only on non-zero entries
                gene_loss = F.mse_loss(masked_reconstructed, masked_original)
                
                if epoch%self.spatial_loss_epoch == 0:
                    total_loss = gene_loss + self.omega*spatial_loss_total
                else:
                    total_loss = gene_loss
                    
                total_loss.backward()
                optimizer.step()

                # Switch to evaluation mode for validation
                self.model.eval()
                with torch.no_grad():
                    masked_val_reconstructed = x_reconstructed[self.val_mask]
                    masked_val_original = self.original_x[self.val_mask]
                    val_loss = F.mse_loss(masked_val_reconstructed, masked_val_original)

                if epoch % self.spatial_loss_epoch == 0:
                    print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}, Spatial Loss: {spatial_loss_total.item()}, Val Loss: {val_loss.item()}, Best Val Loss: {best_val_loss}")
                else:
                    print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}, Val Loss: {val_loss.item()}, Best Val Loss: {best_val_loss}")
                    
                # Early stopping condition - fixed
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state = self.model.state_dict()
                    loss_stagnant_count = 0  # Reset only when a new best is found
                else:
                    loss_stagnant_count += 1

                if loss_stagnant_count >= self.patience:
                    if not self.model.iterative:
                        print(f"Loss stagnant for {self.patience} epochs. Break.")
                        break
                    else:
                        print(f"Loss stagnant for {self.patience} epochs. Initiating update procedure.")
                        previous_view_count = len(self.model.mvG.multi_view_graph_edge_index)
                        new_adata = self.adata.copy()
                        new_adata.X = x_reconstructed.cpu().detach().numpy()
                        self.model.mvG.update_data(new_adata)
                        if not self.model.share_gnn:
                            self.model.update_gnns()
                        if not self.model.share_decoder:
                            self.model.update_decoders()
                        current_view_count = len(self.model.mvG.multi_view_graph_edge_index)
                        print("previous_view_count:",previous_view_count)
                        print("current_view_count:",current_view_count)
                        no_new_views = previous_view_count == current_view_count
                        if no_new_views:
                            print("No new views added. Training will continue until completion.")
                            self.model.mvG.backup_data()
                        else:
                            print(f"New views added. Total views: {current_view_count}. Resuming training.")
                        best_val_loss = float('inf')
                        loss_stagnant_count = 0  # Reset counter after update
                        break  # Restart training loop after update

        self.model.load_state_dict(best_state)
    
        print("Training completed.")
        return best_val_loss

    def evaluate_for_imputation(self):
        assert self.mode == "imputation"
        with torch.no_grad():
            x_reconstructed, _ = self.model.forward(self.initial_x)

            test_mask = self.test_mask.clone().detach()
            original_x = self.original_x.clone().detach()
            masked_original = original_x[test_mask]
            masked_reconstructed = x_reconstructed[test_mask]

            l1_distance = mean_absolute_error(masked_original.cpu(), masked_reconstructed.cpu().detach().numpy())
            cos_sim = cosine_similarity(masked_original.cpu().unsqueeze(0), masked_reconstructed.cpu().detach().unsqueeze(0))[0][0]
            rmse = mean_squared_error(masked_original.cpu(), masked_reconstructed.cpu().detach().numpy(), squared=False)

        return {"L1 Distance": l1_distance, "Cosine Similarity": cos_sim, "RMSE": rmse}
    
    def train_model_for_clustering(self):
        
        assert self.mode == "clustering"
        if not self.clustering_parameters_loaded:
            raise RuntimeError("Parameters for clustering not loaded. Please run load_parameter_for_clustering first.")
        
        # Create a mask for non-zero entries
        non_zero_mask = self.observed_x != 0
    
        print("Starting training...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(1, self.epoch+1):
            # Switch back to training mode
            self.model.train()
            optimizer.zero_grad()
            
            if epoch%self.spatial_loss_epoch == 0:
                x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x, compute_spatial_loss=True)
            else:
                x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x)

            # Apply mask to both reconstructed and original data
            if self.precision == "half":
                masked_reconstructed = x_reconstructed[non_zero_mask].half()
                masked_observed = self.observed_x[non_zero_mask].half()
            else:
                masked_reconstructed = x_reconstructed[non_zero_mask].float()
                masked_observed = self.observed_x[non_zero_mask].float()

            # Compute loss only on non-zero entries
            gene_loss = F.mse_loss(masked_reconstructed, masked_observed)
            
            if epoch%self.spatial_loss_epoch == 0:
                total_loss = gene_loss + self.omega*spatial_loss_total
            else:
                total_loss = gene_loss
                
            total_loss.backward()
            optimizer.step()

            if epoch % self.spatial_loss_epoch == 0:
                print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}, Spatial Loss: {spatial_loss_total.item()}")
            else:
                print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}")
    
        print("Training completed.")
        
    def train_model_for_ranking(self):
        
        assert self.mode == "ranking"
        if not self.ranking_parameters_loaded:
            raise RuntimeError("Parameters for ranking not loaded. Please run load_parameter_for_ranking first.")
        
        no_new_views = False

        # Create a mask for non-zero entries
        non_zero_mask = self.observed_x != 0
    
        print("Starting training...")
    
        while not no_new_views:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            train_loss_history = []
            for epoch in range(1, self.epoch+1):
                # Switch back to training mode
                self.model.train()
                optimizer.zero_grad()
                
                # if epoch%self.spatial_loss_epoch == 0:
                #     x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x, compute_spatial_loss=True)
                # else:
                x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x)
    
                # Apply mask to both reconstructed and original data
                if self.precision == "half":
                    masked_reconstructed = x_reconstructed[non_zero_mask].half()
                    masked_observed = self.observed_x[non_zero_mask].half()
                else:
                    masked_reconstructed = x_reconstructed[non_zero_mask].float()
                    masked_observed = self.observed_x[non_zero_mask].float()
    
                # Compute loss only on non-zero entries
                gene_loss = F.mse_loss(masked_reconstructed, masked_observed)
                
                if epoch%self.spatial_loss_epoch == 0:
                    total_loss = gene_loss + self.omega*spatial_loss_total
                else:
                    total_loss = gene_loss
                    
                train_loss_history.append(total_loss.item())
                total_loss.backward()
                optimizer.step()

                # if epoch % self.spatial_loss_epoch == 0:
                #     print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}, Spatial Loss: {spatial_loss_total.item()}")
                # else:
                print(f"Epoch {epoch}, Current Train Gene Loss: {gene_loss.item()}")

                if epoch >= self.patience:
                    recent_losses = train_loss_history[-self.patience:]
                    if max(recent_losses) - min(recent_losses) < self.threshold:
                        print(f"Training stopped early at epoch {epoch}. Loss improvement less than {self.threshold} for {self.patience} consecutive epochs.")
                        break
            if not self.model.iterative:
                print(f"Loss stagnant for {self.patience} epochs. Break.")
                break
            else:
                previous_view_count = len(self.model.mvG.multi_view_graph_edge_index)
                new_adata = self.adata.copy()
                new_adata.X = x_reconstructed.cpu().detach().numpy()
                self.model.mvG.update_data(new_adata)
                if not self.model.share_gnn:
                    self.model.update_gnns()
                if not self.model.share_decoder:
                    self.model.update_decoders()
                current_view_count = len(self.model.mvG.multi_view_graph_edge_index)
                print("previous_view_count:",previous_view_count)
                print("current_view_count:",current_view_count)
                no_new_views = previous_view_count == current_view_count
                if no_new_views:
                    print("No new views added. Training will continue until completion.")
                    self.model.mvG.backup_data()
                else:
                    print(f"New views added. Total views: {current_view_count}. Resuming training.")
    
        print("Training completed.")
        
        self.save_model_outputs()

    def save_model_outputs(self):
        """Saves the model outputs"""
        self.model.eval()
        self.model.forward(self.initial_x, save_outputs=True)
    
    def test(self):
        self.model.eval()
        # Create a mask for non-zero entries
        non_zero_mask = self.observed_x != 0
        x_reconstructed, spatial_loss_total = self.model.forward(self.initial_x)
        masked_reconstructed = x_reconstructed[non_zero_mask].float()
        masked_observed = self.observed_x[non_zero_mask].float()
        gene_loss = F.mse_loss(masked_reconstructed, masked_observed)
        print(f"Test: Current Train Gene Loss: {gene_loss.item()}")