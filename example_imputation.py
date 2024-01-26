import torch
import argparse
from data import load_data_for_imputation
from multiview_graph import MultiViewGraph
from neural_networks import iMiracle
from train_utils import iMiracleTrainer

parser = argparse.ArgumentParser()
    
# For the data platform and sample
parser.add_argument('--data_platform', type=str, default='10XVisium')
parser.add_argument('--sample_number', type=int, default=151507)

args = parser.parse_args()

# for s in [151507,151508,151509,151510,151669,151670,151671,151672,151673,151674,151675,151676]:
split_number = 0

# for split_number in range(3):
    
    # print(args.sample_number, split_number)

# Data loading and preprocessing
adata, interaction_df, complex_df, initial_x, grn, original_x, val_mask, test_mask = load_data_for_imputation(platform="10xVisium",sample_number=str(args.sample_number),split_number=split_number)
# adata, interaction_df, complex_df, initial_x, grn, original_x, val_mask, test_mask = load_data_for_imputation(platform="Stereoseq",sample_number=151507,split_number=0)
# adata, interaction_df, complex_df, initial_x, grn, original_x, val_mask, test_mask = load_data_for_imputation(platform="SlideseqV2",sample_number=151507,split_number=0)

input_dim = initial_x.shape[1]
hidden_dim = 32
output_dim = adata.shape[1]
device = "cuda"
gnn_type = "Transformer"
share_gnn = False
share_decoder = False
learning_rate = 0.01
patience = 50
omega = 0
spatial_loss_epoch = 1000

# Graph construction
mvG = MultiViewGraph(interaction_db=interaction_df, complex_db=complex_df, adata=adata)

# Model construction
model = iMiracle(mvG=mvG, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, device=device, gnn_type=gnn_type, share_gnn=share_gnn, share_decoder=share_decoder, GRN=grn)

# Trainer construction
Trainer = iMiracleTrainer(mode="imputation", model=model, adata=adata, initial_x=initial_x, observed_x=adata.X, device=device)
Trainer.load_parameter_for_imputation(original_x=original_x, val_mask=val_mask, test_mask=test_mask, learning_rate=learning_rate, patience=patience, omega=omega, spatial_loss_epoch=spatial_loss_epoch)

# Training
Trainer.train_model_for_imputation()

# Evaluate
res = Trainer.evaluate_for_imputation()
print(res)

torch.cuda.empty_cache()