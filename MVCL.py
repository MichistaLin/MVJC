import os

os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import time
from tasks import predict_crime, lu_classify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data prepare ---
def torch_to_topk_edges(k, a_tor):
	values, indices = torch.topk(a_tor, k, dim=1)
	adj = torch.zeros_like(a_tor)
	adj.scatter_(1, indices, values)
	adj[adj > 0] = 1
	edges = adj.nonzero().t().contiguous()
	return edges


def prepare_data(k):
	data_path = "./data/"
	torch.manual_seed(88)
	features = torch.rand(180, 250)  # N=180, F_initial=250

	adj_files = {
		"adj_mov.npy": (180, 180),
		"adj_simi_s.npy": (180, 180),
		"adj_simi_d.npy": (180, 180),
		"adj_simi_poi_180.npy": (180, 180),
		"adj_simi_chk.npy": (180, 180)
	}

	if not os.path.exists(data_path):
		os.makedirs(data_path)
		print(f"Directory {data_path} created.")
		for fname, shape in adj_files.items():
			fpath = os.path.join(data_path, fname)
			if not os.path.exists(fpath):
				print(f"File {fpath} not found, creating random data.")
				np.save(fpath, np.random.rand(*shape))
	else:
		for fname, shape in adj_files.items():
			fpath = os.path.join(data_path, fname)
			if not os.path.exists(fpath):
				print(f"File {fpath} not found, creating random data.")
				np.save(fpath, np.random.rand(*shape))

	adj_mov_np = np.load(os.path.join(data_path, "adj_mov.npy")).squeeze()
	adj_mov = torch.from_numpy(adj_mov_np).float()
	adj_mov_norm = adj_mov / (torch.mean(adj_mov) + 1e-9)  

	k1 = k
	adj_simi_s_np = np.load(os.path.join(data_path, "adj_simi_s.npy"))
	adj_simi_s = torch.from_numpy(adj_simi_s_np).float()
	edges_s = torch_to_topk_edges(k1, adj_simi_s)
	data_s = Data(x=features, edge_index=edges_s)

	adj_simi_d_np = np.load(os.path.join(data_path, "adj_simi_d.npy"))
	adj_simi_d = torch.from_numpy(adj_simi_d_np).float()
	edges_d = torch_to_topk_edges(k1, adj_simi_d)
	data_d = Data(x=features, edge_index=edges_d)

	k2 = k
	adj_simi_poi_np = np.load(os.path.join(data_path, "adj_simi_poi_180.npy"))
	adj_simi_poi = torch.from_numpy(adj_simi_poi_np).float()
	edges_poi = torch_to_topk_edges(k2, adj_simi_poi)
	data_poi = Data(x=features, edge_index=edges_poi)

	adj_simi_chk_np = np.load(os.path.join(data_path, "adj_simi_chk.npy"))
	adj_simi_chk = torch.from_numpy(adj_simi_chk_np).float()
	edges_chk = torch_to_topk_edges(k2, adj_simi_chk)
	data_chk = Data(x=features, edge_index=edges_chk)

	return adj_mov_norm, adj_simi_poi, adj_simi_chk, data_s, data_d, data_poi, data_chk


# --- GAT  ---
class GAT(torch.nn.Module):
	def __init__(self, input_dim, output_dim, l1_output_dim=16, l1_heads=12, l1_drop=0.1, l2_drop=0.1, node_drop=0.3):
		super(GAT, self).__init__()
		self.conv1 = GATConv(input_dim, l1_output_dim, heads=l1_heads, dropout=l1_drop)
		self.conv2 = GATConv(l1_output_dim * l1_heads, output_dim, heads=1, dropout=l2_drop)
		self.node_drop = node_drop

	def forward(self, data):
		x = data.x
		edge_index = data.edge_index
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, p=self.node_drop, training=self.training)
		x = self.conv2(x, edge_index)
		return x


# --- GCFAgg Core ---
class GCFAggCore(nn.Module):
	def __init__(self, input_dim_concat, key_dim, value_dim, mlp_hidden1, mlp_hidden2, output_dim_consensus):
		super(GCFAggCore, self).__init__()
		self.input_dim_concat = input_dim_concat
		self.key_dim = key_dim  # Dimension for Q1, Q2
		self.value_dim = value_dim  # Dimension for R, should be same as input_dim_concat for addition

		self.W_q1 = nn.Linear(input_dim_concat, key_dim)
		self.W_q2 = nn.Linear(input_dim_concat, key_dim)
		self.W_r = nn.Linear(input_dim_concat, value_dim)  # R

		# MLP for consensus representation (Eq. 9 in GCFAgg paper)
		self.mlp_w1 = nn.Linear(value_dim, mlp_hidden1)  # input is Z_cat + Z_hat (value_dim)
		self.mlp_b1 = nn.Parameter(torch.zeros(mlp_hidden1))
		self.mlp_w2 = nn.Linear(mlp_hidden1, mlp_hidden2)
		self.mlp_b2 = nn.Parameter(torch.zeros(mlp_hidden2))
		self.mlp_w3 = nn.Linear(mlp_hidden2, output_dim_consensus)
		self.mlp_b3 = nn.Parameter(torch.zeros(output_dim_consensus))

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
		nn.init.zeros_(self.mlp_b1)
		nn.init.zeros_(self.mlp_b2)
		nn.init.zeros_(self.mlp_b3)

	def forward(self, Z_concat):
		# Z_concat: (N, D_concat)
		N = Z_concat.shape[0]

		Q1 = self.W_q1(Z_concat)  # (N, key_dim)
		Q2 = self.W_q2(Z_concat)  # (N, key_dim)
		R_val = self.W_r(Z_concat)  # (N, value_dim)

		# Calculate S (Eq. 7)
		S_raw = torch.matmul(Q1, Q2.transpose(0, 1)) / np.sqrt(self.key_dim)
		S = F.softmax(S_raw, dim=1)  # (N, N), structure relationship matrix

		# Calculate Z_hat (Eq. 8)
		Z_hat_paper = torch.matmul(S, R_val)  # (N, value_dim)

		# Calculate consensus H_hat (Eq. 9)
		# Input to MLP is (Z + Z_hat_paper)
		if Z_concat.shape[1] != Z_hat_paper.shape[1]:
			raise ValueError(
				f"Dimension mismatch for Z_concat ({Z_concat.shape[1]}) and Z_hat_paper ({Z_hat_paper.shape[1]}). Ensure value_dim equals input_dim_concat.")

		mlp_input = Z_concat + Z_hat_paper

		h = F.relu(torch.matmul(mlp_input, self.mlp_w1.weight.t()) + self.mlp_b1)
		h = torch.matmul(h, self.mlp_w2.weight.t()) + self.mlp_b2
		H_consensus = torch.matmul(h, self.mlp_w3.weight.t()) + self.mlp_b3

		return H_consensus, S


# --- SgCL Loss ---
class SgCLLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SgCLLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-9

    def forward(self, H_consensus, S_similarity, list_H_views):
        N = H_consensus.shape[0]
        num_views = len(list_H_views)
        total_loss = 0.0

        H_consensus_norm = F.normalize(H_consensus, p=2, dim=1) # (N, D_contrast)

        for v_idx in range(num_views):
            H_v = list_H_views[v_idx] # (N, D_contrast)
            H_v_norm = F.normalize(H_v, p=2, dim=1) # (N, D_contrast)

            # sim_matrix[i, j] = C(H_consensus_norm[i], H_v_norm[j])
            # (N, D_contrast) @ (D_contrast, N) -> (N, N)
            sim_matrix = torch.matmul(H_consensus_norm, H_v_norm.transpose(0, 1))

            # Positive similarities for each sample i with its corresponding H_v_i
            # Extracts diagonal elements: C(H_hat_i, H_v_i)
            positive_sim = torch.diag(sim_matrix) / self.temperature # Shape: (N)

            # Denominator calculation: sum_j (1 - S_ij) * exp(C(H_hat_i, H_v_j) / T)
            # S_similarity: (N, N), S_similarity[i,j] is S_ij
            # exp_sim_matrix[i,j] = exp(C(H_hat_i, H_v_j) / T)
            exp_sim_matrix = torch.exp(sim_matrix / self.temperature) # Shape: (N, N)

            # Weights for the sum (1 - S_ij)
            # Unsqueeze S_similarity if it's (N,N) and exp_sim_matrix is (N,N) to allow broadcasting, though direct element-wise should work
            # The paper implies S_ij is for sample i vs sample j, which is S_similarity.
            # So for a given H_hat_i, we consider all H_v_j. S_similarity[i,:] gives all S_ij for fixed i.
            # weights_for_sum[i,j] = (1 - S_ij)
            weights_for_sum = (1.0 - S_similarity) # Shape (N, N)

            # weighted_exp_sum_terms[i,j] = (1 - S_ij) * exp(C(H_hat_i, H_v_j) / T)
            weighted_exp_sum_terms = weights_for_sum * exp_sim_matrix # Shape (N,N)

            # Sum over j for each i
            # denominator_sum[i] = sum_j (1 - S_ij) * exp(C(H_hat_i, H_v_j) / T)
            denominator_sum = torch.sum(weighted_exp_sum_terms, dim=1) # Shape (N)
            denominator_sum = torch.clamp(denominator_sum, min=self.epsilon) # Avoid log(0)

            # Loss for each sample i for current view v
            # log( exp(positive_sim_i) / denominator_sum_i ) = positive_sim_i - log(denominator_sum_i)
            loss_per_sample_for_view = - (positive_sim - torch.log(denominator_sum)) # Shape (N)
            total_loss += torch.sum(loss_per_sample_for_view)

        return total_loss / (N * num_views)


# ---  GCFAgg ---
class MVURE_GCFAgg(nn.Module):
	def __init__(self, initial_feature_dim=250, gat_output_dim=96, num_views=4,
	             contrast_dim=128,  # Dimension for H_hat and H_v (output of MLPs for SgCL)
	             gc_fagg_key_dim=128,  # Key dimension in GCFAggCore
	             # gc_fagg_value_dim is fixed to num_views * gat_output_dim
	             gc_fagg_mlp_hidden1=256,
	             gc_fagg_mlp_hidden2=256,
	             view_mlp_hidden_dim=128):  # Hidden dim for MLPs projecting Z_v to H_v
		super(MVURE_GCFAgg, self).__init__()
		self.num_views = num_views
		self.gat_output_dim = gat_output_dim
		self.contrast_dim = contrast_dim

		# GAT Encoders for each view (Z^v)
		self.gat_s = GAT(initial_feature_dim, gat_output_dim).to(device)
		self.gat_d = GAT(initial_feature_dim, gat_output_dim).to(device)
		self.gat_poi = GAT(initial_feature_dim, gat_output_dim).to(device)
		self.gat_chk = GAT(initial_feature_dim, gat_output_dim).to(device)
		self.view_gats = nn.ModuleList([self.gat_s, self.gat_d, self.gat_poi, self.gat_chk])

		# MLPs to get H^v from Z^v (for SgCL)
		self.view_mlps = nn.ModuleList()
		for _ in range(num_views):
			mlp = nn.Sequential(
				nn.Linear(gat_output_dim, view_mlp_hidden_dim),
				nn.ReLU(),
				nn.Linear(view_mlp_hidden_dim, contrast_dim)
			).to(device)
			self.view_mlps.append(mlp)

		# GCFAgg Core Module
		self.concat_dim = num_views * gat_output_dim
		self.gc_fagg_module = GCFAggCore(
			input_dim_concat=self.concat_dim,
			key_dim=gc_fagg_key_dim,
			value_dim=self.concat_dim,  # value_dim must match input_dim_concat for Z + Z_hat
			mlp_hidden1=gc_fagg_mlp_hidden1,
			mlp_hidden2=gc_fagg_mlp_hidden2,
			output_dim_consensus=contrast_dim  # H_consensus also has contrast_dim
		).to(device)

	def forward(self, list_data_views):
		# list_data_views: [data_s, data_d, data_poi, data_chk]

		list_Z_views = []  # To store Z^v from GATs
		list_H_views_for_sgcl = []  # To store H^v from MLPs (for SgCL loss)

		for i in range(self.num_views):
			Z_v = self.view_gats[i](list_data_views[i])  # Z^v
			list_Z_views.append(Z_v)

			H_v_for_sgcl = self.view_mlps[i](Z_v)  # H^v
			list_H_views_for_sgcl.append(H_v_for_sgcl)

		# Concatenate Z_views to form Z_concat for GCFAgg module
		Z_concat = torch.cat(list_Z_views, dim=1)  # (N, num_views * gat_output_dim)

		# Get consensus representation H_consensus and similarity matrix S
		H_consensus, S_similarity = self.gc_fagg_module(Z_concat)

		# H_consensus is the final embedding for downstream tasks
		# S_similarity and list_H_views_for_sgcl are for the SgCL loss
		return H_consensus, S_similarity, list_H_views_for_sgcl


# ================== main ==================
# if __name__ == '__main__':
def mvcl(k=5, temperature=0.5):
	print("Using device:", device)

	# 1. Prepare Data
	# These adj matrices are not used in this GCFAgg loss, but data prep is kept
	_, _, _, data_s, data_d, data_poi, data_chk = prepare_data(k)

	list_of_data = [data_s.to(device), data_d.to(device), data_poi.to(device), data_chk.to(device)]

	# 2. Initialize Model, Loss, Optimizer
	num_epochs = 500  # GCFAgg paper uses ~100-200 epochs for fine-tuning
	learning_rate = 0.001  # Typical LR for Adam with such models

	# Model parameters (can be tuned)
	initial_feat_dim = data_s.x.shape[1]  # Should be 250
	gat_out_dim = 96
	num_v = 4
	contrast_d = 128  # Dimension of final consensus embedding and H^v for SgCL
	gc_fagg_k_dim = 128
	gc_fagg_mlp_h1 = 256
	gc_fagg_mlp_h2 = 256
	view_mlp_h_dim = 128

	model = MVURE_GCFAgg(
		initial_feature_dim=initial_feat_dim,
		gat_output_dim=gat_out_dim,
		num_views=num_v,
		contrast_dim=contrast_d,
		gc_fagg_key_dim=gc_fagg_k_dim,
		gc_fagg_mlp_hidden1=gc_fagg_mlp_h1,
		gc_fagg_mlp_hidden2=gc_fagg_mlp_h2,
		view_mlp_hidden_dim=view_mlp_h_dim
	).to(device)

	sgcl_loss_fn = SgCLLoss(temperature).to(device)  # Temp from GCFAgg paper
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

	print("Model initialized. Starting training...")
	# 3. Training Loop
	for epoch in range(num_epochs):
		model.train()
		optimizer.zero_grad()

		H_consensus_train, S_similarity_train, list_H_views_train = model(list_of_data)

		loss = sgcl_loss_fn(H_consensus_train, S_similarity_train, list_H_views_train)

		loss.backward()
		optimizer.step()

		print(f'Epoch {epoch + 1}/{num_epochs}, SgCL Loss: {loss.item():.4f}')

	print("Training finished.")
	# 4. Evaluation (using learned H_consensus for downstream tasks)
	model.eval()
	with torch.no_grad():
		H_final_embeddings, _, _ = model(list_of_data)  # Get the consensus embeddings

	H_final_embeddings_cpu = H_final_embeddings.cpu().numpy()
	return H_final_embeddings_cpu
	# print("Final Embedding shape for downstream tasks:", H_final_embeddings_cpu.shape)  # (N, contrast_dim)
	#
	# timestamp = time.strftime("%Y%m%d-%H%M%S")
	# np.save(f"H_final_embeddings_{timestamp}.npy", H_final_embeddings_cpu)
	# # np.save("H_final_embeddings.npy", H_final_embeddings_cpu)
	#
	# print("\n--- Crime Prediction Task ---")
	# predict_crime(H_final_embeddings_cpu)
	#
	# print("\n--- Land Use Classification Task ---")
	# lu_classify(H_final_embeddings_cpu)
	#
	# print("\nScript finished.")