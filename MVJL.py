import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from tasks import predict_crime, predict_checkin, lu_classify



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_to_topk_edges(k, a_tor):

    values, indices = torch.topk(a_tor, k, dim=1) 
    adj = torch.zeros_like(a_tor)
    adj.scatter_(1, indices, values)
    adj[adj > 0] = 1                        
    edges = adj.nonzero().t().contiguous()  
    return edges


def prepare_data(k):

    data_path = "./data/"
    torch.manual_seed(1314)
    features = torch.rand(180, 250)

    k2 = k
    adj_simi_poi_np = np.load(data_path+"poi_simi.npy")
    adj_simi_poi = torch.from_numpy(adj_simi_poi_np).float()
    edges_poi = torch_to_topk_edges(k2, adj_simi_poi)
    data_poi = Data(x=features, edge_index=edges_poi)

    adj_simi_chk_np = np.load(data_path+"adj_simi_chk.npy")
    adj_simi_chk = torch.from_numpy(adj_simi_chk_np).float()
    edges_chk = torch_to_topk_edges(k2, adj_simi_chk)
    data_chk = Data(x=features, edge_index=edges_chk)


    return adj_simi_poi, adj_simi_chk, data_poi, data_chk

def pairwise_inner_product(x, y):
    return torch.matmul(x, y.transpose(-2, -1))


def get_adj_loss(embeddings, adj):
    inner_prod = pairwise_inner_product(embeddings, embeddings)
    return F.mse_loss(inner_prod, adj)


def get_mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = pairwise_inner_product(s_embeddings, t_embeddings)
    phat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-mob * torch.log(phat + 1e-9))
    inner_prod = pairwise_inner_product(t_embeddings, s_embeddings)
    phat = F.softmax(inner_prod, dim=-1)
    loss += torch.sum(-mob.transpose(-2, -1) * torch.log(phat + 1e-9))
    return loss


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, out1, out2, label):
        mob_loss = get_mob_loss(out1, out2, label)
        return mob_loss


# ---------------------- MGFN ----------------------
class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
      
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class ConcatLinear(nn.Module):
    """
    input: (*, a) and (*, b)
    output (*, c)
    """

    def __init__(self, in_1, in_2, out, dropout=0.1):
        super(ConcatLinear, self).__init__()
        self.linear1 = nn.Linear(in_1+in_2, out)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out)

        self.linear2 = nn.Linear(out, out)
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x1, x2):
        src = torch.cat([x1, x2], -1)
        out = self.linear1(src)
        out = src + self.dropout1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.act2(self.linear2(out))
        return out


class GraphStructuralEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(GraphStructuralEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src, src, src,)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MobilityPatternJointLearning(nn.Module):
    """
    input: (7, 180, 180)
    output: (180, 96)
    """

    def __init__(self, graph_num, node_num, output_dim):
        super(MobilityPatternJointLearning, self).__init__()
        self.graph_num = graph_num
        self.node_num = node_num
        self.num_multi_pattern_encoder = 3
        self.num_cross_graph_encoder = 1
        self.multi_pattern_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=4) for _ in range(self.num_multi_pattern_encoder)])
        self.cross_graph_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=4) for _ in range(self.num_cross_graph_encoder)])
        self.fc = DeepFc(self.graph_num*self.node_num, output_dim)
        self.linear_out = nn.Linear(node_num, output_dim)
        self.para1 = torch.nn.Parameter(torch.FloatTensor(1).to(
            device), requires_grad=True)  # the size is [1]
        self.para1.data.fill_(0.7)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1).to(
            device), requires_grad=True)  # the size is [1]
        self.para2.data.fill_(0.3)
        assert node_num % 2 == 0
        self.s_linear = nn.Linear(node_num, int(node_num / 2))
        self.o_linear = nn.Linear(node_num, int(node_num / 2))
        self.concat = ConcatLinear(
            int(node_num / 2), int(node_num / 2), node_num)

    def forward(self, x):
        out = x
        for multi_pattern in self.multi_pattern_blocks:
            out = multi_pattern(out)
        multi_pattern_emb = out
        out = out.transpose(0, 1)
        for cross_graph in self.cross_graph_blocks:
            out = cross_graph(out)
        out = out.transpose(0, 1)
        out = out*self.para2 + multi_pattern_emb*self.para1
        out = out.contiguous()
        out = out.view(-1, self.node_num*self.graph_num)
        out = self.fc(out)
        return out


class MGFN(nn.Module):
    def __init__(self, graph_num, node_num, output_dim):
        super(MGFN, self).__init__()
        self.encoder = MobilityPatternJointLearning(
            graph_num=graph_num, node_num=node_num, output_dim=output_dim)
        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.feature = None

    def forward(self, x):
        self.feature = self.encoder(x)
        out_s = self.decoder_s(self.feature)
        out_t = self.decoder_t(self.feature)
        return self.feature, out_s, out_t


# ---------------------- MVURE ----------------------
class GAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, l1_output_dim=16, l1_heads=12, l1_drop=0.1, l2_drop=0.1, node_drop=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, l1_output_dim,
                             heads=l1_heads, dropout=l1_drop).to(device)
        self.conv2 = GATConv(l1_output_dim * l1_heads,
                             output_dim, heads=1, dropout=l2_drop).to(device)
        self.node_drop = node_drop

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.node_drop, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class CGCGating(nn.Module):

    def __init__(self, num_views, input_dim):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(num_views * input_dim, input_dim),
            nn.ReLU()
        ).to(device)
        self.weight_generator = nn.Linear(input_dim, num_views).to(device)

    def forward(self, view_embeddings):
        batch_size, num_views, input_dim = view_embeddings.shape
        global_context = torch.flatten(view_embeddings, start_dim=1)
        global_context = self.global_proj(global_context)
        weights = F.softmax(self.weight_generator(global_context), dim=-1)
        return torch.einsum('bv,bvd->bd', weights, view_embeddings)


class MultiViewAttention(nn.Module):

    def __init__(self, num_views, input_dim):
        super().__init__()
        self.gating = CGCGating(num_views, input_dim).to(device)

    def forward(self, x):
        return self.gating(x)


class SelfAttention(nn.Module):

    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.shared_gate = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        self.specific_gate = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        self.value_proj = nn.Linear(feature_dim, feature_dim).to(device)

    def forward(self, x):
        batch_size, num_views, _ = x.shape
        global_context = torch.mean(x, dim=1)
        shared_weights = self.shared_gate(global_context)
        specific_weights = torch.stack(
            [self.specific_gate(x[:, i, :]) for i in range(num_views)], dim=1)
        V = self.value_proj(x)
        return torch.einsum('ba,bva,bvf->bf', shared_weights, specific_weights, V)


# ---------------------- model ----------------------
class MVURE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_poi = GAT(250, 96).to(device)
        self.gat_chk = GAT(250, 96).to(device)

 
        self.mgfn = MGFN(graph_num=7, node_num=180, output_dim=96).to(device)

       
        self.self_attn = SelfAttention(
            feature_dim=96, attention_dim=12).to(device)
        self.mv_attn = MultiViewAttention(num_views=3, input_dim=96).to(device)

        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值0.5
        self.beta = nn.Parameter(torch.tensor(0.5))

        
        nn.init.uniform_(self.alpha, 0.4, 0.6)
        nn.init.uniform_(self.beta, 0.4, 0.6)

    def forward(self, mob_pattern, mob_adj, data_poi, data_chk):
        
        out_poi = self.gat_poi(data_poi)
        out_chk = self.gat_chk(data_chk)

        
        out_mob, s_out, t_out = self.mgfn(mob_pattern)

        
        stacked_views = torch.stack([out_mob, out_poi, out_chk], dim=1)

        
        shared_info = self.self_attn(stacked_views)

        
        fused_embedding = self.mv_attn(stacked_views)

        
        out_mob = self.alpha * shared_info + (1 - self.alpha) * out_mob
        out_poi = self.alpha * shared_info + (1 - self.alpha) * out_poi
        out_chk = self.alpha * shared_info + (1 - self.alpha) * out_chk

        out_mob = self.beta * fused_embedding + (1 - self.beta) * out_mob
        out_poi = self.beta * fused_embedding + (1 - self.beta) * out_poi
        out_chk = self.beta * fused_embedding + (1 - self.beta) * out_chk

        return out_mob, out_poi, out_chk, s_out, t_out


# ---------------------- train ----------------------
def train_model(k):

    adj_simi_poi, adj_simi_chk, data_poi, data_chk = prepare_data(k)
    adj_simi_poi = adj_simi_poi.to(device)
    adj_simi_chk = adj_simi_chk.to(device)
    data_poi = data_poi.to(device)
    data_chk = data_chk.to(device)

   
    mob_pattern = np.load("./data/mob_patterns.npy")
    mob_adj = np.load("./data/mob_label.npy")
    mob_pattern = torch.from_numpy(mob_pattern).float().to(device)
    mob_adj = torch.from_numpy(mob_adj).float().to(device)

   
    model = MVURE().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=5e-4)  

  
    mob_criterion = SimLoss().to(device)

    
    model.train()
   

    for epoch in range(900):
        optimizer.zero_grad()

        out_mob, out_poi, out_chk, s_out, t_out = model(
            mob_pattern, mob_adj, data_poi, data_chk)

        
        adj_loss = get_adj_loss(out_poi, adj_simi_poi) + \
            get_adj_loss(out_chk, adj_simi_chk)
        mob_loss = mob_criterion(s_out, t_out, mob_adj)

       
        loss = adj_loss + mob_loss

        loss.backward()
        optimizer.step()

        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | '
                  f'Alpha: {model.alpha.item():.3f} | Beta: {model.beta.item():.3f}')

            
            model.eval()
            with torch.no_grad():
                eval_mob, eval_poi, eval_chk, _, _ = model(
                    mob_pattern, mob_adj, data_poi, data_chk)

                eval_tot = np.concatenate((
                    eval_mob.cpu().numpy(),
                    eval_poi.cpu().numpy(),
                    eval_chk.cpu().numpy()
                ), axis=1)

                mae, rmse, r2 = predict_crime(eval_tot)
                mae1, rmse1, r21 = predict_checkin(eval_tot)
                nmi, ari, f = lu_classify(eval_tot)
             

            model.train()

    model.eval()
    with torch.no_grad():
        prt_mob, prt_poi, prt_chk, _, _ = model(
            mob_pattern, mob_adj, data_poi, data_chk)

    prt_tot = np.concatenate((
        prt_mob.cpu().numpy(),
        prt_poi.cpu().numpy(),
        prt_chk.cpu().numpy()
    ), axis=1)

    return prt_tot
    
if __name__ == "__main__":
def mvjl(k=5):
    print(f"Using device: {device}")
    return train_model(k)
