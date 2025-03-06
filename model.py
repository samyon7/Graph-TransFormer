import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. MLP (Multi-Layer Perceptron)
class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class SimpleGlobalAttention(nn.Module):
    def __init__(self, in_features, num_nodes):
        super(SimpleGlobalAttention, self).__init__()
        self.f_Q = nn.Linear(in_features, in_features)
        self.f_K = nn.Linear(in_features, in_features)
        self.f_V = nn.Linear(in_features, in_features)
        self.num_nodes = num_nodes
        self.embedding_dim = in_features

    def forward(self, Z_0):
        # Z_0: Node embeddings (num_nodes x in_features)

        # 1. Transform Query, Key, Value
        Q = self.f_Q(Z_0)
        K = self.f_K(Z_0)
        V = self.f_V(Z_0)

        # 2. Normalize Q and K (Layer Normalization)
        Q = F.layer_norm(Q, (self.embedding_dim,))
        K = F.layer_norm(K, (self.embedding_dim,))

        # 3. Simplest Possible Attention Mechanism:
        attention_scores = torch.matmul(Q, K.T)

        # 4. Weight them by values.
        attention_weights = F.softmax(attention_scores, dim=-1)
        Z_A = torch.matmul(attention_weights, V)

        return Z_A, attention_weights  # Return attention weights

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: Node features (num_nodes x in_features)
        # adj: Adjacency matrix (num_nodes x num_nodes)

        x = self.linear(x)
        x = torch.spmm(adj, x)  # Local aggregation
        return x

# GraphTransFormer model
class GraphTransFormer(nn.Module):
    def __init__(self, num_features, num_classes, num_nodes, alpha, embedding_dim):
        super(GraphTransFormer, self).__init__()
        self.embedding = MLP(num_features, embedding_dim) # Maps input to latent
        self.simple_global_attention = SimpleGlobalAttention(embedding_dim, num_nodes)
        self.gnn = GNNLayer(embedding_dim, embedding_dim)  # GNN operates on the latent space
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.alpha = alpha # To control how important GCN will be.
        self.num_nodes = num_nodes

    def forward(self, X, A):
        # X: Node features (num_nodes x num_features)
        # A: Adjacency matrix (num_nodes x num_nodes)

        # 1. Embedding
        Z_0 = self.embedding(X)  # Embed input features
        # 2. Simple Global Attention
        AN, attn_weights = self.simple_global_attention(Z_0)

        # 3. GNN
        gnn_output = self.gnn(Z_0, A)  # Propagate

        # 4. Combine
        Z_out = (1 - self.alpha) * AN + self.alpha * gnn_output

        # 5. Prediction
        Y_hat = self.classifier(Z_out)

        return Y_hat, attn_weights
