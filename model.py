import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. MLP (Multi-Layer Perceptron)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. SimpleGlobalAttention
class SimpleGlobalAttention(nn.Module):
    def __init__(self, embedding_dim, num_nodes):
        super(SimpleGlobalAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_nodes, embedding_dim))

    def forward(self, Z_0):
        # Compute attention scores
        attention_scores = torch.matmul(Z_0, self.attention_weights.T)  # Shape: (batch_size, num_nodes)
        attention_scores = F.softmax(attention_scores, dim=1)  # Normalize across nodes

        # Compute weighted sum of node embeddings
        AN = torch.matmul(attention_scores.unsqueeze(1), Z_0).squeeze(1)  # Shape: (batch_size, embedding_dim)
        return AN

# 3. GNNLayer (Graph Neural Network Layer)
class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, Z_0, A):
        # Aggregate neighbor information using adjacency matrix
        Z_agg = torch.matmul(A, Z_0)  # Shape: (batch_size, num_nodes, input_dim)
        Z_out = self.linear(Z_agg)  # Apply linear transformation
        return Z_out

# 4. GraphTransFormer
class GraphTransFormer(nn.Module):
    def __init__(self, num_features, num_classes, num_nodes, alpha, embedding_dim):
        super(GraphTransFormer, self).__init__()
        self.embedding = MLP(num_features, embedding_dim)
        self.simple_global_attention = SimpleGlobalAttention(embedding_dim, num_nodes)
        self.gnn = GNNLayer(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.alpha = alpha
        self.num_nodes = num_nodes

    def forward(self, X, A):
        # 1. Embedding
        Z_0 = self.embedding(X)  # Shape: (batch_size, num_nodes, embedding_dim)

        # 2. Global Attention
        AN = self.simple_global_attention(Z_0)  # Shape: (batch_size, embedding_dim)

        # 3. GNN
        gnn_output = self.gnn(Z_0, A)  # Shape: (batch_size, num_nodes, embedding_dim)

        # 4. Combine
        Z_out = (1 - self.alpha) * AN.unsqueeze(1) + self.alpha * gnn_output  # Shape: (batch_size, num_nodes, embedding_dim)

        # 5. Prediction
        Y_hat = self.classifier(Z_out.mean(dim=1))  # Average over nodes and classify

        return Y_hat
