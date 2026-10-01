from torch import nn
import torch

class LayerNorm(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, X):
        'X: (B, N, D)'
        mu = torch.mean(X, dim=-1, keepdim=True)
        var = ((X - mu)**2).mean(dim=-1, keepdim=True)

        X_norm = (X - mu) / torch.sqrt(var + self.eps)

        return self.gamma * X_norm + self.beta


class AttentionHead(nn.Module):
    # Not used -- only for educational purposes. Use MHA instead
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.W_q = nn.Parameter(torch.empty(embed_dim, head_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, head_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, head_dim))

        nn.init.trunc_normal_(self.W_q, std=0.02)
        nn.init.trunc_normal_(self.W_k, std=0.02)
        nn.init.trunc_normal_(self.W_v, std=0.02)
    

    def forward(self, X):
        Q = torch.matmul(X, self.W_q)
        K = torch.matmul(X, self.W_k)
        V = torch.matmul(X, self.W_v)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / torch.sqrt(torch.tensor(self.head_dim))

        attn_mat = torch.softmax(scores, dim= -1)

        out = torch.matmul(attn_mat, V)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.b_q = nn.Parameter(torch.zeros(embed_dim))
        self.b_k = nn.Parameter(torch.zeros(embed_dim))
        self.b_v = nn.Parameter(torch.zeros(embed_dim))

        self.W_o = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_o = nn.Parameter(torch.zeros(embed_dim))


        nn.init.trunc_normal_(self.W_q, std=0.02)
        nn.init.trunc_normal_(self.W_k, std=0.02)
        nn.init.trunc_normal_(self.W_v, std=0.02)
        nn.init.trunc_normal_(self.W_o, std=0.02)


    def forward(self, X):

        batch_size, num_tokens, embed_dim = X.shape

        Q = torch.matmul(X, self.W_q) + self.b_q  # (B, N, embed_dim)
        K = torch.matmul(X, self.W_k) + self.b_k
        V = torch.matmul(X, self.W_v) + self.b_v

        #(B, N, num_head, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.reshape( 
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        K = K.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        V = V.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        scores = torch.matmul(Q, K.transpose(-2,-1))
        scores = scores * self.scale

        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, V) #(B, num_heads, N, head_dim)
        out = out.transpose(2, 1) #(B, N, num_heads, head_dim)
        out = out.reshape(batch_size, num_tokens, embed_dim)

        out = torch.matmul(out, self.W_o) + self.b_o

        return out





class MLP(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        layers_list = []
        for l in range(num_layers):
            layers_list.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers_list.pop()
        self.mlp = nn.Sequential(*layers_list)

    def forward(self, X):
        return self.mlp(X)




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.head_dim = embed_dim / num_heads

        # self.attn_heads = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.mlp = MLP(3, embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, X):
        
        X = self.mha(self.ln1(X)) + X
        X = self.mlp(self.ln2(X)) + X
        return X



        




def test_transformer_block():
    X = torch.tensor(
        [ 
            [[1,2,3,3],
             [4,5,6,6],
             [7,8,9, 9]]

        ]).float()
    transformer = TransformerBlock(4, 2)
    out = transformer(X)
    print(out)



def test_attention():
    X = torch.tensor([[1,2,3], [4,5,6]]).float()
    attn_h = AttentionHead(3, 2)
    upd = attn_h(X)
    print(upd)

# test_transformer_block()


# layers_list = []
# h=2

# for l in range(3):
#     layers_list.extend([nn.Linear(h,h), nn.ReLU()])

# print(layers_list)
# print(*layers_list)

# print(nn.Linear(3, 2))

