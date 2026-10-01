import torch
from torch import nn
from components import TransformerBlock, LayerNorm


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, images):
        # images: [B, C, H, W]
        # breakpoint()

        x = self.projection(images)
        # [B, embed_dim, grid_height, grid_width]

        x = x.flatten(start_dim=2)
        # [B, embed_dim, number_of_patches]

        x = x.transpose(1, 2)
        # [B, number_of_patches, embed_dim]

        return x



class ViT(nn.Module):
    def __init__(self, patch_size, num_t_blocks, embed_dim, num_heads, num_classes):
        super().__init__()

        self.cls = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, 65, embed_dim)) #64 image patches + 1 for cls

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.patch_embedding = PatchEmbedding(patch_size, 3, embed_dim)

        # self.emb_proj = nn.Linear(patch_dim, embed_dim)
        self.tb1 = TransformerBlock(embed_dim, num_heads)
        self.tb2 = TransformerBlock(embed_dim, num_heads)
        self.final_norm = LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes)


    def forward(self, X):
        # X = self.emb_proj(X)
        batch_size = X.shape[0]
        cls_exp = self.cls.expand(batch_size, -1, -1) # (B, 1, emb_dim)

        X = self.patch_embedding(X) 
        X = torch.cat([cls_exp, X], dim=1) # (B, 65, emb_dim)

        X = X + self.pos_embed # broadcasting will take care of this


        X = self.tb1(X)
        X = self.tb2(X)

        X = self.final_norm(X)
        logits = self.classification_head(X[:, 0]) #CLS used for classification
        return logits






        
