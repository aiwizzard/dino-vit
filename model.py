import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    r"""PatchEmdedding class
    Args:
        image_size(int): size of the image. assume that image shape is square
        in_channels(int): input channel of the image, 3 for RGB color channel
        embed_size(int): output channel size. This is the latent vector size.
                         and is constant throughout the transformer
        patch_size(int): size of the patch

    Attributes:
        n_patches(int): calculate the number of patches.
        patcher: convert image into patches. Basically a convolution layer with
                 kernel size and stride as of the patch size
    """

    def __init__(self, image_size=224, in_channels=3, embed_size=768, patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.patcher = nn.Conv2d(in_channels, embed_size, patch_size, patch_size)

    def forward(self, x):
        # convert the images into patches
        out = self.patcher(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        return out


class ScaledDotProductAttention(nn.Module):
    r"""ScaledDotProductAttention class"""

    def __init__(self, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        dk = query.size(-1)
        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, value)
        return attn


class MultiHeadAttention(nn.Module):
    r"""MultiHeadAttention class

    Args:
    embed_size(int): latent vector size
    head(int): number of heads
    dropout_rate(float): probaility for dropout
    """

    def __init__(self, embed_size=768, head=12, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.key_dim = embed_size // head
        self.attn = ScaledDotProductAttention(dropout_rate)
        self.qkv_weights = nn.Linear(embed_size, embed_size, False)
        self.dense = nn.Linear(embed_size, embed_size, False)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        residual = query
        query = (
            self.qkv_weights(query)
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        key = (
            self.qkv_weights(key)
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        value = (
            self.qkv_weights(value)
            .view(batch_size, -1, self.head, self.key_dim)
            .transpose(1, 2)
        )
        attn = self.attn(query, key, value)
        attn = attn.view(batch_size, -1, self.head * self.key_dim)
        attn = self.dropout(self.dense(attn))
        attn += residual
        attn = self.layer_norm(attn)

        return attn


class MLP(nn.Module):
    r"""MLP class

    Args:
        embed_size(int): latent vector space
        hidden_size(int): intermediate size for the hidden layer
        dropout_rate(float): probability for dropout.
    """
    def __init__(self, embed_size=768, hidden_size=3072, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(embed_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, embed_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dense1(x)
        out = self.gelu(out)
        out = self.dense2(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    r"""Block class
    
    Args:
        embed_size(int): input latent vector size
        head(int): Number of heads in the multihead attention
    """
    def __init__(
        self,
        embed_size=768,
        head=12,
        hidden_size=3072,
        dropout_rate=0.1,
    ):
        super(Block, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_size)
        self.multi_head_attn = MultiHeadAttention(embed_size, head, dropout_rate)
        self.mlp = MLP(embed_size, hidden_size, dropout_rate)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.multi_head_attn(out, out, out)
        out += x
        residual = out
        out = self.layer_norm(out)
        out = self.mlp(out)
        out += residual
        return out


class VisionTransformer(nn.Module):
    r""" VisionTransformer class

    Args:
    n_classes(int): Number of classes
    depth(int): Number of layers
    image_size(int): Image width
    in_channel(int): input channels, Color channel of the image
    embed_size(int): Latent vector size
    patcth_size(int): Size for image patch
    head(int): Number of heads for the multihead attention part
    hidden_size(int): Hidden feature size for the MLP part
    dropout_rate(int): Probability for the dropout layer
    """
    def __init__(
        self,
        n_classes=0,
        depth=12,
        image_size=224,
        in_channels=3,
        embed_size=768,
        patch_size=16,
        head=12,
        hidden_size=3072,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(image_size, in_channels, embed_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [Block(embed_size, head, hidden_size, dropout_rate) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, n_classes) if n_classes > 0 else nn.Identity()
    
    def add_position_emb(self, inp_emb):
        _ ,n_patches, embed_size = inp_emb.shape
        position_emb = nn.Parameter(torch.zeros(1, n_patches, embed_size)).to(inp_emb.device)
        return position_emb
        

    def forward(self, x):
        patch_emb = self.patch_emb(x)
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        inp_emb = torch.cat((cls_token, patch_emb), dim=1)
        inp_emb += self.add_position_emb(inp_emb)
        inp_emb = self.dropout(inp_emb)
        for block in self.blocks:
            out = block(inp_emb)
        out = self.layer_norm(out)
        # Fetch only the embedding for class
        final_cls_token = out[:, 0]
        out = self.head(final_cls_token)
        return out
