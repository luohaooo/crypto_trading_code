"""
OHLC Deep Learning Model Architecture

This module defines the neural network architecture for OHLC image prediction.
The model processes stacked multi-timeframe OHLC images and predicts future returns.
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    OHLC Image Prediction Neural Network

    Architecture:
    - Input: (batch_size, 3, 64, 60) - 3 stacked timeframe images (3min, 15min, 1h)
    - 3 convolutional layers with batch normalization and LeakyReLU
    - Final fully connected layer for regression output
    - Output: (batch_size, 1) - predicted return value
    """

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # Third convolutional layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        # Fully connected layer with dropout
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46080, 1),
        )

        # Softmax layer (currently commented out as this is regression)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, 3, 64, 60)

        Returns:
            Output tensor (batch_size, 1) - predicted returns
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.fc1(x)
        return x

    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'OHLC_CNN',
            'input_shape': '(batch_size, 3, 64, 60)',
            'output_shape': '(batch_size, 1)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'task_type': 'regression'
        }


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(64, 60), patch_size=(4, 5), in_chans=3, embed_dim=256):
        super().__init__()
        ih, iw = img_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0, "img_size必须能被patch_size整除"
        self.grid_h = ih // ph
        self.grid_w = iw // pw
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))

    def forward(self, x):              # [B, C, H, W]
        x = self.proj(x)               # [B, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, v, k = qkv[0], qkv[2], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):  # [B, N, D]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerReg(nn.Module):
    """
    适配输入 [B, 3, 64, 60] 的ViT回归模型，输出 [B, 1]
    """
    def __init__(
        self,
        img_size=(64, 60),
        patch_size=(4, 5),
        in_chans=3,
        out_dim=1,          # 回归维度
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_cls_token=True, # True: 用CLS向量；False: 对tokens做平均
        return_scalar=False # True: 若 out_dim==1 则返回 [B]；默认 False 返回 [B,1]
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.out_dim = out_dim
        self.return_scalar = return_scalar and (out_dim == 1)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_cls_token = use_cls_token
        self.return_scalar = return_scalar

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 可选CLS
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_len = num_patches + 1
        else:
            self.cls_token = None
            pos_len = num_patches

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Encoder
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 回归头
        self.head = nn.Linear(embed_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def forward(self, x):  # x: [B, 3, 64, 60]
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat((cls_tokens, x), dim=1)          # [B,1+N,D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        feat = x[:, 0] if self.use_cls_token else x.mean(dim=1)  # [B, D]
        y = self.head(feat)  # [B, out_dim]

        if self.return_scalar and self.out_dim == 1:
            return y.squeeze(-1)  # [B]
        return y                  # [B, 1] (默认)
    
    def get_model_info(self):
        """用于训练流水线记录模型关键信息。"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = {
            "name": "VisionTransformerReg",
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "out_dim": self.out_dim,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "use_cls_token": self.use_cls_token,
            "num_patches": self.patch_embed.num_patches,
            "params_total": int(n_params),
            "params_trainable": int(n_trainable),
        }
        return info
        




def create_model(device=None, use_parallel=False):
    """
    Create and initialize the OHLC model

    Args:
        device: Target device ('cuda' or 'cpu')
        use_parallel: Whether to use DataParallel for multi-GPU

    Returns:
        Initialized model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net()
    # model = VisionTransformerReg(
    #     img_size=(64, 60),
    #     patch_size=(4, 3),
    #     in_chans=3,
    #     out_dim=1,
    #     embed_dim=256,
    #     depth=6,
    #     num_heads=8,
    #     mlp_ratio=4.0,
    #     drop_rate=0.1,
    #     attn_drop_rate=0.1,
    #     use_cls_token=True,
    #     return_scalar=False
    # )

    if device.type == 'cuda':
        model = model.to(device)
        if use_parallel and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model on device: {device}")

    # Create model
    model = create_model(device)

    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 64, 60).to(device)

    with torch.no_grad():
        output = model(test_input)
        print(f"\nTest forward pass:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output.flatten()[:5].cpu().numpy()}")