import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 1) 单个 timeframe 编码器：Image + Multi-Scalar -> Vector ---------
class SingleTFEncoder(nn.Module):
    """
    输入:
      x_img: (B, 1, 64, 60)      # 单通道图片 (该 timeframe 的K线图)
      x_sc : (B, scalar_dim)     # 对应的多标量 (默认2维, 例如 [log(H/L), 新标量])

    输出:
      feat: (B, d_model)         # 该 timeframe 的向量化特征
    """
    def __init__(self, d_model=192, scalar_dim=2):
        super().__init__()
        self.scalar_dim = scalar_dim

        # 卷积主干: 复用你原先的参数配置
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        # 尺寸路径: (B,1,64,60)->(B,64,27,60)->(B,64,13,60)->(B,128,10,60)->(B,128,5,60)->(B,256,7,60)->(B,256,3,60)
        # self.gap = nn.AdaptiveAvgPool2d((1,1))  # (B,256,1,1) -> (B,256)
        self.fc1 = nn.Sequential(
            nn.Linear(256*3*60, 8),
            nn.Dropout(0.1),
        )

        # 标量支路: scalar_dim -> 32
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Dropout(0.05),
        )

        # 融合投影: (256 + 32) -> d_model
        self.proj = nn.Sequential(
            nn.Linear(8 + 4, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x_img, x_sc):
        # x_sc 保证是 (B, scalar_dim)
        if x_sc.dim() == 1:
            x_sc = x_sc.unsqueeze(1)
        h = self.layer1(x_img)
        h = self.layer2(h)
        h = self.layer3(h)
        h = h.reshape(h.size(0), -1)   
        h = self.fc1(h)                # (B,8)

        z = self.scalar_mlp(x_sc)            
        v = torch.cat([h, z], dim=1)         
        feat = self.proj(v)                  # (B,d_model)
        return feat


# --------- 2) 三个 timeframe 融合 + 回归 ---------
class NetTimeframeAware(nn.Module):
    """
    输入:
      x_imgs:    (B, 3, 64, 60)  或 [ (B,1,64,60), (B,1,64,60), (B,1,64,60) ]
      x_scalars: (B, 3, S)       # 每个 timeframe 有 S 个标量，这里 S=2

    输出:
      out: (B,1)

    参数:
      d_model: 每个 timeframe 编码后的特征维度
      scalar_dim: 每个 timeframe 的标量维度 (=2)
      shared_cnn: 三个 timeframe 是否共享编码器
      fuse: 'attn' | 'concat' | 'mean'
    """
    def __init__(self, d_model=192, scalar_dim=2, shared_cnn=True, fuse="attn"):
        super().__init__()
        self.fuse = fuse
        self.shared_cnn = shared_cnn
        self.scalar_dim = scalar_dim

        if shared_cnn:
            self.encoder = SingleTFEncoder(d_model=d_model, scalar_dim=scalar_dim)
        else:
            self.encoder_0 = SingleTFEncoder(d_model=d_model, scalar_dim=scalar_dim)  
            self.encoder_1 = SingleTFEncoder(d_model=d_model, scalar_dim=scalar_dim)  
            self.encoder_2 = SingleTFEncoder(d_model=d_model, scalar_dim=scalar_dim)  

        if fuse == "attn":
            self.attn_mlp = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1)
            )
            self.fuse_out_dim = d_model
        elif fuse == "concat":
            self.fuse_out_dim = d_model * 3
        elif fuse == "mean":
            self.fuse_out_dim = d_model
        else:
            raise ValueError("fuse must be one of {'attn','concat','mean'}")

        self.head = nn.Sequential(
            nn.Linear(self.fuse_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def _split_imgs(self, x_imgs):
        if isinstance(x_imgs, (list, tuple)):
            x0, x1, x2 = x_imgs
        else:
            x0, x1, x2 = x_imgs[:, 0:1], x_imgs[:, 1:2], x_imgs[:, 2:3]
        return x0, x1, x2

    def forward(self, x_imgs, x_scalars):
        """
        x_imgs:    (B,3,64,60) 或 [ (B,1,64,60), (B,1,64,60), (B,1,64,60) ]
        x_scalars: (B,3,S)      # S=scalar_dim=2，例如:
                                # x_scalars[:,0,:] -> 80h 的两个标量
                                # x_scalars[:,1,:] -> 40h 的两个标量
                                # x_scalars[:,2,:] -> 20h 的两个标量
        """
        B = x_scalars.size(0)
        S = x_scalars.size(2) if x_scalars.dim() == 3 else self.scalar_dim
        assert S == self.scalar_dim, f"Expected scalar_dim={self.scalar_dim}, got {S}"

        x0, x1, x2 = self._split_imgs(x_imgs)
        s0 = x_scalars[:, 0, :]  # (B, S)
        s1 = x_scalars[:, 1, :]
        s2 = x_scalars[:, 2, :]

        if self.shared_cnn:
            f0 = self.encoder(x0, s0)  # (B,d_model)
            f1 = self.encoder(x1, s1)
            f2 = self.encoder(x2, s2)
        else:
            f0 = self.encoder_0(x0, s0)
            f1 = self.encoder_1(x1, s1)
            f2 = self.encoder_2(x2, s2)

        F_stack = torch.stack([f0, f1, f2], dim=1)  # (B,3,d_model)

        if self.fuse == "attn":
            scores = self.attn_mlp(F_stack)      # (B,3,1)
            alpha = torch.softmax(scores, dim=1) # (B,3,1)
            F = (F_stack * alpha).sum(dim=1)     # (B,d_model)
        elif self.fuse == "concat":
            F = F_stack.reshape(B, -1)           # (B, 3*d_model)
        elif self.fuse == "mean":
            F = F_stack.mean(dim=1)              # (B,d_model)

        out = self.head(F)                        # (B,1)
        return out
    
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



def create_model_v3(device=None, use_parallel=False):
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

    model = NetTimeframeAware(d_model=32, scalar_dim=2, shared_cnn=True, fuse="concat")

    if device.type == 'cuda':
        model = model.to(device)
        if use_parallel and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    return model

# model = NetTimeframeAware(d_model=192, scalar_dim=2, shared_cnn=True, fuse="attn").to(device)

# # 例子
# # x_imgs:    (B,3,64,60)
# # x_scalars: (B,3,2)
# pred = model(x_imgs, x_scalars)

# loss = criterion(pred.squeeze(-1), y)  # y: (B,)
