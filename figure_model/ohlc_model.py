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
            nn.Linear(46080, 2),
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