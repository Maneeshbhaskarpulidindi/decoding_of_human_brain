import torch
import torch.nn as nn

# Define TemporalBlock
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super().__init__()
        layers = []
        for i,dilation in enumerate(dilation_rates):
            layers.append(
                nn.Conv1d(
                    in_channels if i==0 else out_channels, 
                    out_channels, kernel_size=kernel_size, 
                    padding=(kernel_size-1 // 2) * dilation, dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels  # Update for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define SpatialBlock
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels  # Update for next layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)


# Define Encoder
class EEGEncoder(nn.Module):
    def __init__(self, num_channels=128, temp_channels=64, spatial_channels=32, kernel_size=3, dilation_rates=[1,2,4,8], num_residual_blocks=2, num_classes=40):
        super().__init__()

        # Temporal Block to capture temporal dependencies
        self.temporal_block = TemporalBlock(num_channels, temp_channels, kernel_size, dilation_rates)

        # Spatial Block to capture spatial dependencies among channels
        self.spatial_block = SpatialBlock(1, spatial_channels, num_layers=2)  # Conv2d expects (batch, channel, height, width)

        # Residual Blocks for added depth
        self.res_blocks = nn.ModuleList([ResidualBlock(spatial_channels) for _ in range(num_residual_blocks)])
        # Fully connected layer for classification
        self.fc = nn.Linear(spatial_channels, num_classes)

    def forward(self, x):
        # Pass through Temporal Block
        x = self.temporal_block(x)  # Shape: (batch_size, temp_channels, time_steps)

        # Reshape for Spatial Block (treat time as "width" in Conv2d)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, temp_channels, time_steps)
        x = self.spatial_block(x)  # Shape: (batch_size, spatial_channels, temp_channels, time_steps)

        # Pass through Residual Blocks
        for res_block in self.res_blocks:
            x = res_block(x)  # Shape maintained
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        # Final classification layer
        x = self.fc(x)  # Shape: (batch_size, num_classes)

        return x


# Example usage
"""if __name__ == "__main__":
    # Instantiate the model
    #model = EEGEncoder(
        num_channels=128,       # EEG channels
        temp_channels=64,       # Temporal features
        spatial_channels=32,    # Spatial features
        kernel_size=3,          # Kernel size for temporal block
        dilation_rates=[1, 2, 4, 8],  # Dilation rates for temporal block
        num_residual_blocks=2,  # Number of residual blocks
        num_classes=40          # Number of classes for classification
    )

    # Sample input with shape (batch_size, num_channels, time_steps)
    x = torch.randn(16, 128, 440)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected shape: (16, 40)
"""