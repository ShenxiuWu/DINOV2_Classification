from torch import nn

class DinoVisionTransformerClassifier(nn.Module):
    """
    A classifier head for the DINOv2 Vision Transformer model.

    Args:
        dino_model (nn.Module): The DINOv2 model.
        config (dict): Configuration dictionary containing the number of output classes.

    Attributes:
        dino_model (nn.Module): The DINOv2 model.
        num_classes (int): Number of output classes.
        classifier (nn.Sequential): The classifier head.
    """
    def __init__(self, dino_model, embed_dim, num_classes):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.dino_model = dino_model
        for param in self.dino_model.parameters():
            param.requires_grad = False
        projtion_dim = self.dino_model.patch_embed.proj
        new_proj=nn.Sequential( #change the projectio
            nn.Conv2d(1, projtion_dim.out_channels, kernel_size=projtion_dim.kernel_size, stride=projtion_dim.stride, padding=projtion_dim.padding, bias=False),
            nn.Identity() )
        self.dino_model.patch_embed.proj=new_proj
        self.num_classes =num_classes
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dino_model(x)
        x = self.dino_model.norm(x)
        x = self.classifier(x)
        return x