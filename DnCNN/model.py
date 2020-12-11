import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    Define the DnCNN neural network which takes noisy image image y and
    output the residual image v such that the estimation for the ground
    truth is y - v.
    """
    def __init__(self, depth=17, n_channels=64,
                 img_channel=1,
                 use_bnorm=True,
                 kernel_size=3):
        """
        Initialization. Build a VGG style network.

        Args:
            depth: depth of the network.
            n_channels:
            img_channel: number of color channels.
                         For grayscale images, img_channel=1.
                         For color images, img_channel=3.
            use_bnorm: it's been demonstrated that batch_normalization is good
                       for boosting performance.
            kernel_size: use small kernel size and remove all pooling layers.
                         If kernel_size = 3, the resulting receptive field size
                         is (2*depth+1, 2*depth+1).

        """
        super().__init__()

        # Define the parameters
        # TODO: why should we redefine the kernel_size if it's already the argument of the function?
        # kernel_size = 3
        padding = 1

        # Construct the network structure.
        layers = []

        # The first block is a Conv + ReLU. The Conv layer has bias.
        layers.append(nn.Conv2d(in_channels=img_channel, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # The next (depth - 2) blocks are repetitive: Conv + BatchNorm + ReLU.
        # The Conv layer doesn't have bias.
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(num_features=n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        # The last layer is a Conv layer.
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=img_channel,
                                kernel_size=kernel_size, padding=padding, bias=False))

        # Make a self.dncnn architecture.
        self.dncnn = nn.Sequential(*layers)

        # Initialize the weights
        self._initialize_weights()


    def _initialize_weights(self):
        # Initialize the weights of the Conv layer using orthogonal initialization.
        # The orthogonal initialization is used to robustly control the trade-off
        # between noise reduction and detail preservation,
        # which is used in the FFDNet paper published by the same author later.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight)
                # print("Orthogonal initialization for nn.Conv2d layer.")

                # Initialize the bias in the nn.Conv2d with constant 0.
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, x, mode="residual"):
        """
        Feed noisy image to the DnCNN network and output residual image.

        Args:
            x: noisy image.

        Returns:
            v: estimate of the residual image.

        Shape:
            x: (N, C, H, W), where N is the batch size,
                             C is the number of color channels (1 for grayscale and 3 for color images).
                             H: height of image. W: width of image.
            v: (N, C, H, W)
        """

       # Instead of directly predicting an estimate for the ground truth of eta,
       # it outputs the noise first and the x_hat is the noisy image - noise (x - v_hat).
       if mode.lower() == "residual":
           # Compute the estimate of the residual: v_hat.
           v_hat = self.dncnn(x)

           # Return the ground truth estimate.
           x_hat = x - v_hat
           return x_hat
        # Directly predict the clean image.
       elif mode.lower() == "direct":
           x_hat = self.dncnn(x)
           return x_hat
       else:
           assert mode in ["residual", "direct"], "You have to choose between ['residual', 'direct']."


