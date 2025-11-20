import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import get_activation
from typing import Dict, Tuple, Optional


class Conv1dBlock(nn.Module):
    """
    A single convolutional processing block for 1D sequence data.

    Structure:
        Conv1d → (BatchNorm) → Activation → (Pooling)

    During training, the block records **activation regularization penalties**
    (L1, L2 norms of the convolution output) in `self.regularization_dict`.

    Attributes
    ----------
    conv : nn.Conv1d
        The convolution layer.
    batchnorm : Optional[nn.BatchNorm1d]
        Batch normalization layer, or None.
    activation_fn : Callable
        Activation function retrieved via `get_activation`.
    pool : Optional[nn.Module]
        MaxPool1d or AvgPool1d or None.
    regularization_dict : Dict[str, torch.Tensor]
        Stores activation penalty values computed during forward passes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        pooling: Optional[str],
        pooling_size: int,
        batch_norm: bool,
        activation: Optional[str],
    ) -> None:
        """
        Initialize a Conv1dBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Convolution kernel size.
        padding : int
            Padding applied to the convolution layer.
        pooling : {'max', 'avg', None}
            Optional pooling type.
        pooling_size : int
            Kernel size for pooling, if enabled.
        batch_norm : bool
            Whether to include BatchNorm1d.
        activation : str or None
            Activation function name used by `get_activation`.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.batchnorm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation_fn = get_activation(activation)

        if pooling == "max":
            self.pool = nn.MaxPool1d(kernel_size=pooling_size, ceil_mode=True)
        elif pooling == "avg":
            self.pool = nn.AvgPool1d(kernel_size=pooling_size, ceil_mode=True)
        else:
            self.pool = None

        self.regularization_dict: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Conv1dBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, length).

        Returns
        -------
        torch.Tensor
            Output tensor after conv → norm → activation → pooling.

        Notes
        -----
        - During training mode, activity regularization penalties
          are computed and stored.
        """
        conv_out = self.conv(x)

        if self.training:
            self._compute_regularization(conv_out)

        z = conv_out
        if self.batchnorm is not None:
            z = self.batchnorm(z)

        z = self.activation_fn(z)

        if self.pool is not None:
            z = self.pool(z)

        return z

    def _compute_activity_penalties(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L1 and L2 activation penalties.

        Parameters
        ----------
        x : torch.Tensor
            Activation tensor.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            (L1_norm, L2_norm)
        """
        l1_loss = torch.norm(x, p=1)
        l2_loss = torch.norm(x, p=2)
        return l1_loss, l2_loss

    def _compute_regularization(self, conv_out: torch.Tensor) -> None:
        """
        Compute and store activation-based regularization penalties.

        Parameters
        ----------
        conv_out : torch.Tensor
            Raw output of the convolution layer.

        Stores
        ------
        conv_activity_l1 : torch.Tensor
            L1 norm of convolution activations.
        conv_activity_l2 : torch.Tensor
            L2 norm of convolution activations.
        """
        l1_conv_activity, l2_conv_activity = self._compute_activity_penalties(conv_out)
        self.regularization_dict = {
            "conv_activity_l1": l1_conv_activity,
            "conv_activity_l2": l2_conv_activity,
        }


class SimpleCNN(nn.Module):
    """
    A simple 1D convolutional neural network with optional multiple conv layers
    and a small fully-connected output head.

    Network structure:
        Conv1dBlock → (additional Conv1dBlocks) → GlobalAvgPool → Dropout →
        fc1 → GELU → fc2 → output_activation

    Supports:
        - L1/L2 weight penalties for all Conv1d and Linear layers.
        - L1/L2 activity penalties for the first convolution layer.

    Regularization hyperparameters control how strongly each penalty contributes
    to the final scalar returned by `get_regularization_penalty()`.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_filters: int = 16,
        num_conv_layers: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.2,
        conv_activation: str = "gelu",
        mlp_activation: str = "gelu",
        output_activation: str = "none",
        batch_norm: bool = True,
        pooling: Optional[str] = None,
        pooling_size: int = 5,
        weight_penalty_l1: float = 0.0,
        weight_penalty_l2: float = 0.0,
        conv_activity_penalty_l1: float = 0.0,
        conv_activity_penalty_l2: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Initialize the SimpleCNN model.

        Parameters
        ----------
        in_channels : int
            Number of input channels to the first conv layer.
        num_filters : int
            Number of output channels for convolution layers.
        num_conv_layers : int
            Number of convolutional layers.
        kernel_size : int
            Kernel size for all convolutional layers.
        dropout : float
            Dropout rate applied before MLP head.
        conv_activation : str
            Activation function name for convolution blocks.
        mlp_activation : str
            Activation function name for MLP blocks.
        output_activation : str
            Activation function name applied to final output.
        batch_norm : bool
            Whether to include batch normalization.
        pooling : {'max', 'avg', None}
            Optional pooling type.
        pooling_size : int
            Pooling kernel size.
        weight_penalty_l1 : float
            Scaling coefficient for L1 weight penalties.
        weight_penalty_l2 : float
            Scaling coefficient for L2 weight penalties.
        conv_activity_penalty_l1 : float
            Scaling coefficient for L1 activity penalties.
        conv_activity_penalty_l2 : float
            Scaling coefficient for L2 activity penalties.
        """
        super().__init__()

        padding = kernel_size // 2

        # First conv layer
        self.conv1 = Conv1dBlock(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            pooling=pooling,
            pooling_size=pooling_size,
            activation=conv_activation,
            batch_norm=batch_norm,
        )

        # Additional conv layers
        conv_layers = []
        for _ in range(1, num_conv_layers):
            conv_layers.append(nn.Dropout(dropout))
            conv_layers.append(
                Conv1dBlock(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    pooling=pooling,
                    pooling_size=pooling_size,
                    activation=conv_activation,
                    batch_norm=batch_norm,
                )
            )
        self.conv_layers = nn.Sequential(*conv_layers)

        # MLP head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.mlp_activation = get_activation(mlp_activation)
        self.fc2 = nn.Linear(num_filters, 1)
        self.output_activation = get_activation(output_activation)

        # Regularization storage
        self.regularization_dict: Dict[str, torch.Tensor] = {}

        # Regularization hyperparameters
        self.weight_penalty_l1 = weight_penalty_l1
        self.weight_penalty_l2 = weight_penalty_l2
        self.conv_activity_penalty_l1 = conv_activity_penalty_l1
        self.conv_activity_penalty_l2 = conv_activity_penalty_l2

    def forward(
        self,
        x_seq: torch.Tensor,
        x_struct: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN.

        Parameters
        ----------
        x_seq : torch.Tensor
            Sequence input tensor of shape (batch, channels, seq_len).
        x_struct : Optional[torch.Tensor]
            Structural input (unused). Included for API compatibility.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch,).
        """
        x = self.conv1(x_seq)
        x = self.conv_layers(x)  # No-op if empty

        # Global average pooling
        x = x.mean(dim=2)

        # MLP head
        x = self.dropout(x)
        x = self.mlp_activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))
        x = x.squeeze(-1)
        return x

    def _compute_regularization(self) -> None:
        """
        Compute and store regularization penalties.

        Computes:
        ----------
        - L1/L2 weight penalties for *all* Conv1d and Linear layers.
        - L1/L2 activity penalties for the first conv layer.

        Stores in self.regularization_dict:
        ------------------------------------
        weight_loss_l1 : torch.Tensor
            Sum of L1 norms of weights across all Conv1d + Linear layers.
        weight_loss_l2 : torch.Tensor
            Sum of L2 norms of weights across all Conv1d + Linear layers.
        conv_activity_loss_l1 : torch.Tensor
            L1 activation penalty from `conv1`, or zero if unavailable.
        conv_activity_loss_l2 : torch.Tensor
            L2 activation penalty from `conv1`, or zero if unavailable.
        """
        device = next(self.parameters()).device

        # -------------------------
        # Weight penalties
        # -------------------------
        weight_loss_l1 = torch.tensor(0.0, device=device)
        weight_loss_l2 = torch.tensor(0.0, device=device)

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                w = module.weight
                weight_loss_l1 += torch.norm(w, p=1)
                weight_loss_l2 += torch.norm(w, p=2)

        # -------------------------
        # Activity penalties
        # -------------------------
        conv_act_l1 = torch.tensor(0.0, device=device)
        conv_act_l2 = torch.tensor(0.0, device=device)

        if self.conv1.regularization_dict:
            conv_act_l1 = self.conv1.regularization_dict.get(
                "conv_activity_l1", conv_act_l1
            ).to(device)
            conv_act_l2 = self.conv1.regularization_dict.get(
                "conv_activity_l2", conv_act_l2
            ).to(device)

        # Store results
        self.regularization_dict = {
            "weight_loss_l1": weight_loss_l1,
            "weight_loss_l2": weight_loss_l2,
            "conv_activity_loss_l1": conv_act_l1,
            "conv_activity_loss_l2": conv_act_l2,
        }

    def get_regularization_penalty(self) -> torch.Tensor:
        """
        Return a single scalar combining all regularization terms.

        Returns
        -------
        torch.Tensor
            Scalar representing:
                λ_w1 * ||W||₁ + λ_w2 * ||W||₂
              + λ_a1 * ||A_conv1||₁ + λ_a2 * ||A_conv1||₂

        Notes
        -----
        This method triggers `_compute_regularization()` internally,
        so it may be called directly after `forward()`.
        """
        self._compute_regularization()

        l1_wt = self.weight_penalty_l1 * self.regularization_dict["weight_loss_l1"]
        l2_wt = self.weight_penalty_l2 * self.regularization_dict["weight_loss_l2"]
        l1_act = self.conv_activity_penalty_l1 * self.regularization_dict["conv_activity_loss_l1"]
        l2_act = self.conv_activity_penalty_l2 * self.regularization_dict["conv_activity_loss_l2"]

        return l1_act + l2_act + l1_wt + l2_wt
