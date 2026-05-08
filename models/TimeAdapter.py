import torch
import torch.nn as nn
# Qwen
class TimeLayer(nn.Module):
    def __init__(self, original_layer, rank=16):
        super().__init__()
        self.original_layer = original_layer
        self.lora_A = nn.Linear(896, rank)
        self.lora_C = nn.LSTM(rank, 896,batch_first=True)
        self.lora_D = nn.LSTM(896, rank,batch_first=True)
        self.lora_B = nn.Linear(rank, 128)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        A = self.lora_A(x)
        C,(_,_) = self.lora_C(A)
        D,(_,_) = self.lora_D(C)
        B = self.lora_B(D)

        return original_output + B

def add_time_adapter(model, rank=8):
    for layer in model.layers:

        layer.self_attn.k_proj = TimeLayer(layer.self_attn.k_proj, rank)
        layer.self_attn.v_proj = TimeLayer(layer.self_attn.v_proj, rank)
    return model
# LLama

class TimeLayer(nn.Module):
    """
    TimeAdapter layer for injecting temporal dependency modeling into LLaMA projections.

    This module wraps the original projection layer with a lightweight temporal adaptation branch.
    The original LLaMA projection is preserved, while an additional low-rank temporal branch is
    introduced to capture sequential patterns through LSTM-based transformations.

    Note:
        Since LLaMA is sensitive to large parameter perturbations, applying TimeAdapter to too many
        layers may affect the original language-modeling capability. In our experiments, applying
        TimeAdapter to a small number of layers, typically 2--6 layers, provides a good trade-off
        between temporal modeling ability and preservation of the pretrained LLaMA representations.
    """

    def __init__(self, original_layer, rank=16):
        super().__init__()

        self.original_layer = original_layer

        self.lora_A = nn.Linear(896, rank)

        
        self.lora_C = nn.LSTM(rank, 896, batch_first=True)

        self.lora_D = nn.LSTM(896, rank, batch_first=True)

        self.lora_B = nn.Linear(rank, 128)

        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
     
        original_output = self.original_layer(x)

        A = self.lora_A(x)
        C, (_, _) = self.lora_C(A)
        D, (_, _) = self.lora_D(C)
        B = self.lora_B(D)
        
        return original_output + B


def add_time_adapter(model, rank=128, num_layers=2):


    # Apply TimeAdapter to the key and value projections of the selected layers.
    # We modify k_proj and v_proj because they directly affect temporal dependency
    # extraction and information aggregation in the attention mechanism.
    for i in range(0, num_layers):
        layer = model.model.layers[i]

        layer.self_attn.k_proj = TimeLayer(layer.self_attn.k_proj, rank)
        layer.self_attn.v_proj = TimeLayer(layer.self_attn.v_proj, rank)

    return model
  
