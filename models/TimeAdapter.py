import torch
import torch.nn as nn
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