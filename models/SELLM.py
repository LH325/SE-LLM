import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP
from .TSCC import AlignFusionModel
from transformers import AutoModel,AutoModelForCausalLM, AutoTokenizer

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

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.step = configs.token_len
        self.word_size = configs.word_size
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        # backbone LLM
        self.llm = AutoModel.from_pretrained("Qwen/Qwen-0.5B-GRPO")
        self.d_model = 896

        # freeze LLM parameters
        for _, p in self.llm.named_parameters():
            p.requires_grad = False

        # numeric token encoder/decoder
        self.ts_encoder = MLP(
            self.step, self.d_model,
            configs.mlp_hidden_dim,
            configs.dropout, configs.mlp_activation
        )
        self.ts_decoder = MLP(
            self.d_model, self.step,
            configs.mlp_hidden_dim,
            configs.dropout, configs.mlp_activation
        )

        # prompt prototypes
        emb_w = self.llm.get_input_embeddings().weight
        self.vocab_size = emb_w.shape[0]
        self.proto_fc = nn.Linear(self.vocab_size, self.word_size)

        self.fuser = AlignFusionModel()
        self.llm = add_time_adapter(self.llm)

    def _standardize(self, x: torch.Tensor):
        mu = x.mean(dim=1, keepdim=True).detach()
        xc = x - mu
        sig = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        xn = xc / sig
        return xn, mu, sig

    def _destandardize(self, y: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor, total_len: int):
        y = y * (sig[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        y = y + (mu[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        return y

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # normalize per-sample over time
        x_norm, mu, sig = self._standardize(x_enc)

        bsz, _, nvars = x_norm.shape

        # [B, T, N] -> [B, N, T]
        x_vars = x_norm.permute(0, 2, 1).contiguous()
        # [B*N, T]
        flat = x_vars.reshape(bsz * nvars, -1)

        # split into non-overlapping tokens
        tokens = flat.unfold(dimension=-1, size=self.step, step=self.step)
        n_tokens = tokens.shape[1]

        # numeric tokens -> embeddings
        ts_emb = self.ts_encoder(tokens)

        word_w = self.llm.get_input_embeddings().weight
        proto = self.proto_fc(word_w.t()).t()            


        fused = self.fuser(ts_emb, proto)


        h = self.llm(inputs_embeds=fused).last_hidden_state


        y_tokens = self.ts_decoder(h)

        y = y_tokens.reshape(bsz, nvars, -1).permute(0, 2, 1).contiguous()

        # de-normalize to original scale
        total_len = n_tokens * self.step
        y = self._destandardize(y, mu, sig, total_len)

        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)