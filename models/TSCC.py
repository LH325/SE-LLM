import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=896, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        noise = x - reconstructed
        return noise, reconstructed, mu, logvar

class CrossModalAligner(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.QK = nn.Linear(1000, 896)
    def forward(self, word, time_seq):
        time_seq = time_seq.mean(1).unsqueeze(0)
        word = word.unsqueeze(0)

        Q = self.query(word)
        K = self.key(time_seq)
        V = self.value(time_seq)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        aligned_time = torch.matmul(attn_weights, V)
        aligned_time = aligned_time.squeeze(0)

        return aligned_time
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.QK = nn.Linear(1000, 896)
    def forward(self, time_seq, word):
        B,_,_ = time_seq.shape
        text_emb = word.unsqueeze(0)

        Q = self.query(time_seq)
        K = self.key(text_emb)
        V = self.value(text_emb)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        aligned_text = torch.matmul(attn_weights, V)


        return aligned_text
class GatedFusion(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
    def l2_normalize(self, x, dim=-1, eps=1e-8):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    def forward(self, time_feat, text_feat, word,s):
        time_mean = time_feat.mean(dim=1)
        word_norm = self.l2_normalize(word, dim=1)
        time_norm = self.l2_normalize(time_mean, dim=1)
        sim_matrix = torch.matmul(time_norm, word_norm.T)
        _, topk_idx = torch.topk(sim_matrix, k=32, dim=1)
        selected_text = word[topk_idx]
        enhanced_text = selected_text.mean(dim=1).unsqueeze(1)
        enhanced_text = torch.softmax(enhanced_text,dim=1)
        enhanced_align = text_feat * enhanced_text
        combined = torch.cat([time_feat, enhanced_align], dim=-1)
        gate = self.gate_net(combined)
        fused = gate * time_feat + (1 - gate) * text_feat
        return self.out_proj(fused)
class AlignFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CrossModalAttention()
        self.t2t = CrossModalAligner()
        self.fusion = GatedFusion()

        self.noise = VAE()

    def forward(self, time_data, text_emb):

        aligned_text = self.attention(time_data, text_emb)
        noise, reconstructed, mu, logvar = self.noise(aligned_text)

        aligned_text = aligned_text - noise
        fused_output = self.fusion(time_data, aligned_text, text_emb,1)
        noise_output = self.fusion(time_data, noise, text_emb, 2)

        fused_output = fused_output+noise_output


        return fused_output

