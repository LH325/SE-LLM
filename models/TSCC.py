import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


# =========================
# AM-VAE Module in TSCC
# =========================
# The following VAE configuration is used for all long-term forecasting
# experiments reported in our paper. The input dimension is set to 896
# to match the hidden representation size used in our experimental setup,
# and the latent dimension is set to 8 for a lightweight latent bottleneck.
        
# These hyperparameters are not fixed architectural constraints. They can
# be adjusted for different datasets, forecasting tasks, or backbone model
# structures according to the representation dimension, task complexity,
# and computational budget.
class VAE(nn.Module):
    def __init__(self, input_dim=896, latent_dim=8):
        super().__init__()

        # Latent Feature Projection:
        # This corresponds to the encoder Fe(C) in the AM-VAE description.
        # It projects the joint temporal-semantic representation C into a
        # lower-dimensional hidden feature space.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )

        # Latent Distribution Estimation:
        # These two linear layers estimate the latent mean mu and log-variance
        # logvar from the encoded joint-space representation.
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        # Anomaly Semantic Modeling:
        # The decoder Fd(z) reconstructs the anomaly-related semantic component DC
        # from the sampled latent variable z.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def reparameterize(self, mu, logvar):
        # Reparameterization Trick:
        # z = mu + epsilon * sigma, where epsilon is sampled from N(0, I).
        # This enables stochastic sampling in the latent semantic space while
        # keeping the sampling process differentiable during training.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x is the Joint Space C obtained from cross-modality alignment.
        # In the paper, C is the cross-attention output CrossAttn(H, S).
        h = self.encoder(x)

        # Estimate latent distribution parameters.
        mu, logvar = self.fc_mu(h), self.fc_var(h)

        # Sample latent variable z from the estimated distribution.
        z = self.reparameterize(mu, logvar)

        # Decode z to reconstruct the anomaly-related semantic component DC.
        DC = self.decoder(z)

        # Anomaly Decomposition:
        # DA represents the de-anomalized semantic component:
        # DA = C - DC.
        DA = x - DC

        # Return both decomposed components and latent distribution parameters.
        return DA, DC, mu, logvar


# =========================
# Optional Reverse Alignment
# =========================
class CrossModalAligner(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # for ETTh
        # self.QK = nn.Linear(1500, 896)
        self.QK = nn.Linear(1000, 896)

    def forward(self, word, time_seq):
        # This optional module performs the reverse alignment direction:
        # semantic/text representations are used as queries to retrieve
        # temporally aggregated features.
        time_seq = time_seq.mean(1).unsqueeze(0)
        word = word.unsqueeze(0)

        Q = self.query(word)
        K = self.key(time_seq)
        V = self.value(time_seq)

        # Scaled dot-product cross-attention.
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Text-guided temporal alignment output.
        aligned_time = torch.matmul(attn_weights, V)
        aligned_time = aligned_time.squeeze(0)

        return aligned_time


# =========================
# Cross-Modality Alignment
# =========================
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()
        self.d_model = d_model

        # Linear projections for cross-attention.
        # In TSCC, temporal embeddings H are used as queries,
        # while semantic embeddings S are used as keys and values.
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # for ETTh
        # self.QK = nn.Linear(1500, 896)
        self.QK = nn.Linear(1000, 896)

    def forward(self, time_seq, word):
        # time_seq corresponds to TS embeddings H with shape [B, N, C].
        # word corresponds to the semantic space S with shape [V, C] or [Ks, C].
        B, _, _ = time_seq.shape
        text_emb = word.unsqueeze(0)

        # Cross-Modality Alignment:
        # Q = H, K = S, V = S.
        # This implements C = CrossAttn(H, S), where C is the Joint Space.
        Q = self.query(time_seq)
        K = self.key(text_emb)
        V = self.value(text_emb)

        # Compute temporal-semantic attention weights.
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Joint Space C:
        # aligned_text is the semantic representation aligned to temporal embeddings.
        aligned_text = torch.matmul(attn_weights, V)

        return aligned_text


# =========================
# Structural Prior Infusion
# + Channel Dependency Enhancement
# + Gated Fusion
# =========================
class GatedFusion(nn.Module):
    def __init__(self, d_model=896):
        super().__init__()

        # Channel Dependency Enhancement:
        # The gate network corresponds to the MLP used to generate channel-wise
        # gates conditioned on TS embeddings and enhanced semantic representations.
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # F_llm:
        # This linear projection maps the fused representation to the token space
        # expected by the LLM.
        self.out_proj = nn.Linear(d_model, d_model)

    def l2_normalize(self, x, dim=-1, eps=1e-8):
        # L2 normalization used when computing the temporal-semantic similarity matrix M.
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def forward(self, time_feat, text_feat, word, s):
        # time_feat corresponds to temporal embeddings H.
        # text_feat corresponds to one AM-VAE component, either DC or DA.
        # word corresponds to semantic prototypes S.

        # Structural Prior Infusion:
        # Average temporal embeddings along the segment dimension:
        # Mean_N(H).
        time_mean = time_feat.mean(dim=1)

        # Normalize semantic prototypes S and temporal features Mean_N(H).
        # This corresponds to:
        # M = Norm2(Mean_N(H)) x Norm2(S)^T.
        word_norm = self.l2_normalize(word, dim=1)
        time_norm = self.l2_normalize(time_mean, dim=1)

        # Compute sample-level temporal-semantic similarity matrix M.
        sim_matrix = torch.matmul(time_norm, word_norm.T)

        # Select top-K semantic prototypes according to the similarity matrix.
        # These selected prototypes serve as structural semantic cues.
        _, topk_idx = torch.topk(sim_matrix, k=32, dim=1)
        selected_text = word[topk_idx]

        # Aggregate the selected top-K semantic prototypes into a structural prior.
        enhanced_text = selected_text.mean(dim=1).unsqueeze(1)

        # Optionally normalize the aggregated top-K semantic cues over the feature dimension.
        # For different datasets or model architectures, directly multiplying raw semantic embeddings
        # may introduce amplitude perturbations or additional noise. Applying softmax converts the
        # selected semantic cues into a smoother structural prior before they are applied to the
        # aligned text features. In practice, this normalization step can be enabled or disabled
        # depending on the training setting and task sensitivity.
        enhanced_text = torch.softmax(enhanced_text, dim=1)

        # Apply the aggregated semantic prior to the AM-VAE output component.
        # This corresponds to enhancing DA or DC with the selected semantic prior.
        enhanced_align = text_feat * enhanced_text

        # Channel Dependency Enhancement:
        # Concatenate temporal embeddings H with the enhanced semantic representation.
        # This corresponds to MLP([H, DA]) or MLP([H, DC]) in the paper.
        combined = torch.cat([time_feat, enhanced_align], dim=-1)

        # Generate channel-wise gates conditioned on temporal representations.
        gate = self.gate_net(combined)

        # Gated Fusion:
        # Fuse temporal embeddings and semantic components:
        # gate * H + (1 - gate) * DA/DC.
        fused = gate * time_feat + (1 - gate) * text_feat

        # Map the enhanced semantics to the LLM token space.
        return self.out_proj(fused)


# =========================
# TSCC Module
# =========================
class AlignFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Cross-Modality Alignment module:
        # It aligns temporal embeddings H with semantic space S and produces Joint Space C.
        self.attention = CrossModalAttention()

        # CrossModalAligner performs the reverse alignment direction compared with
        # CrossModalAttention. While CrossModalAttention uses temporal representations
        # as queries to retrieve semantic/text features, CrossModalAligner uses
        # semantic/text representations as queries to align temporal features.

        # In our long-term forecasting experiments, this additional text-guided temporal
        # alignment did not consistently bring performance improvements, and therefore
        # it is not the key component used to obtain the reported main results.
        # However, we keep this module as an optional complementary design, since it may
        # be beneficial for other tasks, datasets, or backbone architectures when combined
        # with the proposed framework.
        self.t2t = CrossModalAligner()

        # Gated fusion module:
        # Used for both the anomaly semantic branch DC and the de-anomaly semantic branch DA.
        self.fusion = GatedFusion()

        # AM-VAE module:
        # It decomposes Joint Space C into DA and DC.
        self.noise = VAE()

    def forward(self, time_data, text_emb):
        # time_data corresponds to TS embeddings H.
        # text_emb corresponds to semantic space S.

        # Step 1: Cross-Modality Alignment.
        # Obtain Joint Space C = CrossAttn(H, S).
        aligned_text = self.attention(time_data, text_emb)

        # Step 2: AM-VAE-based Anomaly Pattern Modeling.
        # Decompose Joint Space C into de-anomaly semantic DA and anomaly semantic DC.
        DA, _, mu, logvar = self.noise(aligned_text)

        # aligned_text is the VAE input C and is not modified by the VAE.
        # The VAE reconstructs the anomaly-related component DC_decoded and
        # computes DA = C - DC_decoded. Before fusion, we recover DC using the
        # equivalent residual form DC = C - DA to explicitly preserve the numerical
        # decomposition consistency C = DA + DC. This does not change the AM-VAE
        # formulation in the paper.

        DC = aligned_text - DA

        # Step 3: Gated Fusion for the anomaly semantic branch.
        # This corresponds to generating GC from DC.
        fused_output = self.fusion(time_data, DC, text_emb, 1)

        # Step 4: Gated Fusion for the de-anomaly semantic branch.
        # This corresponds to generating GA from DA.
        noise_output = self.fusion(time_data, DA, text_emb, 2)

        # Step 5: Feature Fusion.
        # Fuse anomaly-enhanced and de-anomaly-enhanced semantics:
        # GA + GC, which is then used as the enhanced semantic representation
        # for subsequent LLM processing.
        fused_output = fused_output + noise_output

        return fused_output
