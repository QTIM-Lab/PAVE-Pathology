import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEmbedding(nn.Module):
    """
    Embeds spatial coordinates into feature space of dimension embed_dim (same as patch features).
    """
    def __init__(self, embed_dim=1024, n_coords=2):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(n_coords, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
      )

    def forward(self, coords):
        # Normalize coordinates to [-1, 1] relative to the current bag
        # Note: In production, consider normalizing by global slide dimensions
        min_coords = torch.min(coords, dim=0, keepdim=True)[0]
        max_coords = torch.max(coords, dim=0, keepdim=True)[0]
        norm_coords = (coords - min_coords) / (max_coords - min_coords + 1e-6)
        norm_coords = norm_coords * 2 - 1
        return self.embedding_layer(norm_coords)

class SufficiencyGate(nn.Module):
    """
    Lightweight attention + classifier to filter out insufficient bags.
    Decides between Grade 0 (Insufficient) and Grade 1/2/3/4 (Sufficient).
    Outputs p_sufficient and bag summary m_gate.
    """
    def __init__(self, embed_dim=1024, hidden_dim=512):
        super().__init__()
        self.gate_att = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )
        self.gate_classifier = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, h):
        a_gate = F.softmax(self.gate_att(h), dim=0)
        m_gate = torch.mm(a_gate.transpose(1, 0), h) # Global bag summary
        logit_sufficient = self.gate_classifier(m_gate)
        p_sufficient = torch.sigmoid(logit_sufficient)
        return p_sufficient, m_gate

class SpatialTransformer(nn.Module):
    """
    Adds positional information and refines features using self-attention.
    """
    def __init__(self, embed_dim=1024, n_coords=2, nhead=8, dim_feedforward=1024, dropout=0.25, use_pos_embed=True):
        super().__init__()
        self.use_pos_embed = use_pos_embed
        if self.use_pos_embed:
            self.pos_embed = PositionalEmbedding(embed_dim=embed_dim, n_coords=n_coords)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                batch_first=True
            ), 
            num_layers=1
        )

    def forward(self, h, coords=None):
        if self.use_pos_embed and coords is not None:
            h_pos = h + self.pos_embed(coords)
        else:
            h_pos = h
        h_refined = self.transformer(h_pos.unsqueeze(0)).squeeze(0)
        return h_refined

class DiagnosticAggregator(nn.Module):
    """
    Aggregates refined patch features into a bag-level diagnostic representation.
    """
    def __init__(self, embed_dim=1024, hidden_dim=512):
        super().__init__()
        self.diag_attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_refined):
        a_diag = F.softmax(self.diag_attention(h_refined), dim=0)
        m_diag = torch.mm(a_diag.transpose(1, 0), h_refined)
        return m_diag, a_diag

class OrdinalHead(nn.Module):
    """
    Ordinal classification for the diagnostic grades of continuous severity.
    Uses CORAL-style thresholds for consistent grading (Normal -> Low -> High -> Cancer).
    A single weight matrix with multiple biases for thresholds.
    """
    def __init__(self, embed_dim=1024, n_diagnostic_classes=4):
        super().__init__()
        self.ordinal_projector = nn.Linear(embed_dim, 1, bias=False)
        self.ordinal_biases = nn.Parameter(torch.zeros(n_diagnostic_classes - 1))
        
        # Initialize biases to spread out thresholds (e.g., 1, 0, -1)
        # This prevents the "all-or-nothing" collapse where the model can only predict min or max class.
        num_thresholds = n_diagnostic_classes - 1
        initial_values = torch.linspace(1, -1, steps=num_thresholds)
        with torch.no_grad():
            self.ordinal_biases.copy_(initial_values)

    def forward(self, m_diag):
        logit_base = self.ordinal_projector(m_diag)
        logits_ordinal = logit_base + self.ordinal_biases
        p_ordinal = torch.sigmoid(logits_ordinal)
        return p_ordinal, logits_ordinal

class SAO_MIL(nn.Module):
    """
    Spatially-Aware Ordinal MIL (SAO-MIL)
    Merges spatial reasoning (Transformer), diagnostic filtering (Sufficiency Gate),
    and rank-consistency (CORAL).
    """
    def __init__(self, embed_dim=1024, n_diagnostic_classes=4, dropout=0.25, use_pos_embed=True):
        super().__init__()
        # n_diagnostic_classes = 4 (Normal, Low, High, Cancer)
        # Total grades = 5 (0 is handled by the gate)
        
        self.L = embed_dim
        self.D = 512
        
        # 1. SUFFICIENCY GATE
        self.sufficiency_gate = SufficiencyGate(embed_dim=self.L, hidden_dim=self.D)

        # 2. SPATIAL & ATTENTION
        self.spatial_transformer = SpatialTransformer(embed_dim=self.L, dropout=dropout, use_pos_embed=use_pos_embed)
        self.diagnostic_aggregator = DiagnosticAggregator(embed_dim=self.L, hidden_dim=self.D)

        # 3. ORDINAL HEAD
        self.ordinal_head = OrdinalHead(embed_dim=self.L, n_diagnostic_classes=n_diagnostic_classes)

    def forward(self, h, coords, instance_eval=False, bag_label=None):
        # --- PHASE 1: SUFFICIENCY (Binary) ---
        p_sufficient, m_gate = self.sufficiency_gate(h)

        # --- PHASE 2: SPATIAL REFINEMENT ---
        h_refined = self.spatial_transformer(h, coords)

        # --- PHASE 3: DIAGNOSTIC AGGREGATION ---
        m_diag, a_diag = self.diagnostic_aggregator(h_refined)

        # --- PHASE 4: ORDINAL RANKS (CORAL) ---
        p_ordinal, logits_ordinal = self.ordinal_head(m_diag)

        # --- FUSION LOGIC ---
        # Final Grade 0 probability is (1 - p_sufficient)
        # Diagnostic probabilities (1-4) are scaled by p_sufficient
        final_diagnostic_probs = p_ordinal * p_sufficient 

        results = {
            'p_sufficient': p_sufficient,       # Grade 0 vs Others
            'p_ordinal': final_diagnostic_probs, # Cumulative ranks for 1, 2, 3, 4
            'm_diag': m_diag,                     # Latent vector for visualization
            'logits': logits_ordinal,
            'attention': a_diag
        }

        if instance_eval and bag_label is not None:
            results['inst_loss'] = self.ordinal_instance_eval(h_refined, a_diag, bag_label)
        
        return final_diagnostic_probs, results

    def ordinal_instance_eval(self, h, A, bag_label):
        """
        Instance-level auxiliary task:
        Ensures high-attention patches align with the bag's ordinal rank.
        """
        device = h.device
        # We only perform this for diagnostic slides (Grade 1-4)
        if bag_label.item() <= 0:
            return torch.tensor(0.0).to(device)

        k_sample = 8
        A = A.view(-1) # [N]
        k = min(k_sample, len(A))
        
        # Get top-k attended patches
        top_ids = torch.topk(A, k)[1]
        top_h = torch.index_select(h, dim=0, index=top_ids)
        
        # Predict ordinal rank for these patches
        top_probs, _ = self.ordinal_head(top_h)
        
        # Create rank-based targets
        # bag_label 1 (Normal) -> diag_label 0 -> target [0, 0, 0]
        # bag_label 2 (Low)    -> diag_label 1 -> target [1, 0, 0]
        diag_label = bag_label.item() - 1
        num_thresholds = self.ordinal_head.ordinal_biases.shape[0]
        
        target_rank = torch.zeros(num_thresholds).to(device)
        if diag_label > 0:
            target_rank[:diag_label] = 1.0
        
        target_rank = target_rank.repeat(k, 1)
        
        inst_loss = F.binary_cross_entropy(top_probs, target_rank)
        return inst_loss

    @staticmethod
    def calculate_loss(output, label):
        """
        Fused Loss:
        Loss = BCE(Sufficiency) + I(Sufficient) * OrdinalRankLoss(Diagnosis)
        """
        # label is 0, 1, 2, 3, or 4
        
        # 1. Sufficiency Loss (Binary: 0 vs [1,2,3,4])
        is_diag = (label > 0).float().view(-1, 1)
        loss_suff = F.binary_cross_entropy(output['p_sufficient'], is_diag)
        
        # 2. Ordinal Loss (Only for diagnostic slides)
        # We only compute this loss if there are actually diagnostic slides in the batch
        # But since we are likely doing batch size 1 (common in MIL), we check the label.
        
        if label.item() > 0:
            # Shift label 1-4 to 0-3 for the CORAL head
            diag_label = label.item() - 1
            # Generate target: e.g. Grade 2 (Low) -> [1, 0, 0]
            # Number of ordinal thresholds is n_diagnostic_classes - 1 = 3
            
            num_thresholds = output['p_ordinal'].shape[1] # Should be 3
            target_rank = torch.zeros(num_thresholds).to(label.device)
            if diag_label > 0:
                target_rank[:diag_label] = 1.0
            
            loss_diag = F.binary_cross_entropy(output['p_ordinal'], target_rank.unsqueeze(0))
        else:
            loss_diag = 0.0

        return loss_suff + loss_diag
