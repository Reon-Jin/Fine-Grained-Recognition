import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification
from config import Config


class ArcMarginProduct(nn.Module):
    """
    Large Margin ArcFace implementation with device-safe buffers.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin

        m_tensor = torch.tensor(self.m, dtype=torch.float32)
        pi_tensor = torch.tensor(3.141592653589793, dtype=torch.float32)
        self.register_buffer('cos_m', torch.cos(m_tensor))
        self.register_buffer('sin_m', torch.sin(m_tensor))
        self.register_buffer('th',    torch.cos(pi_tensor - m_tensor))
        self.register_buffer('mm',    torch.sin(pi_tensor - m_tensor) * m_tensor)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class TransFGDeiT(nn.Module):
    """
    TransFGDeiT Enhanced for fine-grained classification with multi-layer attention fusion.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        cfg = Config()

        self.warmup_epochs = getattr(cfg, 'WARMUP_EPOCHS', 5)
        self.current_epoch = 0

        model_cfg = AutoConfig.from_pretrained(
            'facebook/deit-base-patch16-224',
            output_attentions=True,
            output_hidden_states=True,
            num_labels=num_classes
        )
        setattr(model_cfg, 'drop_path_rate', getattr(cfg, 'DROP_PATH_RATE', 0.1))
        self.deit = AutoModelForImageClassification.from_pretrained(
            'facebook/deit-base-patch16-224',
            config=model_cfg,
            ignore_mismatched_sizes=True
        )
        D = model_cfg.hidden_size

        self.num_parts = cfg.NUM_PARTS
        self.pre_attn_ln  = nn.LayerNorm(D)
        self.part_attention = nn.MultiheadAttention(embed_dim=D, num_heads=cfg.ATTN_HEADS, dropout=cfg.PART_DROPOUT)
        self.post_attn_ln = nn.LayerNorm(D)
        self.weight_pool = nn.Linear(D, 1)
        self.pool_temp = nn.Parameter(torch.tensor(1.0))
        self.part_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.PART_DROPOUT),
        )
        self.fusion_dropout = nn.Dropout(cfg.PART_DROPOUT)
        self.classifier_dropout = nn.Dropout(getattr(cfg, 'CLS_DROPOUT', 0.5))

        s_val = getattr(cfg, 'ARC_S', 30.0)
        m_val = getattr(cfg, 'ARC_M', 0.50)
        self.arcface = ArcMarginProduct(D * 2, num_classes, s=s_val, m=m_val, easy_margin=False)

    def forward(self, x: torch.Tensor, labels=None):
        out = self.deit(
            pixel_values=x,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        hidden = out.hidden_states[-1]
        attns = out.attentions

        B, Np1, D = hidden.size()
        cls_embed = hidden[:, 0]
        patch_embeds = hidden[:, 1:]

        # Multi-layer attention fusion (last 4 layers)
        fused_scores = []
        for layer in attns[-4:]:
            # mean over heads then cls→patch
            fused_scores.append(layer.mean(dim=1)[:, 0, 1:])
        attn_scores = torch.stack(fused_scores, dim=0).mean(dim=0)  # [B, N]

        idx = torch.topk(attn_scores, self.num_parts, dim=1).indices
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
        parts = patch_embeds.gather(1, idx_exp)

        parts = self.pre_attn_ln(parts)
        t = parts.permute(1, 0, 2)
        if self.current_epoch >= self.warmup_epochs:
            attn_out, _ = self.part_attention(t, t, t)
        else:
            attn_out = t
        attn_out = attn_out.permute(1, 0, 2)
        attn_out = self.post_attn_ln(attn_out)

        raw_w = self.weight_pool(attn_out).squeeze(-1)
        weights = torch.softmax(raw_w / self.pool_temp.clamp_min(0.3), dim=1)
        part_feat = torch.sum(attn_out * weights.unsqueeze(-1), dim=1)
        part_feat = self.part_mlp(part_feat)

        fusion = torch.cat([cls_embed, part_feat], dim=1)
        fusion = self.fusion_dropout(fusion)
        fusion = self.classifier_dropout(fusion)

        infer_logits = F.linear(F.normalize(fusion), F.normalize(self.arcface.weight)) * self.arcface.s
        train_logits = self.arcface(fusion, labels) if labels is not None else infer_logits

        return train_logits, {
            'cls': cls_embed,
            'part': part_feat,
            'attn_scores': attn_scores,
            'logits_infer': infer_logits,
        }
