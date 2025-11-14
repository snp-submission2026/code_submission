import torch
import torch.nn as nn
from transformers import AutoModel

class BertWithFacets_FiLM(nn.Module):
    def __init__(self, facet_dim, hidden_dim, num_classes, num_languages,
                 dropout_rate=0.3, model_name="xlm-roberta-large"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        self.lang_emb = nn.Embedding(num_languages, 16)
        self.fnorm = nn.LayerNorm(facet_dim)
        self.fproj = nn.Sequential(
            nn.Linear(facet_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.text_attn = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.text_proj = nn.Linear(self.hidden_size, 256)
        self.facet_proj = nn.Linear(128, 256)

        self.lang_gamma_text = nn.Linear(16, 256)
        self.lang_beta_text  = nn.Linear(16, 256)
        self.lang_gamma_facet = nn.Linear(16, 256)
        self.lang_beta_facet  = nn.Linear(16, 256)

        self.gate = nn.Sequential(
            nn.Linear(256 * 2 + 16, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, facets, language, return_gate=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        attn = self.text_attn(hidden)
        text_vec = torch.sum(attn * hidden, dim=1)

        fvec = self.fproj(self.fnorm(facets))
        lang_vec = self.lang_emb(language)

        text_p = self.text_proj(text_vec)
        facet_p = self.facet_proj(fvec)

        gamma_t = self.lang_gamma_text(lang_vec)
        beta_t  = self.lang_beta_text(lang_vec)
        gamma_f = self.lang_gamma_facet(lang_vec)
        beta_f  = self.lang_beta_facet(lang_vec)

        text_p  = gamma_t * text_p + beta_t
        facet_p = gamma_f * facet_p + beta_f

        fusion_input = torch.cat([text_p, facet_p, lang_vec], dim=1)
        gate_val = self.gate(fusion_input)
        fused = gate_val * text_p + (1 - gate_val) * facet_p

        logits = self.fc(fused)
        return (logits, gate_val) if return_gate else logits
