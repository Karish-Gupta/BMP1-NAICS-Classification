import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class NAICSHierarchicalEnsemble(nn.Module):
    def __init__(self, 
                model_name="answerdotai/ModernBERT-large", 
                
                # BASED ON 2017 NAICS CODES: https://classification.codes/classifications/industry/naics-usa
                num_sectors=20,     # 2-digit
                num_subsectors=99, # 3-digit
                num_industries=1057 # 6-digit
                ):
        super(NAICSHierarchicalEnsemble, self).__init__()
        
        # Context Encoder (Name + Address)
        self.context_encoder = AutoModel.from_pretrained(model_name)
        
        # Enrichment Encoder (Scraped Text)
        self.enrichment_encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.context_encoder.config.hidden_size
        
        # Fusion Layer (Concatenating Stream A and Stream B)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Hierarchical Output Heads
        self.sector_head = nn.Linear(hidden_size, num_sectors)
        self.subsector_head = nn.Linear(hidden_size, num_subsectors)
        self.industry_head = nn.Linear(hidden_size, num_industries)

    def forward(self, context_input, enrichment_input):
        # Business context
        out_a = self.context_encoder(**context_input).last_hidden_state[:, 0, :]
        
        # Scraped Text - Use a Gating mechanism for empty text
        # If scraper failed, this stream is essentially zeros or a [MISSING] token
        out_b = self.enrichment_encoder(**enrichment_input).last_hidden_state[:, 0, :]
        
        # Fusion
        fused_repr = self.fusion(torch.cat((out_a, out_b), dim=1))
        
        # Predict at all 3 levels of the hierarchy
        sector_logits = self.sector_head(fused_repr)
        subsector_logits = self.subsector_head(fused_repr)
        industry_logits = self.industry_head(fused_repr)
        
        return sector_logits, subsector_logits, industry_logits