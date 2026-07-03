import torch
import torch.nn as nn
from transformers import AutoModel

class NepaliTextEncoder(nn.Module):
    def __init__(self, model_name="google/muril-base-cased", hidden_size=768, freeze_base=False):
        super(NepaliTextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        """
        Extracts the [CLS] token representation as the semantic feature vector.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output is the representation of the [CLS] token
        return outputs.pooler_output
