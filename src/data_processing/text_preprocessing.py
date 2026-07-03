import torch
from transformers import AutoTokenizer
import re

class TextPreprocessor:
    def __init__(self, model_name: str = "google/muril-base-cased", max_length: int = 256):
        """
        Initializes the text preprocessor with a HuggingFace tokenizer.
        Default is MuRIL (Multilingual Representations for Indian Languages) which has strong Nepali support.
        Another option could be 'Shushant/nepaliBERT'.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        Basic normalization for Nepali text.
        Removes unnecessary whitespaces and basic HTML tags if any.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags (if any crept in during scraping)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Further Nepali-specific normalization could be added here
        # e.g., standardizing devanagari numerals, etc.
        return text

    def tokenize(self, texts: list):
        """
        Tokenizes a list of strings and returns PyTorch tensors (input_ids, attention_mask).
        """
        # First clean the text
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        # Tokenize using HuggingFace tokenizer
        encoding = self.tokenizer(
            cleaned_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

if __name__ == "__main__":
    # Simple test
    print("Initializing TextPreprocessor with MuRIL tokenizer...")
    preprocessor = TextPreprocessor()
    
    sample_text = [
        "यो एउटा नक्कली समाचार हो जसले मानिसहरूलाई भ्रमित पार्न सक्छ।",  # This is a fake news that can confuse people.
        "नेपाल राष्ट्र बैंकले नयाँ नीति जारी गरेको छ। <br> विस्तृत जानकारी तल दिइएको छ।" # NRB has issued a new policy...
    ]
    
    print("\nOriginal Text:")
    for text in sample_text:
        # Avoid UnicodeEncodeError in Windows cmd by encoding first
        safe_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        print(f" - {safe_text}")
        
    print("\nTokenizing...")
    tokens = preprocessor.tokenize(sample_text)
    
    print("\nTokenization complete!")
    print(f"Input IDs shape: {tokens['input_ids'].shape}")
    print(f"Attention Mask shape: {tokens['attention_mask'].shape}")
    
    # Show decoded tokens for the first sample to verify tokenizer works on Nepali
    decoded_0 = preprocessor.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    # Filter out padding for cleaner display
    decoded_0 = [token for token in decoded_0 if token not in [preprocessor.tokenizer.pad_token]]
    
    # Safe print for the tokens
    safe_decoded = [t.encode('utf-8', errors='ignore').decode('utf-8') for t in decoded_0]
    print(f"\nDecoded tokens for first sample:\n{safe_decoded}")
