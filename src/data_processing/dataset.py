import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class VeritasDataset(Dataset):
    """
    PyTorch Dataset for the VeritasNP Multimodal Fake News Detection.
    Loads data from multiple CSV files and provides text, metadata, and labels.
    """
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing the CSV files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all CSV parts
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        df_list = [pd.read_csv(f, on_bad_lines='skip') for f in all_files]
        self.data_frame = pd.concat(df_list, ignore_index=True)
        
        # Drop rows missing crucial text or labels
        self.data_frame = self.data_frame.dropna(subset=['news_context', 'label'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # Extract features
        text = str(row['news_context'])
        label = int(row['label'])
        
        # Extract metadata (useful for graph building or tabular features)
        metadata = {
            'category': str(row.get('category', 'unknown')),
            'source_type': str(row.get('source_type', 'unknown')),
            'meta_intent': str(row.get('meta_intent', 'unknown')),
            'meta_style': str(row.get('meta_style', 'unknown')),
            'news_id': str(row.get('news_id', 'unknown'))
        }

        sample = {
            'text': text,
            'metadata': metadata,
            'label': torch.tensor(label, dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloader(data_dir: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    """
    Helper function to create a PyTorch DataLoader for the VeritasDataset.
    """
    dataset = VeritasDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    # Quick sanity check
    # Assumes the script is in src/data_processing and data is in the root's 'data set' folder
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(project_root, 'data set')
    
    if os.path.exists(test_dir):
        print(f"Testing dataset load from {test_dir}...")
        ds = VeritasDataset(test_dir)
        print(f"Successfully loaded dataset with {len(ds)} records.")
        sample = ds[0]
        print(f"\nFirst Sample:")
        print(f"Label: {sample['label']}")
        print(f"Metadata: {sample['metadata']}")
        print(f"Text Snippet: {sample['text'][:100].encode('utf-8', errors='ignore').decode('utf-8')}...")
    else:
        print(f"Data directory not found at {test_dir}. Please verify the path.")
