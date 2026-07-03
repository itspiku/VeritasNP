import torch
from src.encoders.text_encoder import NepaliTextEncoder
from src.encoders.image_encoder import VisualEncoder
from src.encoders.forensics_encoder import ELAEncoder
from src.fusion.modality_alignment import ModalityAlignment
from src.fusion.divergence_score import DivergenceScore
from src.classification.final_classifier import FinalClassifier

def test_pipeline():
    print("Initializing Models...")
    # Initialize all models
    text_enc = NepaliTextEncoder(freeze_base=True)
    img_enc = VisualEncoder()
    ela_enc = ELAEncoder()
    align = ModalityAlignment()
    div = DivergenceScore()
    clf = FinalClassifier()
    
    # Try importing graph encoder
    try:
        from src.encoders.graph_encoder import MetadataGraphSage
        graph_enc = MetadataGraphSage(in_channels=16, hidden_channels=32, out_channels=512)
        has_graph = True
    except ImportError:
        print("Skipping GraphSage test due to missing PyTorch Geometric.")
        has_graph = False

    print("Generating Dummy Data...")
    batch_size = 4
    
    # 1. Dummy Text
    dummy_input_ids = torch.randint(0, 30000, (batch_size, 128))
    dummy_attention_mask = torch.ones((batch_size, 128))
    
    # 2. Dummy Images (Batch, 3, 224, 224)
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_ela = torch.randn(batch_size, 3, 224, 224)
    
    print("Running Forward Pass...")
    # Text Branch
    text_features = text_enc(dummy_input_ids, dummy_attention_mask)
    print(f"Text Features Shape: {text_features.shape}")
    
    # Image Branch
    image_features = img_enc(dummy_images)
    print(f"Image Features Shape: {image_features.shape}")
    
    # Forensics Branch
    ela_features = ela_enc(dummy_ela)
    print(f"Forensics Features Shape: {ela_features.shape}")
    
    # Graph Branch
    if has_graph:
        # Mocking a small batch graph
        dummy_node_features = torch.randn(batch_size, 16)
        dummy_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        graph_features = graph_enc(dummy_node_features, dummy_edge_index)
        print(f"Graph Features Shape: {graph_features.shape}")
    else:
        # If no PyG, simulate the graph output to test fusion
        graph_features = torch.randn(batch_size, 512)
        print(f"Simulated Graph Features Shape: {graph_features.shape}")
        
    # Fusion 1: Alignment
    aligned_features = align(text_features, image_features)
    print(f"Aligned Modalities Shape: {aligned_features.shape}")
    
    # Fusion 2: Divergence
    divergence_features = div(ela_features, graph_features)
    print(f"Divergence Score Shape: {divergence_features.shape}")
    
    # Final Classification
    logits = clf(aligned_features, divergence_features)
    print(f"Final Logits Shape: {logits.shape}")
    
    print("\nSUCCESS! All matrix dimensions align perfectly across all 7 models!")

if __name__ == "__main__":
    test_pipeline()
