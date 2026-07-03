# VeritasNP: Multimodal Fake News Detection for Nepali

VeritasNP is an advanced, multimodal fake news detection system designed specifically for the Nepali language. It leverages a novel architecture that combines text semantics, visual context, image forensics, and metadata relationships to robustly classify news articles as real or fake.

## 🏗️ Architecture Overview

The system uses a highly integrated multimodal approach to analyze multiple aspects of a news article simultaneously:

1. **Text Branch (Semantic Analysis)**
   - **Preprocessing**: Text undergoes tokenization and normalization.
   - **Encoder**: Uses language-specific models (MuRIL or NepBERT) to extract rich semantic features from the Nepali text.

2. **Image Branch (Visual & Contextual Analysis)**
   - **Preprocessing**: Images are resized and normalized.
   - **Encoder**: Uses vision-language models (CLIP) or standard vision models (EfficientNet) to extract semantic visual features.

3. **Forensics Branch (Manipulation Detection)**
   - **Preprocessing**: Error Level Analysis (ELA) is extracted from the image to highlight compression artifacts and detect tampering.
   - **Encoder**: A ResNet-50 model encodes the ELA map to identify regions of digital manipulation.

4. **Metadata Branch (Network & Source Context)**
   - **Preprocessing**: Metadata (source credibility, temporal data, propagation graphs) is structured into a graph.
   - **Encoder**: GraphSage encodes these relationships into dense representations.

5. **Fusion and Classification**
   - **Modality Alignment**: Semantically aligns features from the Text Encoder and Image Encoder.
   - **Divergence Score**: Computes a score based on conflicts detected between the Forensics Encoder and the Graph Encoder.
   - **Final Classifier**: Fuses the aligned multimodal representations and the divergence score to output a binary Real/Fake prediction.

## 📂 Project Structure

The project code is modularized to reflect the architecture components:

```
VeritasNP/
├── data/                      # Raw and processed datasets, including images and CSVs
├── notebooks/                 # Exploratory data analysis (EDA) and prototyping
├── src/                       # Core model source code
│   ├── data_processing/       # Tokenization, image resizing, ELA extraction, graph building
│   ├── encoders/              # Text (MuRIL/NepBERT), Image (CLIP), Forensics (ResNet), Graph (GraphSage)
│   ├── fusion/                # Modality alignment and Divergence score logic
│   ├── classification/        # Final classification layers
│   ├── training/              # Training loops, loss functions, and evaluation scripts
│   └── utils/                 # Metrics and common utilities
├── api/                       # FastAPI backend to serve the model
├── app/                       # Streamlit frontend web application
└── bot/                       # Telegram bot for user interactions
```

## 🚀 Execution Steps (How to Complete the Project)

### Step 1: Environment Setup
1. Create a Python virtual environment: `python -m venv venv`
2. Activate the virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in necessary environment variables (API keys, DB paths).

### Step 2: Data Preparation (`src/data_processing/`)
1. **Text**: Implement normalization and tokenization for Nepali text in `text_preprocessing.py`.
2. **Image**: Implement image loading, resizing, and normalization in `image_preprocessing.py`.
3. **Forensics**: Implement Error Level Analysis (ELA) generation to highlight image tampering.
4. **Graph Building**: Construct metadata graphs from raw CSVs (linking sources, posts, and timestamps) in `graph_builder.py`.

### Step 3: Model Implementation (`src/encoders/` & `src/fusion/`)
1. Integrate the pre-trained MuRIL/NepBERT models for text encoding.
2. Integrate CLIP/EfficientNet for extracting visual semantics.
3. Implement the ResNet-50 pipeline for processing ELA maps.
4. Implement GraphSage to process the constructed metadata graphs.
5. Build the **Modality Alignment** module to project text and image features into a shared space.
6. Build the **Divergence Score** module to compute conflicts between expected metadata and image forensics.

### Step 4: Training & Evaluation (`src/training/`)
1. Write the PyTorch `Dataset` and `DataLoader` classes to feed all 4 modalities (Text, Image, ELA, Graph).
2. Define the unified loss function (incorporating alignment loss, divergence loss, and classification loss).
3. Train the model using the `trainer.py` script.
4. Evaluate using accuracy, F1-score, and precision/recall metrics.

### Step 5: Deployment (`api/`, `app/`, `bot/`)
1. Export the trained model weights.
2. Update the FastAPI `api/` to accept text and image inputs, run preprocessing, and return predictions.
3. Integrate the API with the Streamlit `app/` and the Telegram `bot/`.

---
*Developed for robust, real-time Nepali fake news detection.*
