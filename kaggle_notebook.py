# ============================================================
# CELL 1: INSTALL DEPENDENCIES (Run this first)
# ============================================================
!pip install -q transformers==4.44.2 accelerate==0.33.0 datasets evaluate bitsandbytes scikit-learn shap
# Note: If bitsandbytes fails to import later, restart the notebook kernel via the menu: Kernel -> Restart Kernel.

# ============================================================
# CELL 2: IMPORTS & CONFIGURATION
# ============================================================
import os
import re
import io
import zipfile
import unicodedata
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset
import evaluate

# Global Config
SEED = 42
MAX_LEN = 256  # Truncates news articles to first 256 tokens. Optimal for speed/accuracy balance.
RAW_DATA_PATH = "/kaggle/input/fnd-raw-news" # Name of the Kaggle dataset you will upload
OUTPUT_DIR = "/kaggle/working/"

# Helper to set seeds
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"GPUs Available: {torch.cuda.device_count()}")

# ============================================================
# CELL 3: DATA LOADING & PREPROCESSING
# ============================================================
def read_zip(zip_path):
    """Reads csv/json from a zip file into a pandas dataframe."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('/'): continue
            lower = name.lower()
            with z.open(name) as fh:
                raw = fh.read()
            try:
                if lower.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
                elif lower.endswith('.tsv'):
                    df = pd.read_csv(io.BytesIO(raw), sep='\t', encoding='utf-8-sig')
                elif lower.endswith('.json'):
                    df = pd.read_json(io.BytesIO(raw))
                elif lower.endswith('.jsonl'):
                    df = pd.read_json(io.BytesIO(raw), lines=True)
                else:
                    continue
                print(f"  Loaded {name}: {df.shape}")
                frames.append(df)
            except Exception as e:
                print(f"  Failed to read {name}: {e}")
    if not frames: raise ValueError("No valid data files found in zip.")
    return pd.concat(frames, ignore_index=True)

# --- ADJUST COLUMN NAMES HERE IF YOURS ARE DIFFERENT ---
TEXT_COLS = ["text", "content", "article", "body", "news", "title_text", "title"]
LABEL_COLS = ["label", "is_fake", "fake", "class", "target"]

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def normalize_label(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v in {"1", "true", "fake", "false_news", "fake_news"}: return 1
        if v in {"0", "false", "real", "true_news", "genuine"}: return 0
    try: return int(v)
    except: return None

def normalize_text(t):
    if not isinstance(t, str): return ""
    t = unicodedata.normalize("NFC", t) # Crucial for Devanagari script
    t = t.replace("\ufeff", "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

print("Loading Nepali Data...")
df_ne = read_zip(os.path.join(RAW_DATA_PATH, "data_nepali.zip"))
print("Loading English Data...")
df_en = read_zip(os.path.join(RAW_DATA_PATH, "data_english.zip"))

# Process data
df_ne['lang'] = 'ne'
df_en['lang'] = 'en'
df = pd.concat([df_ne, df_en], ignore_index=True)

tcol = pick_column(df, TEXT_COLS)
lcol = pick_column(df, LABEL_COLS)
if not tcol or not lcol:
    raise ValueError(f"Could not find text/label columns. Found: {df.columns}")

df['text'] = df[tcol].apply(normalize_text)
df['label'] = df[lcol].apply(normalize_label)
df = df[df['label'].notna() & (df['text'].str.len() > 20)].copy()
df['label'] = df['label'].astype(int)

# Add language token prefix for the XLM-R model
df['input'] = df['lang'].map({"ne":"<ne> ", "en":"<en> "}) + df['text']

# Stratified Split (70/15/15) based on language and label
df["strat"] = df["lang"] + "_" + df["label"].astype(str)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, val_test_idx = next(sss1.split(df, df["strat"]))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(sss2.split(df.iloc[val_test_idx], df.iloc[val_test_idx]["strat"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# CELL 4: TRAINING FUNCTION
# ============================================================
def train_model(model_name, train_data, val_data, save_path, batch_size=8, grad_accum=2, epochs=3):
    print(f"\n=== Training {model_name} ===")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0:"real", 1:"fake"}, label2id={"real":0, "fake":1}
    )
    model.config.use_cache = False
    
    # Tokenize datasets
    def tok_func(batch):
        return tokenizer(batch["input"], truncation=True, max_length=MAX_LEN)
    
    cols = ["input", "label"]
    train_ds = Dataset.from_pandas(train_data[cols]).map(tok_func, batched=True, remove_columns=["input"])
    val_ds = Dataset.from_pandas(val_data[cols]).map(tok_func, batched=True, remove_columns=["input"])
    
    # Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {**acc_metric.compute(predictions=preds, references=p.label_ids),
                **f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")}
    
    # Training Arguments optimized for Kaggle T4 x4
    args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit", # Saves VRAM
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1, # Save disk space
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        ddp_find_unused_parameters=False,
        seed=SEED
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final_models", model_name.replace("/", "_"))
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")
    
    # Clear VRAM
    del model, trainer
    torch.cuda.empty_cache()
    return final_dir

# ============================================================
# CELL 5: TRAIN MODEL 1 - XLM-RoBERTa-large (Bilingual)
# ============================================================
# Trains on the full 98,000 dataset (Nepali + English)
xlmr_save_path = "/kaggle/working/xlmr_checkpoints"
xlmr_final_dir = train_model(
    model_name="xlm-roberta-large", 
    train_data=train_df, 
    val_data=val_df, 
    save_path=xlmr_save_path,
    batch_size=8, 
    grad_accum=2, 
    epochs=3
)

# ============================================================
# CELL 6: TRAIN MODEL 2 - MuRIL-base (Nepali Specialist)
# ============================================================
# Trains only on the Nepali subset
train_ne = train_df[train_df.lang == 'ne'].reset_index(drop=True)
val_ne = val_df[val_df.lang == 'ne'].reset_index(drop=True)

muril_save_path = "/kaggle/working/muril_checkpoints"
muril_final_dir = train_model(
    model_name="google/muril-base-cased", 
    train_data=train_ne, 
    val_data=val_ne, 
    save_path=muril_save_path,
    batch_size=16, # Base model is smaller, we can fit larger batch
    grad_accum=1, 
    epochs=3
)

# ============================================================
# CELL 7: ENSEMBLE & EVALUATION ON TEST SET
# ============================================================
from scipy.special import softmax

def get_predictions(model_path, data):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda").eval()
    
    def tok_func(batch):
        return tokenizer(batch, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    
    probs = []
    batch_size = 32
    for i in range(0, len(data), batch_size):
        batch_texts = data["input"].iloc[i:i+batch_size].tolist()
        inputs = tok_func(batch_texts)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs.append(softmax(logits.cpu().numpy(), axis=-1)[:, 1]) # Prob of Fake
    return np.concatenate(probs)

print("Generating XLM-R predictions on test set...")
xlmr_probs = get_predictions(xlmr_final_dir, test_df)

print("Generating MuRIL predictions on test set (using full test set for combined evaluation)...")
muril_probs = get_predictions(muril_final_dir, test_df)

# Ensemble: Average probabilities
ensemble_probs = (xlmr_probs + muril_probs) / 2.0
ensemble_preds = (ensemble_probs >= 0.5).astype(int)
y_true = test_df["label"].values

print("\n" + "="*50)
print("FINAL ENSEMBLE TEST METRICS")
print("="*50)
print(classification_report(y_true, ensemble_preds, target_names=["Real", "Fake"]))
print(f"Accuracy: {accuracy_score(y_true, ensemble_preds):.4f}")
print(f"Macro F1: {f1_score(y_true, ensemble_preds, average='macro'):.4f}")

# Per-language evaluation
test_df_reset = test_df.reset_index(drop=True)
for lang in ["ne", "en"]:
    mask = test_df_reset["lang"] == lang
    print(f"\nResults for {lang.upper()}:")
    print(f"  Accuracy: {accuracy_score(y_true[mask], ensemble_preds[mask]):.4f}")
    print(f"  F1 Score: {f1_score(y_true[mask], ensemble_preds[mask], average='macro'):.4f}")

# Save predictions and test data for SHAP
test_df_reset["xlmr_prob"] = xlmr_probs
test_df_reset["muril_prob"] = muril_probs
test_df_reset["ensemble_prob"] = ensemble_probs
test_df_reset["ensemble_pred"] = ensemble_preds
test_df_reset.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# ============================================================
# CELL 8: SHAP EXPLAINABILITY
# ============================================================
import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP explanations...")
# Use XLM-R for global explainability as it handles both languages
tokenizer = AutoTokenizer.from_pretrained(xlmr_final_dir)
model = AutoModelForSequenceClassification.from_pretrained(xlmr_final_dir).to("cuda").eval()

class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def __call__(self, texts):
        if isinstance(texts, str): texts = [texts]
        inputs = self.tokenizer(list(texts), return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy() # Return P(Fake)

pred = Predictor(model, tokenizer)
explainer = shap.Explainer(pred, masker=shap.maskers.Text(tokenizer), output_names=["Real", "Fake"])

# Explain 2 Nepali and 2 English samples
sample_ne = test_df_reset[test_df_reset.lang == "ne"].sample(2, random_state=42)["text"].tolist()
sample_en = test_df_reset[test_df_reset.lang == "en"].sample(2, random_state=42)["text"].tolist()
samples = sample_ne + sample_en

shap_values = explainer(samples)

# Save plots
for i in range(len(samples)):
    lang = "Nepali" if i < 2 else "English"
    plt.figure()
    shap.plots.text(shap_values[i], display=False, show=False)
    plt.title(f"SHAP Explanation - {lang} Sample {i%2+1}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_explanation_{lang}_{i%2+1}.png"), bbox_inches="tight", dpi=150)
    plt.close()

print("\nTraining, Evaluation, and Explainability complete. Download files from /kaggle/working/ output section.")