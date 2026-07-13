
# Bilingual Fake News Detection (Nepali & English) - Complete Kaggle Guide

This project trains a **Two-Model Ensemble** (XLM-RoBERTa-large + MuRIL-base) to detect fake news in both Nepali and English languages. It is specifically optimized to run **completely within a single free Kaggle Notebook session (under 12 hours)** using 4x T4 GPUs.

## Table of Contents
1. [Project Architecture & Methodology](#1-project-architecture--methodology)
2. [Step 1: Kaggle Account Setup & Verification](#2-step-1-kaggle-account-setup--verification)
3. [Step 2: Uploading Your Data to Kaggle](#3-step-2-uploading-your-data-to-kaggle)
4. [Step 3: Setting up the Kaggle Notebook](#4-step-3-setting-up-the-kaggle-notebook)
5. [Step 4: Running the Code](#5-step-4-running-the-code)
6. [Step 5: Downloading Your Trained Models](#6-step-5-downloading-your-trained-models)
7. [Troubleshooting & Common Errors](#7-troubleshooting--common-errors)

---

## 1. Project Architecture & Methodology

To achieve maximum accuracy within Kaggle's 12-hour time limit, we use a "Pruned Ensemble" approach instead of training three separate massive models.

### The Two Models:
1. **Model 1: XLM-RoBERTa-large (The Heavy Lifter)**
   - **Trained on:** The full 98,000 dataset (Nepali + English combined).
   - **Why:** XLM-R is a state-of-the-art cross-lingual model. By training it on both languages simultaneously, it learns shared "deception patterns" that work across language barriers. It achieves >99% accuracy in academic papers.
   - **Hardware Setup:** Batch size 8, Gradient Accumulation 2. Takes ~5-6 hours.

2. **Model 2: MuRIL-base (The Nepali Specialist)**
   - **Trained on:** The 55,000 Nepali-only subset.
   - **Why:** MuRIL (Multilingual Representations for Indian Languages) is specifically pre-trained on Devanagari script. It captures Nepali grammatical nuances and morphology that XLM-R might miss.
   - **Hardware Setup:** Batch size 16. Takes ~2 hours.

### The Ensemble Process:
After both models are trained, their predictions on the test set are combined using **Soft-Voting** (averaging their probability scores). This blocks individual model blind spots, resulting in a highly reliable final prediction.

---

## 2. Step 1: Kaggle Account Setup & Verification

If you are using Kaggle for the first time, you **must** verify your account to use GPUs.

1. Go to [Kaggle.com](https://www.kaggle.com/) and sign up for a free account.
2. Click on your profile picture (top right) -> **Settings**.
3. Scroll down to the **Phone Verification** section.
4. Enter your phone number and verify it via SMS.
   - *Why? Kaggle requires phone verification to prevent abuse of their free GPU resources. Without this, the GPU options in the next steps will be grayed out.*

---

## 3. Step 2: Uploading Your Data to Kaggle

You have two files: `data_nepali.zip` and `data_english.zip`. We need to upload these to Kaggle as a "Dataset" so the notebook can read them.

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2. Click the **+ New Dataset** button (top right).
3. A drag-and-drop box will appear. Drag both `data_nepali.zip` and `data_english.zip` into this box.
4. **Title:** Type `fnd-raw-news` (this exact name is important).
5. **Visibility:** Select **Private** (only you can see it).
6. Click **Create**.
   - Kaggle will now process the zips. Wait until the status says "Ready".

---

## 4. Step 3: Setting up the Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **+ New Notebook**.
2. In the notebook editor, look at the **right-hand sidebar** to configure the following settings:

### A. Input Data (Add your uploaded dataset)
- In the right sidebar, find the **Input** section.
- Click **+ Add Input**.
- Search for `fnd-raw-news` (the dataset you just created).
- Click the **+** button next to it. It should now appear under your inputs.

### B. Accelerator (The GPU)
- In the right sidebar, find the **Accelerator** section.
- Click the dropdown menu (it usually defaults to "None" or "CPU").
- Select **GPU T4 x4**.
  - *A popup will appear asking you to confirm. Click "Turn On".*
  - *This gives you 4 Tesla T4 GPUs, each with 16GB VRAM. This multi-GPU setup is what allows us to train XLM-R-large in under 6 hours.*

### C. Internet
- In the right sidebar, find the **Internet** section.
- Toggle it to **On**.
  - *Why? We need internet to download the base XLM-R and MuRIL models from HuggingFace during the first run.*

### D. Environment
- Leave as default (should be "Pin to original environment").

---

## 5. Step 4: Running the Code

1. Copy the entire Python code provided in `kaggle_notebook.py` above.
2. Paste it into the first cell of your Kaggle Notebook.
3. The code is divided into 8 sections (marked by `# CELL X`). You can either run it all at once or run cell-by-cell.

### Execution Steps:
1. Click the **Run** button (or `Shift + Enter`) to run Cell 1 (Installs). Wait for it to finish.
2. Run Cell 2 (Imports). Ensure it prints `GPUs Available: 4`.
3. Run Cell 3 (Data Loading). It will print the shapes of your data. Ensure it finds `Train: 68600 | Val: 14700 | Test: 14700` (numbers may vary slightly based on your exact data rows).
4. Run Cell 4 (Training Function setup).
5. **Run Cell 5 (Train XLM-R).**
   - *This will take approximately 5 to 6 hours.*
   - You can monitor the progress in the output log. Look for `loss` and `eval_f1` metrics.
6. **Run Cell 6 (Train MuRIL).**
   - *This will take approximately 1.5 to 2 hours.*
7. **Run Cell 7 (Evaluation).**
   - This will print the final `Accuracy` and `Macro F1` scores for the ensemble. It will also save a `test_predictions.csv` file.
8. **Run Cell 8 (SHAP).**
   - This generates explanation images and saves them as `.png` files.

**Total Estimated Time:** ~8 to 9 hours. This gives you a safe 3-hour buffer before Kaggle's 12-hour kill switch.

---

## 6. Step 5: Downloading Your Trained Models

Once the notebook finishes running, you need to save your trained models so you don't lose them when the session ends.

1. Look at the **right-hand sidebar** in your Kaggle notebook.
2. Find the **Output** section (underneath Data).
3. You will see a folder structure. Navigate to `/kaggle/working/final_models/`.
4. You will see two folders: `xlm-roberta-large` and `google_muril-base-cased`. These contain your trained weights.
5. You can download individual files by clicking the download icon next to them.
6. **Best Practice (Saving Version):**
   - Click the **Save Version** button at the top right of the screen.
   - Select **Save & Run All (Commit)**.
   - Ensure **Save Output** is checked.
   - Click **Save**.
   - This permanently saves a snapshot of your code, your trained model weights, and your SHAP images as a "Dataset" that you can access later even after the session expires.

---

## 7. Troubleshooting & Common Errors

### Error: `CUDA out of memory`
- **Cause:** The GPU ran out of VRAM. This usually happens if Kaggle assigns you a T4 x2 instead of T4 x4, or if the batch size is too large.
- **Fix:** Ensure your Accelerator is set to `GPU T4 x4`. If it still happens, reduce `batch_size` in Cell 5 from `8` to `4`, and increase `grad_accum` from `2` to `4`.

### Error: `ImportError: Using bitsandbytes 8-bit quantization requires Accelerate`
- **Cause:** Kaggle environment mismatch.
- **Fix:** After running Cell 1 (Installs), click the menu at the top: **Kernel -> Restart Kernel**. Do not run Cell 1 again. Proceed directly to Cell 2. The packages are still installed, but restarting resolves the mismatch.

### Error: `OSError: Couldn't connect to huggingface.co`
- **Cause:** Internet is turned off, or Kaggle blocked the connection.
- **Fix:** Ensure **Internet** is toggled "On" in the right sidebar before starting.

### Error: `Could not find text/label columns`
- **Cause:** The column names in your CSV/JSON files inside the zips do not match the hardcoded names in Cell 3.
- **Fix:** Look at the output log when Cell 3 runs. It will print `Loaded ... : (rows, cols)`. Note your actual column names. Go to Cell 3, find the line `TEXT_COLS = [...]` and `LABEL_COLS = [...]`, and add your specific column names to those lists.

### Session Timed Out at Hour 12
- **Cause:** Kaggle's hard limit. If your models were not saved before this, they are lost.
- **Prevention:** You must run the "Save Version" step (Step 6 above) immediately after Cell 5 and Cell 6 finish, not just at the very end. If Cell 5 finishes at hour 6, save a version immediately before starting Cell 6.
```