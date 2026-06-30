# VeritasNP

VeritasNP is a bilingual (Nepali and English) fake news detection system. It aims to build the first benchmark dataset for Nepali fake news detection and provides a real-time web application and Telegram bot for classifying news headlines.

## Project Structure
- \data/\: Raw, processed, and external datasets.
- \scrapers/\: Scripts for gathering headlines from news portals and fact-checking sites.
- otebooks/\: Jupyter notebooks for data exploration and model training.
- \src/\: Core logic for preprocessing, inference, and models.
- \pi/\: FastAPI backend to serve the model.
- \pp/\: Streamlit frontend web application.
- \ot/\: Telegram bot for user interactions.

## Setup
1. Create a virtual environment: \python -m venv venv2. Activate the virtual environment.
3. Install dependencies: \pip install -r requirements.txt4. Copy \.env.example\ to \.env\ and fill in the required variables.
