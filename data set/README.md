# Nepali Fake News Detection Dataset 🇳🇵

## Overview
The **Nepali Fake News Detection Dataset** is a curated, research-oriented dataset developed to support
**fake news detection**, **misinformation analysis**, and **Natural Language Processing (NLP)** research
for the Nepali language.

The dataset contains Nepali-language news content collected from social media platforms and official
online news portals, manually reviewed and labeled to distinguish between **real** and **fake** news.
It is designed for use in academic research, machine learning experiments, and benchmark evaluations.

---

## Motivation
With the rapid growth of digital media in Nepal, misinformation and fake news have become critical
societal challenges. However, high-quality, labeled datasets for the Nepali language remain limited.
This dataset aims to bridge that gap by providing structured, annotated Nepali news data for researchers,
students, and practitioners.

---

## Dataset Structure
- **Format:** CSV (UTF-8 encoded)
- **Language:** Nepali
- **Label Type:** Binary classification
- **Domain:** News and social media content

Multiple CSV files may be provided, each following the same schema.

---

## Column Description

| Column Name     | Description |
|-----------------|-------------|
| `news_context`  | Full Nepali news text or post content |
| `label`         | Binary label (`0` = Real news, `1` = Fake news) |
| `category`      | News category (politics, economy, health, society, technology, agriculture, tourism, education, crime, disaster) |
| `source_type`   | Source of the news (social_media, official_news, official_portal, others) |
| `news_id`       | Unique identifier for each news instance |
| `generated_at`  | Timestamp indicating when the data entry was created |
| `meta_intent`   | Intent of the content (informative, misleading) |
| `meta_style`    | Writing style (neutral, sensational, opinionated) |

---

## Label Definition

### `label`
- **0 — Real News:** Verified or factual information from reliable sources
- **1 — Fake News:** False, misleading, or manipulated information

---

## Data Sources
- Social media platforms (e.g., Facebook, TikTok)
- Official Nepali news portals
- Online media outlets and public pages

All data has been collected from publicly available sources.

---

## Intended Use
This dataset can be used for:
- Fake news detection and classification
- Nepali language NLP research
- Text classification and feature engineering
- Misinformation and disinformation analysis
- Social media analytics
- Benchmarking machine learning models

---

## Example Usage (Python)

```python
import pandas as pd

df = pd.read_csv("all_data.csv")
print(df.head())
