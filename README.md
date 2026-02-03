# Quramet:Multi-Dimensional Classification of Qur’anic Metaphors

This repository contains the official implementation for the paper: **"Decoding the Rhetorical Layers of Isti'aara: A Multi-Dimensional Computational Framework for Qur'anic Metaphor Analysis"**.

## 📌 Overview

This project introduces a **Structured Rhetorical Parsing** framework for Qur'anic metaphors (*Isti'aara*). Unlike traditional binary detection, we classify metaphors across three simultaneous dimensions using State-of-the-Art Arabic Transformer models.

### The 3 Rhetorical Dimensions:
1.  **Type (Form):** Explicit (*Tasrihiyya*) vs. Implicit (*Makniyya*).
2.  **Origin (Root):** Primary (*Asliyya*) vs. Derivative (*Tabiyya*).
3.  **Context (Coherence):** Candidate (*Murashaha*), Abstract (*Mujarrada*), or Absolute (*Mutlaqa*).

## 📂 Dataset

The dataset represents an exhaustive census of metaphors derived from authoritative classical exegeses (*Al-Kashshaf*, *Ibn Ashur*).
- **Size:** 1,181 unique verses.
- **Format:** Multi-label annotated CSV.

## 🚀 Models

We evaluate three Arabic-specific Transformer architectures:
1.  **MARBERT** (Best Performing)
2.  **CamelBERT-CA** (Classical Arabic Optimized)
3.  **AraBERT v2**

## 🛠️ Installation

```bash
git clone https://github.com/YourUsername/Quranic-Metaphor-Analysis.git
cd Quranic-Metaphor-Analysis
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License.
```
