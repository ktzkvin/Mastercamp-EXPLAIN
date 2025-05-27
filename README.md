# PatentSense – Semantic Classification and Explanation of Patent Claims

## ⚙️ Technical Context

Patent classification is a cornerstone of intellectual property analysis, yet automating this task at scale remains a challenge. Legal and technical documents such as patent *claims* are verbose, intricate, and demand nuanced understanding.

**PatentSense** is an NLP-powered app for:
1. **Classifying patent claims into CPC (Cooperative Patent Classification) categories**
2. **Highlighting tokens that influenced the prediction**
3. **Summarizing claims using transformer-based models**

---

## 🧠 Method Overview

### 1. Input Handling

Users can interact via:
- A **text box**: paste a single claim
- A **CSV upload**: bulk classification, one claim per row

> Each claim is cleaned of HTML tags before processing.

---

### 2. Classification Pipeline

- Claims are tokenized and embedded using a **fine-tuned RoBERTa encoder**
- A **multi-label classifier** predicts one or more CPC top-level sections:

  | Code | Category |
  |------|----------|
  | A | Human Necessities |
  | B | Performing Operations, Transporting |
  | C | Chemistry, Metallurgy |
  | D | Textiles, Paper |
  | E | Fixed Constructions |
  | F | Mechanical Engineering |
  | G | Physics |
  | H | Electricity |
  | Y | General Tagging |

- The classifier returns binary predictions across 9 dimensions

---

### 3. Explanation & Visual Highlighting

- A **non-finetuned BERT** model extracts token-level embeddings
- Importance is computed as the **L2 norm** of each embedding
- Words are visualized in a heatmap style using `matplotlib`, where brighter color = higher influence

---

### 4. Abstractive Summarization

- Claims are first **shortened** via sentence selection based on token salience
- A **Pegasus transformer model** summarizes the condensed version
- Useful for patent analysts to get a quick overview of long, technical text

---

## 🧪 Core Dependencies

- `transformers`, `torch`, `nltk`, `streamlit`
- `scikit-learn`, `joblib`, `beautifulsoup4`
- `matplotlib`, `sentencepiece`, `shap`

To install, make sure your `requirements.txt` includes:

```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

The rest of the dependencies are standard pip packages, installable with:

```bash
pip install -r requirements.txt
```

---

## 💻 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-user/patentsense.git
cd patentsense
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
.venv\Scripts\Activate

# Unix-like
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install all project dependencies
pip install -r requirements.txt

# Download the pre-trained CPC model (~400MB)
pip install gdown
gdown https://drive.google.com/uc?id=1K5OKo7DGb2h6lR1C-iA4ftFgr9MqO3L2
```

### 4. Launch the app

```bash
streamlit run streamlit-main.py
```

---

## 📂 Model Files

Make sure `model.pkl` (the CPC classifier) is placed in the project root directory. If you're using a CPU-only machine, ensure it's saved in a format that doesn't require CUDA to load.

---

**PatentSense brings interpretability and classification to the heart of innovation, letting machines read patents like experts, only faster.**
