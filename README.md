<div align="center">

# 📄 DocClassify AI
### Intelligent Document Classification System

**An end-to-end NLP project that classifies text into one of six topics using TF-IDF and a tuned Linear Support Vector Machine.**

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render">
</p>

<p>
  <img src="https://img.shields.io/badge/Test%20Accuracy-91.07%25-2EA043?style=for-the-badge" alt="Test accuracy">
  <img src="https://img.shields.io/badge/Weighted%20F1-0.9105-2EA043?style=for-the-badge" alt="Weighted F1">
  <img src="https://img.shields.io/badge/SparkIIT-AI%20Internship%20Project-7C3AED?style=for-the-badge" alt="SparkIIT AI Internship Project">
</p>

**[🚀 Live Demo](https://docclassify-ai.onrender.com)** · **[📓 Training Notebook](./Kaushikee_Karmakar_Project1_Document_Classification.ipynb)** · **[📑 Project Report (PDF)](./Kaushikee_Karmakar_Project1_Report.pdf)**

</div>

---

## 🎯 Overview

DocClassify AI takes a text passage and predicts which of six topics it belongs to:

- Computer graphics
- Cars and automobiles
- Hockey
- Medicine
- Space
- Gun politics

The Streamlit app displays the predicted category, relative scores for the top predictions, and words that contributed to the selected class. It also includes a model-performance tab with evaluation metrics and a confusion matrix.

Built as **SparkIIT AI Training and Internship Program — Project 1** (powered by Wipro DICE ID), this project demonstrates a complete NLP workflow: dataset preparation, text cleaning, feature extraction, model comparison, hyperparameter tuning, evaluation, error analysis, and deployment.

> **Scope note:** This is a text classifier, not an OCR-based document scanner. Paste text into the app; scanned PDFs and images are not directly processed.

## ✨ Features

- 📝 Classifies pasted text into six predefined topics
- 📊 Shows scores for the top three predictions
- 🔎 Highlights influential words for the predicted category
- 🧪 Compares Naive Bayes, Logistic Regression, Linear SVM, tuned Linear SVM, and a MiniLM embedding baseline
- 📈 Displays test metrics, per-category results, and a confusion matrix
- ⚡ Includes example text for each category
- 🌐 Deployed as a Streamlit app on Render

## 🧠 Machine Learning Workflow

```mermaid
flowchart LR
    A["20 Newsgroups dataset"] --> B["Select six categories"]
    B --> C["Clean and tokenize text"]
    C --> D["TF-IDF features"]
    D --> E["Compare classification models"]
    E --> F["Tune Linear SVM"]
    F --> G["Evaluate on held-out test set"]
    G --> H["Save model with joblib"]
    H --> I["Streamlit app on Render"]
```

### Text-processing approach

1. Convert text to lowercase.
2. Tokenize alphabetic terms and remove English stop words.
3. Convert the cleaned text into TF-IDF features using unigrams and bigrams.
4. Use a Linear SVM to predict one of the six categories.

The TF-IDF vectorizer is fitted on the training data rather than the test data to avoid test-set leakage.

## 📦 Dataset

The project uses **scikit-learn's 20 Newsgroups dataset**, restricted to these categories:

| Newsgroup | Topic |
|---|---|
| `comp.graphics` | Computer graphics |
| `rec.autos` | Cars and automobiles |
| `rec.sport.hockey` | Hockey |
| `sci.med` | Medicine |
| `sci.space` | Space |
| `talk.politics.guns` | Gun politics |

The reported experiment uses a stratified 80/20 train-test split and **1,131 held-out test documents**. The corpus consists of historical Usenet posts, so its results should not be treated as a guarantee of performance on modern documents.

## 🏆 Results

### Model comparison

Reported results on the held-out test set:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.9045 | 0.9058 | 0.9045 | 0.9044 |
| Logistic Regression | 0.8974 | 0.8977 | 0.8974 | 0.8974 |
| Linear SVM | 0.9089 | 0.9090 | 0.9089 | 0.9088 |
| **Tuned Linear SVM (`C=0.5`)** | **0.9107** | **0.9108** | **0.9107** | **0.9105** |
| MiniLM embeddings + Logistic Regression | 0.8930 | — | 0.8930 | 0.8930 |

Precision, recall, and F1 are reported as support-weighted averages. Results are from the project's reported evaluation, not a new independent benchmark.

### Per-category performance

| Category | Precision | Recall | F1-score | Support |
|---|:---:|:---:|:---:|---:|
| `comp.graphics` | 0.905 | 0.953 | 0.928 | 190 |
| `rec.autos` | 0.904 | 0.909 | 0.907 | 187 |
| `rec.sport.hockey` | 0.948 | 0.943 | 0.946 | 194 |
| `sci.med` | 0.893 | 0.911 | 0.902 | 192 |
| `sci.space` | 0.903 | 0.880 | 0.891 | 191 |
| `talk.politics.guns` | 0.911 | 0.864 | 0.887 | 177 |

### What the results suggest

- **TF-IDF works well for this dataset.** The tuned Linear SVM achieved approximately 91.07% test accuracy.
- **The TF-IDF models performed similarly.** The small differences should not be overinterpreted from a single train-test split.
- **The embedding baseline did not outperform the tuned TF-IDF model** in this experiment. This is a result for this dataset and setup, not proof that TF-IDF is universally better than embeddings.
- **Gun-politics posts had the lowest recall** among the six categories in the reported per-class results.
- **Historical dataset bias matters.** Vocabulary patterns from 1990s Usenet discussions may not generalize to contemporary text.

## 🖥️ App and Prediction Scores

The app presents the predicted category and the top three class scores. These scores are calculated by applying a softmax transformation to the SVM decision scores.

**Important:** these values are relative scores, not calibrated probabilities. A high score does not guarantee that a prediction is correct, especially for vague text or text outside the six supported topics.

The app also displays influential words and a model-performance tab with aggregate metrics, a model comparison, per-category results, and a confusion matrix.

> The live demo is hosted on Render. If the service is idle, its first load may take longer while the instance wakes up.

## 🗂️ Repository Structure

```text
Document-Classification-AI/
├── app.py
├── doc_classifier.joblib
├── requirements.txt
├── Kaushikee_Karmakar_Project1_Document_Classification.ipynb
├── Kaushikee_Karmakar_Project1_Report.pdf
├── .streamlit/
│   └── config.toml
├── assets/
└── README.md
```

- `app.py` — Streamlit interface and prediction logic
- `doc_classifier.joblib` — Saved TF-IDF vectorizer and trained classifier bundle
- `requirements.txt` — Python package dependencies
- `Kaushikee_Karmakar_Project1_Document_Classification.ipynb` — Training and evaluation notebook
- `Kaushikee_Karmakar_Project1_Report.pdf` — Project report
- `.streamlit/config.toml` — Streamlit theme configuration
- `assets/` — Project assets, if used

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/KK-Dev0/Document-Classification-AI.git
cd Document-Classification-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies and launch the app

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser.

### Dependency compatibility

The saved model was serialized with `joblib`. Scikit-learn models can be version-sensitive, so use a scikit-learn version compatible with the one used to train the saved artifact. This repository currently pins `scikit-learn==1.6.1` in `requirements.txt`; avoid changing that version casually without testing the app.

## 🔁 Retrain or Explore the Model

Open [`Kaushikee_Karmakar_Project1_Document_Classification.ipynb`](./Kaushikee_Karmakar_Project1_Document_Classification.ipynb) in Google Colab or a local Jupyter environment and run the notebook cells in order. The dataset is available through scikit-learn.

Retraining may produce slightly different results if the data split, random seed, package versions, or preprocessing steps differ. If you replace the saved model, make sure the output structure remains compatible with `app.py`, which loads `doc_classifier.joblib`.

## 🚀 Deployment on Render

For a Render Web Service connected to this repository, use:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

Keep the Python runtime and dependency versions compatible with the saved model artifact. After changing dependencies or the model file, test the deployment before describing it as working.

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| NLP and feature extraction | TF-IDF, unigrams, bigrams, stop-word filtering |
| Machine learning | scikit-learn, Naive Bayes, Logistic Regression, Linear SVM |
| Embedding baseline | MiniLM sentence embeddings + Logistic Regression |
| Data handling | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Model serialization | joblib |
| Deployment and version control | Render, GitHub |
| Experiment environment | Google Colab / Jupyter |

## ⚠️ Limitations and Future Improvements

- **Limited topic coverage:** the model supports only six categories and always selects one of them, even if the input belongs elsewhere.
- **Historical data:** the model is trained on old Usenet discussions and may not transfer well to modern language.
- **Single split:** cross-validation and evaluation on a separate modern dataset would provide stronger evidence of generalization.
- **Uncalibrated scores:** the displayed softmax scores are relative rankings, not calibrated probabilities.
- **Bag-of-words limitations:** TF-IDF does not fully represent context, word order, or meaning.
- **Potential improvements:** add an unknown-topic rejection option, test on contemporary documents, evaluate probability calibration, and compare against stronger embedding or transformer-based models.

## 👩‍💻 Author

**Kaushikee Karmakar**  
BCA Student, Asansol Engineering College  
GitHub: [@KK-Dev0](https://github.com/KK-Dev0)

## 🙏 Acknowledgments

- **SparkIIT** and **Wipro DICE ID** for the AI training and internship program
- The **20 Newsgroups** dataset distributed through scikit-learn
- The open-source communities behind scikit-learn, Streamlit, sentence-transformers, and the Python data-science ecosystem

---

<div align="center">

⭐ If you find this project useful, consider starring the repository.

</div>
