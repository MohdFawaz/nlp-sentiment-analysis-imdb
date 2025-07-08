# nlp-sentiment-analysis-imdb
# Sentiment Analysis on IMDB Movie Reviews Using ML and Deep Learning

This project performs sentiment classification on 50,000 IMDB movie reviews using a variety of machine learning and deep learning models. The goal is to classify each review as either positive or negative while comparing model performance, generalization, and overfitting behavior.

## 🎯 Objective

- Classify movie reviews into positive or negative categories.
- Explore five different machine learning techniques.
- Address overfitting through regularization and early stopping.

## 📦 Dataset

- 50,000 IMDB reviews (balanced: 25,000 positive, 25,000 negative)
- Pre-labeled for binary classification

## ⚙️ Data Preprocessing

### ✅ Cleaning Steps
- Removed HTML tags using regex
- Removed punctuation and numbers
- Normalized whitespaces
- Dropped single-character tokens

### 🔄 Tokenization & Embedding
- Tokenized using Keras Tokenizer (vocab size: 5,000)
- Converted to integer sequences and padded to length 100
- Used pre-trained **GloVe embeddings (100d)** for deep learning models
- Embedding matrix initialized as non-trainable

### 🧠 Feature Vectors
- **TF-IDF vectorization** used for SVM and Naive Bayes
- Labels encoded as 0 (negative) and 1 (positive)
- Dataset split into 80% training / 20% testing

---

## 🤖 Models Used

### 1. Traditional ML
- **Naive Bayes**: Multinomial NB on TF-IDF
- **Support Vector Machine (SVM)**: Linear SVM with TF-IDF
  - Tuned regularization parameter `C` to reduce overfitting

### 2. Deep Learning Models (All used GloVe Embeddings)
Each architecture tested in 3 configurations:
- Basic
- With Dropout + Early Stopping
- With L2 Regularization + Early Stopping

#### 🔹 Simple Neural Network (NN)
- Embedding → Flatten → Dense(sigmoid)

#### 🔹 Convolutional Neural Network (CNN)
- Embedding → Conv1D(128 filters, kernel=5) → GlobalMaxPool → Dense(sigmoid)

#### 🔹 Recurrent Neural Network (RNN) with LSTM
- Embedding → LSTM(128 units) → Dense(sigmoid)

---

## 🧪 Training Strategy

- **Batch size:** 128
- **Epochs:** up to 20 with Early Stopping
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- Early stopping monitors validation loss to prevent overfitting

---

## 🏆 Top 3 Models & Performances

### 1. 🧠 Adjusted SVM (C=0.1)
- **Accuracy:** 89.66%
- Reduced overfitting by increasing regularization
- Confusion Matrix:
  - True Negatives: 4,377
  - True Positives: 4,589
  - False Positives: 584
  - False Negatives: 450

### 2. 🔁 RNN with LSTM + Dropout
- **Accuracy:** 86.27%
- Generalized well, no sign of overfitting
- Confusion Matrix:
  - TN: 4,407, TP: 4,231
  - FP: 554, FN: 808

### 3. 🧮 Naive Bayes
- **Accuracy:** 85.24%
- Very stable and simple, with low variance
- Confusion Matrix:
  - TN: 4,233, TP: 4,291
  - FP: 728, FN: 748

---

## 📉 Overfitting Challenges & Solutions

| Problem | Solution |
|--------|----------|
| Basic NN underfitted | Switched to CNN and RNN to capture local/global patterns |
| SVM overfitted at high C | Reduced C value for stronger regularization |
| Deep models overfit on small epochs | Applied dropout or L2 + early stopping |

---

## 📊 Model Comparison Summary

| Model                | Accuracy | Overfitting Handling        |
|----------------------|----------|-----------------------------|
| Adjusted SVM (C=0.1) | 89.66%   | Stronger regularization     |
| RNN with LSTM        | 86.27%   | Dropout + Early Stopping    |
| Naive Bayes          | 85.24%   | Inherently well-generalized |
| CNN (L2 Reg)         | 83.2%    | L2 + Early Stopping         |
| Basic NN             | ~<80%    | Poor at capturing context   |

---

## 🧠 Key Takeaways

- Deep models benefit greatly from **pre-trained embeddings** and **regularization**
- Classical models like **Naive Bayes** remain strong baselines
- **Adjusted SVM** gave the best performance with good generalization
- Overfitting can be controlled with dropout, L2, and early stopping

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Moh’d Abu Quttain**  

