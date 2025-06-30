# 🤖 AI for Software Engineering - Week 3 Project

Welcome to my AI for Software Engineering project! This repository contains three main components that demonstrates my understanding and implementation of AI tools, covering theoretical knowledge, practical implementation skills, and ethical considerations with optimization techniques.

---

## 📚 Project Overview

This project is structured around three core pillars:

1. **🧠 Theory** - Theoretical understanding of AI Tools and concepts
2. **⚙️ Practical** - Implementation skills of AI tools and algorithms  
3. **⚖️ Ethics & Optimization** - Ethical AI development and software optimization skills

---

## 📁 Project Structure

```
AI-for-SE-WEEK-3/
├── 📄 Theoritical_understanding_of_AI.pdf    # Theoretical AI concepts
├── 🐍 Question_1_iris_decision_tree.ipynb    # Decision Tree Implementation
├── 📝 Nlpamazon.ipynb                        # NLP & Sentiment Analysis
├── 🧠 cnnmodel.ipynb                         # CNN for Image Classification
├── 📋 README.md                              # This documentation

```

---

## 🧠 Theory Component

### Theoretical Understanding of AI Tools
- **File**: `Theoritical_understanding_of_AI.pdf`
- **Content**: Comprehensive theoretical foundation covering:
  - AI fundamentals and concepts
  - Machine learning principles
  - Deep learning architectures
  - Natural Language Processing basics
  - Ethical considerations in AI development

---

## ⚙️ Practical Implementation Component

### 1. Decision Tree Classification - Iris Dataset
**File**: `Question_1_iris_decision_tree.ipynb`

**Implementation Details**:
- **Algorithm**: Decision Tree Classifier
- **Dataset**: Iris flower dataset (scikit-learn)
- **Features**: Sepal length/width, Petal length/width
- **Classes**: Setosa, Versicolor, Virginica
- **Performance**: 93.33% accuracy with weighted precision and recall

**Key Features**:
- Data preprocessing with missing value imputation
- Label encoding for categorical variables
- Stratified train-test split (80-20)
- Comprehensive model evaluation metrics
- Classification report with per-class performance

**Results**:
```
Accuracy: 0.9333
Precision (weighted): 0.9333
Recall (weighted): 0.9333
```

### 2. Natural Language Processing - Amazon Reviews Analysis
**File**: `Nlpamazon.ipynb`

**Implementation Details**:
- **Dataset**: Amazon Reviews (Kaggle)
- **Technologies**: spaCy, Kaggle API
- **Analysis**: Named Entity Recognition (NER) + Sentiment Analysis
- **Entities Detected**: Organizations, Products
- **Sentiment Classification**: Positive, Negative, Neutral

**Key Features**:
- Automated dataset download from Kaggle
- Rule-based sentiment analysis using keyword matching
- Named Entity Recognition for product and organization identification
- Ethical considerations documentation
- Sample review analysis with entity extraction

**Sample Output**:
```
--- Review 1 ---
Text: "Stuning even for the non-gamer: This sound track was beautiful!"
Entities: ['Chrono Cross']
Sentiment: Positive
```

### 3. Convolutional Neural Network - MNIST Digit Classification
**File**: `cnnmodel.ipynb`

**Implementation Details**:
- **Dataset**: MNIST handwritten digits
- **Architecture**: CNN with TensorFlow/Keras
- **Model Structure**:
  - Conv2D layers (32, 64 filters)
  - MaxPooling2D layers
  - Dense layers (128, 10 neurons)
  - Softmax activation for classification

**Key Features**:
- Image normalization (0-255 → 0-1)
- Data pipeline optimization with tf.data
- Batch processing (64 samples)
- Model training with validation
- Visual prediction results

**Performance**:
```
Test Accuracy: 0.9886 (98.86%)
Training completed in 5 epochs
```

---

## ⚖️ Ethics & Optimization Component

### Ethical Considerations

**Documented in NLP Analysis**:
- ⚠️ **Dataset Bias**: Overrepresentation of certain products/languages
- ⚠️ **Sentiment Ambiguity**: Sarcasm and nuanced language challenges
- ⚠️ **Rule-based Limitations**: Simple keyword matching may misclassify

**Mitigation Strategies**:
- Transparent rule-based systems for auditability
- TensorFlow Fairness Indicators for bias tracking
- Domain-adapted ML sentiment classifiers
- Continuous user feedback and misclassification auditing

### Optimization Techniques

**Data Pipeline Optimization**:
- TensorFlow data pipeline with `tf.data.AUTOTUNE`
- Caching and prefetching for improved performance
- Batch processing for memory efficiency
- Parallel processing for data loading

**Model Performance Optimization**:
- Appropriate train-test splits (80-20)
- Stratified sampling for balanced classes
- Hyperparameter tuning considerations
- Model evaluation with multiple metrics

---

## 🛠️ Technical Stack

### Core Technologies
- **Python**: Primary programming language
- **Jupyter Notebooks**: Interactive development environment
- **scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning framework
- **spaCy**: Natural Language Processing
- **pandas**: Data manipulation
- **matplotlib**: Data visualization

### Dependencies
```
numpy
pandas
scikit-learn
tensorflow
tensorflow-datasets
spacy
kaggle
matplotlib
```

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.7+ installed
2. Jupyter Notebook or Google Colab
3. Required Python packages (see installation below)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/AI-for-SE-WEEK-3.git
cd AI-for-SE-WEEK-3

# Install dependencies
pip install numpy pandas scikit-learn tensorflow tensorflow-datasets spacy matplotlib kaggle

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Notebooks
1. **Decision Tree**: Open `Question_1_iris_decision_tree.ipynb`
2. **NLP Analysis**: Open `Nlpamazon.ipynb` (requires Kaggle API setup)
3. **CNN Model**: Open `cnnmodel.ipynb`

---

## 📊 Project Outcomes

### Learning Objectives Achieved
✅ **Theoretical Understanding**: Comprehensive AI concepts covered  
✅ **Practical Implementation**: Three diverse AI applications implemented  
✅ **Ethical Awareness**: Bias detection and mitigation strategies documented  
✅ **Optimization Skills**: Performance optimization techniques applied  

### Technical Skills Demonstrated
- Machine Learning algorithm implementation
- Deep Learning with CNNs
- Natural Language Processing
- Data preprocessing and feature engineering
- Model evaluation and performance analysis
- Ethical AI development practices


---

## 📞 Support & Contact

For questions:
- victorbusami1@gmail.com


---

## 📄 License

This project is part of the AI for Software Engineering course curriculum.

---

*"The future of AI is not just about building intelligent systems, but about building them responsibly."* 🤖✨

