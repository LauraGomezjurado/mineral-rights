# 🏛️ Mineral Rights Document Classification System

An advanced AI-powered system for automatically classifying legal deed documents to determine whether they contain mineral rights reservations. The system combines state-of-the-art OCR, natural language processing, and self-consistent sampling techniques to achieve high accuracy on complex legal documents.

## 🎯 Project Overview

**Primary Objective**: Automatically classify land deed PDFs into **"Has Mineral Rights Reservations"** and **"No Mineral Rights Reservations"** categories with high accuracy and confidence scoring.

**Key Innovation**: Uses chunk-by-chunk early stopping analysis and self-consistent sampling to maximize accuracy while minimizing processing time and costs.

## ✨ Key Features

- 🔍 **Smart OCR**: Claude-powered text extraction with high accuracy on legal documents
- 🧠 **AI Classification**: Advanced prompt engineering with self-consistent sampling
- ⚡ **Early Stopping**: Intelligent page-by-page analysis that stops when reservations are found
- 📊 **Confidence Scoring**: Machine learning-based confidence assessment
- 📈 **Comprehensive Evaluation**: Detailed accuracy metrics and performance analysis
- 🎮 **Interactive Demo**: Easy-to-use demonstration system

## 🏗️ System Architecture

### Core Components

1. **Document Processor** (`document_classifier.py`)
   - PDF to image conversion
   - Claude-powered OCR text extraction
   - Multi-page processing with smart strategies

2. **AI Classification Engine**
   - Self-consistent sampling with temperature variation
   - Confidence scoring using logistic regression
   - Early stopping based on confidence thresholds

3. **Batch Processing** (`batch_processor.py`)
   - Large-scale document processing
   - Comprehensive evaluation metrics
   - Detailed reporting and analysis

4. **Evaluation System**
   - Accuracy assessment with confusion matrices
   - Performance metrics (precision, recall, F1-score)
   - Misclassification analysis

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Environment Setup

1. **Set up Anthropic API key** (required for Claude OCR and classification):
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   Or edit the API key directly in `document_classifier.py`

2. **Verify installation**:
   ```bash
   python -c "import anthropic, fitz, PIL; print('✅ All dependencies installed')"
   ```

### 🎮 Interactive Demo

The easiest way to see the system in action:

```bash
# Run interactive demo
./run_demo.sh

# Or run directly with Python
python demo.py "data/reservs/Indiana Co. PA DB 550_322.pdf"
```

**Demo Options:**
- Document WITH mineral rights reservations
- Document WITHOUT mineral rights reservations  
- Custom document path
- Side-by-side comparison

### 📋 Single Document Processing

```python
from document_classifier import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
result = processor.process_document("path/to/deed.pdf")

# View results
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Pages processed: {result['pages_processed']}")
```

### 📊 Batch Processing & Evaluation

```bash
# Process entire dataset and generate evaluation report
python batch_processor.py

# Custom batch processing
python -c "
from batch_processor import process_batch
results = process_batch(['data/reservs', 'data/no-reservs'])
"
```

## 📁 Project Structure

```
mineral-rights/
├── 📄 demo.py                    # Interactive demonstration script
├── 🔧 run_demo.sh               # Demo runner with menu options
├── 🤖 document_classifier.py    # Core classification engine
├── 📊 batch_processor.py        # Batch processing and evaluation
├── 🧪 test_accuracy.py          # Accuracy testing utilities
├── 📈 evaluate_full_dataset.py  # Comprehensive evaluation
├── 🔍 test_false_positive.py    # False positive analysis
├── ⚙️  config.py                # Configuration settings
├── 📋 requirements.txt          # Python dependencies
├── 📚 README.md                 # This file
├── 📂 data/                     # Document datasets
│   ├── reservs/                 # Documents WITH reservations
│   ├── no-reservs/             # Documents WITHOUT reservations
│   └── samples/                # Sample documents
├── 📂 src/                      # Source utilities
│   ├── ocr_evaluation.py       # OCR testing and comparison
│   ├── simulate_pipeline.py    # Pipeline simulation
│   └── utils.py                # Utility functions
├── 📂 outputs/                  # Processing results
├── 📂 batch_results/           # Batch processing outputs
├── 📂 demo_results/            # Demo outputs
└── 📂 experiments/             # Research and testing
```

## 🔬 Technical Details

### Classification Process

1. **Document Input**: PDF file of legal deed
2. **Page Strategy**: Sequential processing with early stopping
3. **OCR Extraction**: Claude-powered text extraction per page
4. **AI Analysis**: Self-consistent sampling with multiple attempts
5. **Confidence Scoring**: ML-based confidence assessment
6. **Early Stopping**: Stop when reservations found (Class 1)
7. **Final Decision**: Weighted voting with confidence scores

### Key Legal Phrases Detected

The system identifies various forms of mineral rights language:

**Positive Indicators (Has Reservations):**
- "reserves", "excepts", "retains" mineral rights
- "coal", "oil", "gas", "minerals", "mining rights"
- "subject to mineral rights reserved in prior deed"
- "1/2 of mineral rights", "except 1/8 royalty interest"
- "Grantor reserves all mineral rights"

**Negative Indicators (Boilerplate/Disclaimers):**
- Legal disclaimers and warranty text
- Title insurance notices
- Standard recording acknowledgments
- "Rights otherwise created, transferred, excepted or reserved BY THIS INSTRUMENT"

### Processing Strategies

- **`sequential_early_stop`** (Default): Process pages sequentially, stop when reservations found
- **`first_few`**: Process first 3 pages only
- **`first_and_last`**: Process first 2 and last pages
- **`all`**: Process all pages (legacy mode)

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: Of predicted reservations, how many were correct
- **Recall**: Of actual reservations, how many were found
- **Specificity**: Of actual no-reservations, how many were correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence Statistics**: Distribution of confidence scores
- **Processing Efficiency**: Pages processed, early stopping rate

## 🧪 Evaluation & Testing

### Run Full Evaluation

```bash
# Comprehensive dataset evaluation
python evaluate_full_dataset.py

# Test specific accuracy scenarios
python test_accuracy.py

# Analyze false positives
python test_false_positive.py
```

### Custom Evaluation

```python
from batch_processor import process_batch, generate_evaluation_report

# Process custom directories
results = process_batch([
    "path/to/positive/samples",
    "path/to/negative/samples"
], output_dir="custom_results")

# Generate detailed report
generate_evaluation_report(results, Path("custom_results"))
```

## ⚙️ Configuration

### Key Parameters

```python
# In document_classifier.py
max_samples = 10              # Max AI samples per classification
confidence_threshold = 0.7    # Early stopping threshold
max_tokens_per_page = 8000   # OCR token limit per page
page_strategy = "sequential_early_stop"  # Processing strategy
```

### API Configuration

```python
# Anthropic Claude API
model = "claude-3-5-sonnet-20241022"
max_tokens = 1000            # Classification response limit
temperature = 0.7            # Response randomness
```

## 📈 Usage Examples

### Basic Classification

```python
from document_classifier import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document("deed.pdf")

if result['classification'] == 1:
    print("⚠️ Has mineral rights reservations")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Found on page: {result.get('stopped_at_chunk')}")
else:
    print("✅ No mineral rights reservations")
    print(f"Pages analyzed: {result['pages_processed']}")
```

### Batch Analysis

```python
from batch_processor import process_batch

# Process multiple directories
results = process_batch([
    "data/reservs",
    "data/no-reservs"
], output_dir="analysis_results")

# Print summary
correct = sum(1 for r in results if r['correct_prediction'])
total = len(results)
print(f"Accuracy: {correct/total:.3f} ({correct}/{total})")
```

### Custom Processing

```python
# Custom processing parameters
result = processor.process_document(
    "document.pdf",
    max_samples=15,                    # More thorough analysis
    confidence_threshold=0.8,          # Higher confidence requirement
    page_strategy="first_few",         # Only first few pages
    max_tokens_per_page=10000         # Higher OCR token limit
)
```

## 🛠️ Technology Stack

- **🤖 AI/ML**: Anthropic Claude, scikit-learn, NumPy
- **📄 Document Processing**: PyMuPDF, Pillow, pdf2image
- **📊 Data Analysis**: pandas, NumPy
- **🐍 Core**: Python 3.8+, pathlib, json
- **🔧 Utilities**: tqdm, datetime, re

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```bash
   # Set environment variable
   export ANTHROPIC_API_KEY="your-key"
   # Or edit document_classifier.py line 18
   ```

2. **PDF Processing Error**:
   ```bash
   # Install system dependencies
   brew install poppler  # macOS
   sudo apt-get install poppler-utils  # Ubuntu
   ```

3. **Memory Issues with Large PDFs**:
   ```python
   # Reduce token limit
   result = processor.process_document(
       "large_doc.pdf", 
       max_tokens_per_page=4000
   )
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Process with verbose output
result = processor.process_document("doc.pdf")
```

## 📋 Output Files

The system generates several types of output files:

### Demo Results (`demo_results/`)
- `demo_result_*.json`: Complete analysis data
- `demo_text_*.txt`: Extracted OCR text
- `demo_summary_*.txt`: Human-readable report

### Batch Results (`batch_results/`)
- `evaluation_report.txt`: Comprehensive accuracy analysis
- `detailed_results.csv`: Per-document results
- `summary_stats.json`: Metrics summary
- `*_result.json`: Individual document results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- PyMuPDF for PDF processing
- scikit-learn for machine learning utilities

---

**Built for accurate, scalable mineral rights classification** ⚖️

*For questions or support, please open an issue on GitHub.* 