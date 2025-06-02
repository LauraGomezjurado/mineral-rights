# Mineral Rights Agent 

A multi-step AI agent that classifies land deed PDFs into **with reservations** and **without reservations** categories using state-of-the-art OCR and LLM techniques.

## Project Goals

**Primary Objective**: Maximize classification accuracy by breaking down the agent into modular, testable components.

**Key Challenge**: Handle diverse document types including handwritten, machine-typed, and digital PDFs with varying quality and formats.

## Pipeline Architecture 

### Pipeline Components:

1. **OCR Layer**: Extract clean text using multiple OCR engines (TrOCR, Donut, Tesseract)
2. **Text Normalization**: Clean and standardize extracted text
3. **Phrase Detection**: Identify key legal phrases indicating mineral rights reservations
4. **Classification**: Determine reservation status using LLM reasoning
5. **Self-Verification**: Validate classification with confidence scoring

## Key Legal Phrases

The agent looks for phrases indicating mineral rights reservations:

- "reserving unto the grantor"
- "except and reserve"
- "subject to mineral rights"
- "coal rights reserved"
- "oil and gas rights reserved"
- "mineral interests reserved"

## Quick Start

### 1. Run Pipeline Simulation
```bash
cd mineral-rights-agent
python src/simulate_pipeline.py --input data/samples/reserv_deed_1.pdf
```

### 2. Explore OCR Evaluation
```bash
jupyter notebook notebooks/ocr_evaluation.ipynb
```

### 3. Test Prompt Engineering
```bash
jupyter notebook notebooks/prompt_experiments.ipynb
```

##  Project Structure 

## 🔬 Evaluation Strategy

Each pipeline component is evaluated independently:

- **OCR Accuracy**: Text extraction quality across different document types
- **Phrase Detection**: Precision/recall for key legal phrases
- **Classification**: Accuracy on labeled deed samples
- **End-to-End**: Overall pipeline performance

## 🛠️ Technology Stack

- **OCR**: TrOCR, Donut, PaddleOCR, Tesseract
- **LLM**: OpenAI GPT-4, Claude, or local models
- **Processing**: Python, PyTorch, Transformers
- **Evaluation**: Jupyter notebooks, pandas

## 📈 Next Steps

1. **Phase 1**: OCR evaluation and optimization
2. **Phase 2**: Prompt engineering and classification
3. **Phase 3**: End-to-end pipeline integration
4. **Phase 4**: Production deployment and monitoring

---

*Built for accurate, scalable mineral rights classification* ⚖️ 