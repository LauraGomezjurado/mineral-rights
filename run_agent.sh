#!/bin/bash

# Process single document
python document_classifier.py

# Process batch of documents
python batch_processor.py

# Custom parameters
python -c "
from document_classifier import DocumentProcessor
processor = DocumentProcessor()
result = processor.process_document(
    'data/reservs/Somerset DB 1820_349.pdf',
    max_samples=15,
    confidence_threshold=0.8
)
print(f'Classification: {result[\"classification\"]}')
print(f'Confidence: {result[\"confidence\"]:.3f}')
"
