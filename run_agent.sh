#!/bin/bash

# # Process single document
# python document_classifier.py

# # Process batch of documents
# python batch_processor.py

# Custom parameters - MODIFY THE PATH BELOW TO TEST  SPECIFIC PDF
python -c "
from document_classifier import DocumentProcessor
processor = DocumentProcessor()

# Change this path to your specific PDF file
pdf_path = 'data/reservs/Allegheny DB 2470_179 - 4.23.2025.pdf'

result = processor.process_document(
    pdf_path,
    max_samples=5,  # Reduced for faster testing
    confidence_threshold=0.7
)
print(f'Document: {pdf_path}')
print(f'Classification: {result[\"classification\"]} ({\"Has Reservations\" if result[\"classification\"] == 1 else \"No Reservations\"})')
print(f'Confidence: {result[\"confidence\"]:.3f}')
print(f'Samples Used: {result[\"samples_used\"]}')
print(f'Early Stopped: {result[\"early_stopped\"]}')
"
