#!/usr/bin/env python3
"""
Mineral Rights Classification Web App
====================================

Simple web interface for non-technical users to upload deed PDFs
and get mineral rights reservation analysis.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from document_classifier import DocumentProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize the document processor
processor = None

def init_processor():
    """Initialize the document processor"""
    global processor
    if processor is None:
        try:
            print("🔧 Initializing document processor...")
            processor = DocumentProcessor()
            print("✅ Document processor initialized successfully")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            raise
        except Exception as e:
            print(f"❌ Error initializing processor: {e}")
            # Try to provide more helpful error message
            if "proxies" in str(e):
                print("💡 This appears to be an Anthropic library version issue")
            raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def format_confidence_level(confidence):
    """Convert confidence score to user-friendly description"""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"

def generate_explanation(classification, confidence):
    """Generate user-friendly explanation of the results"""
    if classification == 1:  # Has reservations
        if confidence >= 0.7:
            return {
                "result": "Mineral Rights Reservations Found",
                "explanation": "Our AI analysis found strong evidence that this deed contains mineral rights reservations. This means the seller (grantor) has kept some or all mineral rights when transferring the property.",
                "recommendation": "We recommend having a qualified attorney review this document to understand the specific mineral rights that have been reserved."
            }
        else:
            return {
                "result": "Possible Mineral Rights Reservations",
                "explanation": "Our AI analysis suggests this deed may contain mineral rights reservations, but the confidence level is moderate. The language in the document is somewhat ambiguous.",
                "recommendation": "Due to the uncertainty, we strongly recommend having a qualified attorney review this document to clarify the mineral rights status."
            }
    else:  # No reservations
        if confidence >= 0.7:
            return {
                "result": "No Mineral Rights Reservations Found",
                "explanation": "Our AI analysis indicates this deed does not contain mineral rights reservations. The property transfer appears to include all rights, including mineral rights.",
                "recommendation": "Based on this analysis, the mineral rights appear to transfer with the property. However, for complete certainty, consider having an attorney review the document."
            }
        else:
            return {
                "result": "Unclear - No Clear Reservations Found",
                "explanation": "Our AI analysis did not find clear evidence of mineral rights reservations, but the confidence level is low. The document language may be complex or ambiguous.",
                "recommendation": "Due to the low confidence, we recommend having a qualified attorney review this document to confirm the mineral rights status."
            }

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Initialize processor if needed
        if processor is None:
            init_processor()
        
        # Process the document
        result = processor.process_document(
            filepath,
            max_samples=8,
            confidence_threshold=0.7,
            page_strategy="sequential_early_stop"
        )
        
        # Generate user-friendly explanation
        explanation = generate_explanation(result['classification'], result['confidence'])
        
        # Prepare response
        response_data = {
            'success': True,
            'filename': filename,
            'classification': result['classification'],
            'confidence': result['confidence'],
            'confidence_level': format_confidence_level(result['confidence']),
            'explanation': explanation,
            'processing_details': {
                'pages_processed': result['pages_processed'],
                'samples_used': result['samples_used'],
                'early_stopped': result['early_stopped'],
                'text_length': result['ocr_text_length']
            }
        }
        
        # Save results
        result_filename = f"result_{timestamp}_{filename.rsplit('.', 1)[0]}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        with open(result_path, 'w') as f:
            json.dump({
                'original_result': result,
                'user_friendly_response': response_data
            }, f, indent=2)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass  # Don't fail if cleanup fails
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing document: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        if processor is None:
            init_processor()
        return jsonify({'status': 'healthy', 'processor_ready': processor is not None})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port) 