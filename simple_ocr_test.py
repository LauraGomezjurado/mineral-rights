#!/usr/bin/env python3
"""
Simple OCR Comparison for Mineral Rights Deeds
==============================================

Just run all premium models on your deeds and compare results.
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# Set API keys
os.environ['OPENAI_API_KEY'] = "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"
os.environ['ANTHROPIC_API_KEY'] = "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"

# Import OCR engines
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, DonutProcessor
    print("✅ Transformers available")
except ImportError:
    print("❌ Transformers not available")
    sys.exit(1)

try:
    import openai
    print("✅ OpenAI available")
except ImportError:
    print("❌ OpenAI not available")

try:
    import anthropic
    print("✅ Anthropic available")
except ImportError:
    print("❌ Anthropic not available")

def pdf_to_image(pdf_path):
    """Convert first page of PDF to image"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    mat = fitz.Matrix(2, 2)  # 2x zoom
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    doc.close()
    
    from io import BytesIO
    return Image.open(BytesIO(img_data))

def run_trocr(image, model_type="printed"):
    """Run TrOCR"""
    print(f"🚀 Running TrOCR-{model_type}...")
    
    processor = TrOCRProcessor.from_pretrained(f"microsoft/trocr-large-{model_type}")
    model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/trocr-large-{model_type}")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text

def run_donut(image):
    """Run Donut"""
    print("🍩 Running Donut...")
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_document>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    generated_ids = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Clean up Donut output
    import re
    text = re.sub(r'<[^>]*>', '', text)
    return text

def run_gpt4o(image):
    """Run GPT-4o"""
    print("🧠 Running GPT-4o...")
    
    import base64
    from io import BytesIO
    
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    client = openai.OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Extract ALL text from this legal deed document. Pay special attention to any mineral rights reservations. Format as clean markdown."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def run_claude(image):
    """Run Claude 3.5 Sonnet"""
    print("🔮 Running Claude...")
    
    import base64
    from io import BytesIO
    
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract ALL text from this legal deed document. Pay special attention to any mineral rights reservations. Format as clean markdown."
                    }
                ]
            }
        ]
    )
    
    return response.content[0].text

def process_document(pdf_path):
    """Process one document with Claude and GPT-4o only"""
    print(f"\nProcessing: {pdf_path.name}")
    
    # Convert to image
    image = pdf_to_image(pdf_path)
    
    results = {}
    
    # Run all models
    # try:
    #     results['trocr_printed'] = run_trocr(image, "printed")
    # except Exception as e:
    #     results['trocr_printed'] = f"Error: {e}"
    
    # try:
    #     results['trocr_handwritten'] = run_trocr(image, "handwritten")
    # except Exception as e:
    #     results['trocr_handwritten'] = f"Error: {e}"
    
    # try:
    #     results['donut'] = run_donut(image)
    # except Exception as e:
    #     results['donut'] = f"Error: {e}"
    
    try:
        results['gpt4o'] = run_gpt4o(image)
    except Exception as e:
        results['gpt4o'] = f"Error: {e}"
    
    try:
        results['claude'] = run_claude(image)
    except Exception as e:
        results['claude'] = f"Error: {e}"
    
    return results

def main():
    print(" OCR Comparison for Mineral Rights Deeds")
    print("=" * 60)
    
    # Process ALL PDF files from both directories
    pdf_files = []
    data_dirs = [
        '/Users/lauragomez/Desktop/mineral-rights/data/no-reservs',
        '/Users/lauragomez/Desktop/mineral-rights/data/reservs'
    ]
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))  # Remove [:3] to get ALL files
            pdf_files.extend(pdfs)
            print(f"📁 Found {len(pdfs)} PDFs in {data_path.name}")
        else:
            print(f"❌ Directory not found: {data_dir}")
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    print(f"\n🚀 Processing {len(pdf_files)} documents with Claude and GPT-4o...")
    
    all_results = {}
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            results = process_document(pdf_path)
            all_results[pdf_path.name] = results
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name}: {e}")
    
    # Save results
    output_dir = Path("ocr_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "claude_vs_gpt4o_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n Completed! Results saved to {output_dir}/claude_vs_gpt4o_results.json")
    print("\n Claude vs GPT-4o Summary:")
    
    claude_wins = 0
    gpt4o_wins = 0
    ties = 0
    
    for doc_name, results in all_results.items():
        print(f"\n📄 {doc_name}:")
        
        # Determine which directory this file came from
        file_type = " RESERVATIONS" if any(pdf for pdf in pdf_files if pdf.name == doc_name and "reservs" in str(pdf.parent)) else "❌ NO RESERVATIONS"
        print(f"    Expected: {file_type}")
        
        claude_words = 0
        gpt4o_words = 0
        claude_found_reservations = False
        gpt4o_found_reservations = False
        
        for model, text in results.items():
            if "Error:" in str(text):
                print(f"   ❌ {model}: {text}")
            else:
                word_count = len(str(text).split())
                has_reservation = any(word in str(text).lower() for word in ['reserving', 'except', 'subject to', 'mineral rights', 'oil and gas'])
                
                if model == 'claude':
                    claude_words = word_count
                    claude_found_reservations = has_reservation
                elif model == 'gpt4o':
                    gpt4o_words = word_count
                    gpt4o_found_reservations = has_reservation
                
                print(f"   ✅ {model}: {word_count} words, {'🔍 reservations found' if has_reservation else 'no reservations'}")
        
        # Compare performance
        if claude_words > gpt4o_words * 1.1:  # Claude significantly more words
            claude_wins += 1
            print("    Winner: Claude (more comprehensive)")
        elif gpt4o_words > claude_words * 1.1:  # GPT-4o significantly more words
            gpt4o_wins += 1
            print("    Winner: GPT-4o (more comprehensive)")
        else:
            ties += 1
            print("    Tie (similar performance)")
    
    print(f"\n🏆 FINAL SCORE:")
    print(f"   Claude wins: {claude_wins}")
    print(f"   GPT-4o wins: {gpt4o_wins}")
    print(f"   Ties: {ties}")
    print(f"   Total documents: {len(all_results)}")

if __name__ == "__main__":
    main() 