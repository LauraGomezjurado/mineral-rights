"""
Utility functions for the mineral rights classification pipeline
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any

def load_prompts() -> Dict[str, str]:
    """Load prompt templates from files"""
    prompts = {}
    
    prompt_dir = Path(__file__).parent.parent / "prompts"
    
    try:
        with open(prompt_dir / "classify_prompt.txt", 'r') as f:
            prompts["classify"] = f.read().strip()
    except FileNotFoundError:
        prompts["classify"] = "Default classification prompt"
    
    try:
        with open(prompt_dir / "verify_prompt.txt", 'r') as f:
            prompts["verify"] = f.read().strip()
    except FileNotFoundError:
        prompts["verify"] = "Default verification prompt"
    
    return prompts

def extract_sample_text(pdf_path: str) -> str:
    """Extract sample text from PDF for demonstration"""
    pdf_name = Path(pdf_path).name.lower()
    
    # Sample text based on file type
    if "reserv" in pdf_name or "reserv" in str(pdf_path).lower():
        return """
        DEED OF CONVEYANCE
        
        KNOW ALL MEN BY THESE PRESENTS, that John Smith and Mary Smith, husband and wife,
        of Washington County, Pennsylvania, for and in consideration of the sum of Five Thousand
        Dollars ($5,000.00), do grant and convey unto Robert Johnson, his heirs and assigns,
        
        ALL that certain piece or parcel of land situate in Washington County, Pennsylvania,
        containing 50 acres, more or less, EXCEPTING AND RESERVING unto the grantors,
        their heirs and assigns, all oil, gas, coal and other minerals in and under said land,
        together with the right to enter upon said land for the purpose of exploring,
        drilling, mining and removing the same.
        
        TO HAVE AND TO HOLD the above described premises unto the said grantee,
        his heirs and assigns forever, SUBJECT TO the mineral rights herein reserved.
        
        IN WITNESS WHEREOF, the parties have executed this deed this 15th day of March, 2023.
        """
    else:
        return """
        WARRANTY DEED
        
        KNOW ALL MEN BY THESE PRESENTS, that William Brown and Susan Brown, husband and wife,
        of Allegheny County, Pennsylvania, for and in consideration of the sum of Ten Thousand
        Dollars ($10,000.00), do grant, bargain, sell and convey unto Michael Davis,
        his heirs and assigns forever,
        
        ALL that certain lot or piece of ground situate in Allegheny County, Pennsylvania,
        being Lot No. 15 in Block 3 of the Riverside Addition, containing 0.25 acres,
        more or less, as shown on the recorded plat thereof.
        
        TO HAVE AND TO HOLD the above described premises unto the said grantee,
        his heirs and assigns forever, with all and singular the rights, privileges,
        hereditaments and appurtenances thereunto belonging.
        
        The grantors covenant that they have good right to sell and convey said premises
        and that the same are free and clear of all encumbrances.
        
        IN WITNESS WHEREOF, the parties have executed this deed this 20th day of April, 2023.
        """

def detect_key_phrases(text: str) -> List[Dict[str, Any]]:
    """Detect key phrases indicating mineral rights reservations"""
    
    # Key phrases that indicate reservations
    reservation_phrases = [
        r"reserving unto the grantor",
        r"except and reserv\w*",
        r"excepting and reserving",
        r"subject to mineral rights",
        r"coal rights reserved",
        r"oil and gas rights reserved",
        r"mineral interests reserved",
        r"mineral rights.*reserved",
        r"oil.*gas.*coal.*reserved",
        r"all minerals.*reserved"
    ]
    
    detected = []
    text_lower = text.lower()
    
    for phrase_pattern in reservation_phrases:
        matches = re.finditer(phrase_pattern, text_lower, re.IGNORECASE)
        for match in matches:
            detected.append({
                "phrase": match.group(),
                "pattern": phrase_pattern,
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.9  # High confidence for exact matches
            })
    
    return detected

def save_classification_result(result: Dict[str, Any], output_path: str):
    """Save classification result to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def load_classification_result(input_path: str) -> Dict[str, Any]:
    """Load classification result from JSON file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_confidence_score(confidence: float) -> str:
    """Format confidence score with appropriate emoji"""
    if confidence >= 0.9:
        return f"🟢 {confidence:.2f} (High)"
    elif confidence >= 0.7:
        return f"🟡 {confidence:.2f} (Medium)"
    else:
        return f"🔴 {confidence:.2f} (Low)"

def extract_legal_entities(text: str) -> Dict[str, List[str]]:
    """Extract legal entities from deed text"""
    entities = {
        "grantors": [],
        "grantees": [],
        "locations": [],
        "dates": []
    }
    
    # Simple regex patterns for demonstration
    grantor_pattern = r"(?:grantor[s]?|seller[s]?)[\s:,]*([A-Z][a-z]+ [A-Z][a-z]+)"
    grantee_pattern = r"(?:grantee[s]?|buyer[s]?)[\s:,]*([A-Z][a-z]+ [A-Z][a-z]+)"
    location_pattern = r"([A-Z][a-z]+ County, [A-Z][a-z]+)"
    date_pattern = r"(\d{1,2}(?:st|nd|rd|th)? day of [A-Z][a-z]+,? \d{4})"
    
    entities["grantors"] = re.findall(grantor_pattern, text)
    entities["grantees"] = re.findall(grantee_pattern, text)
    entities["locations"] = re.findall(location_pattern, text)
    entities["dates"] = re.findall(date_pattern, text)
    
    return entities 