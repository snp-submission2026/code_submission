import unicodedata
import math
from collections import Counter
from typing import Dict

SCRIPT_GROUPS: Dict[str, str] = {
    # Latin-based languages (English, Spanish)
    'LATIN': 'Latin',
    # South Asian languages (Bengali)
    'BENGALI': 'Bengali',
    # South Asian languages (Hindi)
    'DEVANAGARI': 'Devanagari', 
    # Middle Eastern languages (Arabic)
    'ARABIC': 'Arabic',
    "SPANISH": 'Latin',  # Spanish uses Latin script
    # Punctuation, symbols, numbers, and whitespace (Common to all)
    'COMMON': 'Common', 
    'INHERITED': 'Common',
    'UNKNOWN': 'Other'
}

def get_script_group(char: str) -> str:
    """
    Determines the simplified script group for a given character.
    """
    try:
        # Get the character's general category (e.g., Lu=Letter Uppercase, Nd=Number Decimal Digit)
        category = unicodedata.category(char)
        
        # Fast check for common/neutral characters (symbols, numbers, punctuation)
        if category.startswith('N') or category.startswith('Z') or category.startswith('P') or category.startswith('S'):
             return 'Common'
             
        # Use the name property to find the general script block
        # This is more complex but more precise for specific blocks like Bengali/Devanagari
        script_name = unicodedata.name(char).split()[0].upper()
        
        # Map to our predefined major script types
        return SCRIPT_GROUPS.get(script_name, 'Other')

    except (ValueError, IndexError):
        # Catch control characters or unmapped symbols
        return 'Other'

def calculate_script_entropy(text: str) -> float:
    """
    Calculates the Shannon entropy based on the frequency distribution of 
    scripts (Latin, Bengali, Devanagari, Arabic, Common) in the input text.
    Higher entropy indicates greater code-switching complexity.
    """
    if not text:
        return 0.0

    # 1. Tally script occurrences
    script_counts = Counter()
    total_relevant_chars = 0
    
    for char in text:
        script = get_script_group(char)
        
        # Only count characters that define the script (i.e., not pure "Common" symbols/spaces)
        if script not in ['Common', 'Other']:
            script_counts[script] += 1
            total_relevant_chars += 1
        elif script == 'Common':
            # Optionally count Common to see if the text is mostly symbols, but exclude from script mixing entropy
            # For pure script mixing, we only count script-defining characters
            pass
            
    if total_relevant_chars == 0:
        return 0.0

    # 2. Calculate Shannon Entropy: H = - sum(p_i * log2(p_i))
    entropy = 0.0
    
    for count in script_counts.values():
        if count > 0:
            # Probability p_i
            p_i = count / total_relevant_chars
            
            # H = - (p_i * log2(p_i))
            entropy -= p_i * math.log2(p_i)
            
    return round(entropy, 6)
