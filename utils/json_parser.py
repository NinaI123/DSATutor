"""
Utility for parsing JSON from LLM responses
Handles markdown code block wrappers and other formatting issues
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(response_text: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown wrappers and mixed content
    
    Args:
        response_text: Raw text from LLM that should contain JSON
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: If valid JSON cannot be extracted
    """
    text = response_text.strip()
    
    # Pattern 1: ```json ... ```
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
        text = text.rstrip("`").strip()
        # Remove any trailing "json" or other markers
        if text.startswith("json"):
            text = text[4:].strip()
    
    # Pattern 2: ``` ... ```
    elif text.startswith("```"):
        text = text.replace("```", "", 1).strip()
        # Check if first line is 'json' marker
        if text.startswith("json"):
            text = text[4:].strip()
        text = text.rstrip("`").strip()
    
    # Pattern 3: Mixed text with JSON block inside
    # Look for JSON-like content between curly braces or square brackets
    if not (text.startswith('{') or text.startswith('[')):
        # Try to find JSON content in the text
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
    
    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        fixed_text = _fix_common_json_issues(text)
        if fixed_text != text:
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass
        
        # Log more context for debugging
        logger.error(f"Failed to parse JSON. Original response:\n{response_text[:500]}")
        logger.error(f"Cleaned text:\n{text[:500]}")
        logger.error(f"JSON Error: {e}")
        raise


def _fix_common_json_issues(text: str) -> str:
    """
    Attempt to fix common JSON formatting issues from LLM responses
    """
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix unescaped newlines inside strings (common LLM error)
    # This regex looks for newlines that are NOT followed by a quote or whitespace-quote (heuristic)
    # A better approach is to handle specific known control chars that break JSON
    
    # Replace actual newlines with escaped newlines
    # But be careful not to replace newlines outside strings if they are formatting
    # Simple brute force: replace all control chars with spaces or escapes if they are inside quotes?
    # Hard to do with regex alone.
    
    # Heuristic: Replace newlines with \n if they seem to be in a string (between quotes)
    # For now, let's just make sure we don't have literal control chars
    
    def replace_newlines(match):
        return match.group(0).replace('\n', '\\n').replace('\r', '')
        
    # Attempt to sanitize strings: escape newlines inside double quotes
    # This is complex, but often the issue is just simple multiline strings
    
    # Simply escaping all newlines in the text might break formatting, but valid JSON shouldn't have newlines
    # except as whitespace. 
    # If the LLM returned "formatted" JSON with indentation, those newlines are fine.
    # The error "Invalid control character" specifically means a literal control char INSIDE a string.
    
    # Standard cleanup: invalid control characters (0-31) except tab, newline, return
    # But JSON requires newline in string to be escaped.
    
    # Let's try flexible loading using regex to strip bad chars if needed
    # Or, specifically target the "Invalid control character" error by escaping unescaped newlines
    # that appear to be inside values.
    
    return text


def validate_response_structure(data: dict, required_keys: list) -> bool:
    """
    Validate that parsed response contains required keys
    
    Args:
        data: Parsed JSON dict
        required_keys: List of keys that must be present
        
    Returns:
        True if all required keys present, False otherwise
    """
    missing = [key for key in required_keys if key not in data]
    if missing:
        logger.warning(f"Response missing keys: {missing}")
        return False
    return True
