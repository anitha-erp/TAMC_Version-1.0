# ===============================================================
# 🌍 TRANSLATION SERVICE - Multi-lingual Support
# Supports: Telugu (te), English (en), Hindi (hi)
# ===============================================================

import re
import logging
from functools import lru_cache
from typing import Dict, Tuple, Optional
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# ===============================================================
# AGRICULTURAL TERMINOLOGY DICTIONARY
# Custom translations for domain-specific terms
# ===============================================================

AGRICULTURAL_TERMS = {
    "te": {  # Telugu
        "quintal": "క్వింటాల్",
        "quintals": "క్వింటాల్స్",
        "moisture": "తేమ",
        "arrivals": "రాకలు",
        "arrival": "రాక",
        "bags": "సంచులు",
        "bag": "సంచి",
        "farmers": "రైతులు",
        "farmer": "రైతు",
        "market": "మార్కెట్",
        "price": "ధర",
        "prices": "ధరలు",
        "forecast": "అంచనా",
        "weather": "వాతావరణం",
        "rain": "వర్షం",
        "temperature": "ఉష్ణోగ్రత",
        "crop": "పంట",
        "crops": "పంటలు",
        "harvest": "పంట కోత",
        "yield": "దిగుబడి",
        "supply": "సరఫరా",
        "demand": "డిమాండ్",
        "commodity": "వస్తువు",
        "commodities": "వస్తువులు",
        "lots": "లాట్లు",
        "lot": "లాట్",
        "revenue": "ఆదాయం",
        "income": "ఆదాయం",
        "profit": "లాభం",
        "loss": "నష్టం",
        "trend": "ధోరణి",
        "increase": "పెరుగుదల",
        "decrease": "తగ్గుదల",
        "stable": "స్థిరంగా",
        "volatile": "అస్థిరమైన",
        "recommendation": "సిఫార్సు",
        "recommendations": "సిఫార్సులు",
        "risk": "ప్రమాదం",
        "risks": "ప్రమాదాలు",
        "opportunity": "అవకాశం",
        "opportunities": "అవకాశాలు",
        "insight": "అంతర్దృష్టి",
        "insights": "అంతర్దృష్టులు"
    },
    "hi": {  # Hindi
        "quintal": "क्विंटल",
        "quintals": "क्विंटल",
        "moisture": "नमी",
        "arrivals": "आगमन",
        "arrival": "आगमन",
        "bags": "बोरियां",
        "bag": "बोरी",
        "farmers": "किसान",
        "farmer": "किसान",
        "market": "बाजार",
        "price": "कीमत",
        "prices": "कीमतें",
        "forecast": "पूर्वानुमान",
        "weather": "मौसम",
        "rain": "बारिश",
        "temperature": "तापमान",
        "crop": "फसल",
        "crops": "फसलें",
        "harvest": "कटाई",
        "yield": "उपज",
        "supply": "आपूर्ति",
        "demand": "मांग",
        "commodity": "वस्तु",
        "commodities": "वस्तुएं",
        "lots": "लॉट",
        "lot": "लॉट",
        "revenue": "राजस्व",
        "income": "आय",
        "profit": "लाभ",
        "loss": "हानि",
        "trend": "रुझान",
        "increase": "वृद्धि",
        "decrease": "कमी",
        "stable": "स्थिर",
        "volatile": "अस्थिर",
        "recommendation": "सिफारिश",
        "recommendations": "सिफारिशें",
        "risk": "जोखिम",
        "risks": "जोखिम",
        "opportunity": "अवसर",
        "opportunities": "अवसर",
        "insight": "अंतर्दृष्टि",
        "insights": "अंतर्दृष्टि"
    }
}

# ===============================================================
# NUMBER AND UNIT PROTECTION
# Preserve numerical data during translation
# ===============================================================

def protect_numbers_and_units(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace numbers, currency, percentages, and units with placeholders
    to prevent translation. Returns modified text and placeholder mapping.
    """
    if not text:
        return text, {}
    
    placeholders = {}
    counter = 0
    
    # Patterns to protect (order matters!)
    patterns = [
        # Currency with numbers: ₹10,286 or ₹10286.50
        (r'₹[\d,]+(?:\.\d+)?', 'CURRENCY'),
        # Percentages: 12.5% or 12%
        (r'\d+(?:\.\d+)?%', 'PERCENT'),
        # Temperature: 25°C or 25 °C
        (r'\d+(?:\.\d+)?\s*°C', 'TEMP'),
        # Dates: 2025-12-10, 10/12/2025, 10-12-2025
        (r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', 'DATE'),
        (r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', 'DATE'),
        # Numbers with commas: 10,286 or 1,234,567
        (r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', 'NUMBER'),
        # Plain numbers: 123 or 123.45
        (r'\d+(?:\.\d+)?', 'NUMBER'),
    ]
    
    protected_text = text
    
    for pattern, prefix in patterns:
        matches = re.finditer(pattern, protected_text)
        for match in reversed(list(matches)):  # Reverse to maintain positions
            value = match.group()
            placeholder = f"__{prefix}{counter}__"
            placeholders[placeholder] = value
            protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
            counter += 1
    
    return protected_text, placeholders


def restore_numbers_and_units(text: str, placeholders: Dict[str, str]) -> str:
    """Restore protected numbers and units after translation"""
    if not text or not placeholders:
        return text
    
    restored_text = text
    for placeholder, original_value in placeholders.items():
        restored_text = restored_text.replace(placeholder, original_value)
    
    return restored_text


# ===============================================================
# AGRICULTURAL DICTIONARY APPLICATION
# ===============================================================

def apply_agricultural_dictionary(text: str, target_lang: str) -> str:
    """
    Replace agricultural terms with custom translations before
    sending to Google Translate. Case-insensitive replacement.
    """
    if target_lang not in AGRICULTURAL_TERMS or not text:
        return text
    
    terms_dict = AGRICULTURAL_TERMS[target_lang]
    result = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(terms_dict.items(), key=lambda x: len(x[0]), reverse=True)
    
    for english_term, translated_term in sorted_terms:
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(english_term) + r'\b'
        result = re.sub(pattern, translated_term, result, flags=re.IGNORECASE)
    
    return result


# Import commodity and location mappings
try:
    from commodity_location_mapping import apply_mappings
except ImportError:
    logger.warning("commodity_location_mapping module not found, mappings will not be applied")
    def apply_mappings(text, source_lang):
        return text

# ===============================================================
# CORE TRANSLATION FUNCTION WITH CACHING
# ===============================================================

@lru_cache(maxsize=2000)
def cached_translate(text: str, target_lang: str) -> str:
    """
    Translate text with caching for performance.
    Cache size: 2000 entries (covers most common phrases)
    """
    if not text or not text.strip():
        return text
    
    # No translation needed for English
    if target_lang == "en":
        return text
    
    # Validate target language
    if target_lang not in ["te", "hi"]:
        logger.warning(f"Unsupported language: {target_lang}, defaulting to English")
        return text
    
    try:
        # Step 1: Protect numbers and units
        protected_text, placeholders = protect_numbers_and_units(text)
        
        # Step 2: Apply agricultural dictionary
        dict_applied_text = apply_agricultural_dictionary(protected_text, target_lang)
        
        # Step 3: Translate remaining text
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = translator.translate(dict_applied_text)
        
        # Step 4: Restore protected numbers and units
        final_text = restore_numbers_and_units(translated_text, placeholders)
        
        return final_text
        
    except Exception as e:
        logger.error(f"Translation error for '{text[:50]}...' to {target_lang}: {e}")
        return text  # Fallback to original text


# ===============================================================
# REVERSE TRANSLATION (Telugu/Hindi → English)
# ===============================================================

def translate_query_to_english(query: str, source_lang: str) -> str:
    """
    Translate user query from Telugu or Hindi to English.
    
    This function is used to translate incoming user queries so that
    commodity and location names can be matched against the English
    database.
    
    Args:
        query: User query in Telugu, Hindi, or English
        source_lang: Source language code ('te', 'hi', or 'en')
    
    Returns:
        Translated query in English
    
    Example:
        >>> translate_query_to_english("రేపు వరంగల్లు చెల్లి ధర ఎంత?", "te")
        "what is tomorrow warangal chilli price?"
    """
    if not query or not query.strip():
        return query
    
    # No translation needed for English
    if source_lang == "en":
        return query
    
    # Validate source language
    if source_lang not in ["te", "hi"]:
        logger.warning(f"Unsupported source language: {source_lang}, returning original query")
        return query
    
    try:
        # Step 1: Apply commodity and location mappings first
        # This ensures accurate translation of agricultural terms
        mapped_query = apply_mappings(query, source_lang)
        
        logger.info(f"Query after mapping: {mapped_query}")
        
        # Step 2: Translate remaining text to English
        translator = GoogleTranslator(source=source_lang, target='en')
        translated_query = translator.translate(mapped_query)
        
        logger.info(f"Original query ({source_lang}): {query}")
        logger.info(f"Translated query (en): {translated_query}")
        
        return translated_query
        
    except Exception as e:
        logger.error(f"Error translating query from {source_lang} to English: {e}")
        logger.error(f"Original query: {query}")
        # Fallback: return original query to prevent crashes
        return query


# ===============================================================

# AI RESPONSE TRANSLATION
# ===============================================================

def translate_ai_response(response: Dict, target_lang: str) -> Dict:
    """
    Translate AI-generated content in the response.
    Only translates user-facing text, preserves data fields.
    
    DO NOT TRANSLATE:
    - Commodity names
    - Market/location names
    - Dates
    - Numbers and units
    
    TRANSLATE ONLY:
    - summary
    - recommendations[].action, timing, expected_outcome
    - risk_assessment.specific_risks[].risk, mitigation
    - opportunities[].opportunity, action_required
    - key_insights[].insight
    """
    if not response or target_lang == "en":
        return response
    
    try:
        # Translate summary
        if "summary" in response and response["summary"]:
            response["summary"] = cached_translate(response["summary"], target_lang)
        
        # Translate recommendations
        if "recommendations" in response and isinstance(response["recommendations"], list):
            for rec in response["recommendations"]:
                if "action" in rec:
                    rec["action"] = cached_translate(rec["action"], target_lang)
                if "timing" in rec:
                    rec["timing"] = cached_translate(rec["timing"], target_lang)
                if "expected_outcome" in rec:
                    rec["expected_outcome"] = cached_translate(rec["expected_outcome"], target_lang)
        
        # Translate risk assessment
        if "risk_assessment" in response and isinstance(response["risk_assessment"], dict):
            risks = response["risk_assessment"].get("specific_risks", [])
            if isinstance(risks, list):
                for risk in risks:
                    if "risk" in risk:
                        risk["risk"] = cached_translate(risk["risk"], target_lang)
                    if "mitigation" in risk:
                        risk["mitigation"] = cached_translate(risk["mitigation"], target_lang)
        
        # Translate opportunities
        if "opportunities" in response and isinstance(response["opportunities"], list):
            for opp in response["opportunities"]:
                if "opportunity" in opp:
                    opp["opportunity"] = cached_translate(opp["opportunity"], target_lang)
                if "action_required" in opp:
                    opp["action_required"] = cached_translate(opp["action_required"], target_lang)
        
        # Translate key insights
        if "key_insights" in response and isinstance(response["key_insights"], list):
            for insight in response["key_insights"]:
                if "insight" in insight:
                    insight["insight"] = cached_translate(insight["insight"], target_lang)
        
        # Translate market intelligence
        if "market_intelligence" in response and isinstance(response["market_intelligence"], dict):
            mi = response["market_intelligence"]
            if "supply_demand_balance" in mi:
                mi["supply_demand_balance"] = cached_translate(mi["supply_demand_balance"], target_lang)
            if "timing_strategy" in mi:
                mi["timing_strategy"] = cached_translate(mi["timing_strategy"], target_lang)
        
        # Translate interpretation reasoning
        if "interpretation" in response and isinstance(response["interpretation"], dict):
            if "reasoning" in response["interpretation"]:
                response["interpretation"]["reasoning"] = cached_translate(
                    response["interpretation"]["reasoning"], target_lang
                )
        
        return response
        
    except Exception as e:
        logger.error(f"Error translating AI response: {e}")
        return response  # Return original on error


# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================

def get_cache_info():
    """Get translation cache statistics"""
    return cached_translate.cache_info()


def clear_translation_cache():
    """Clear the translation cache"""
    cached_translate.cache_clear()
    logger.info("Translation cache cleared")


# ===============================================================
# TESTING
# ===============================================================

if __name__ == "__main__":
    # Test basic translation
    print("Testing translation service...")
    
    # Test 1: Simple translation
    text1 = "Hello, how are you?"
    print(f"\nTest 1 - Simple translation:")
    print(f"English: {text1}")
    print(f"Telugu: {cached_translate(text1, 'te')}")
    print(f"Hindi: {cached_translate(text1, 'hi')}")
    
    # Test 2: Agricultural terms
    text2 = "The quintal price is increasing due to moisture levels and farmer arrivals."
    print(f"\nTest 2 - Agricultural terms:")
    print(f"English: {text2}")
    print(f"Telugu: {cached_translate(text2, 'te')}")
    print(f"Hindi: {cached_translate(text2, 'hi')}")
    
    # Test 3: Number protection
    text3 = "Price is ₹10,286 with 12.5% increase at 25°C on 2025-12-10"
    print(f"\nTest 3 - Number protection:")
    print(f"English: {text3}")
    print(f"Telugu: {cached_translate(text3, 'te')}")
    print(f"Hindi: {cached_translate(text3, 'hi')}")
    
    # Test 4: Cache performance
    print(f"\nTest 4 - Cache info:")
    print(f"Before: {get_cache_info()}")
    cached_translate(text1, 'te')  # Should hit cache
    print(f"After repeat: {get_cache_info()}")
