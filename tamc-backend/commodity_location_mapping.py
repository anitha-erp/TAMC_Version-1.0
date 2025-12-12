# ===============================================================
# 🌍 COMMODITY AND LOCATION MAPPING
# Telugu/Hindi to English mappings for accurate translation
# ===============================================================

"""
This module provides mappings for commodity and location names from
Telugu and Hindi to English. These mappings are used before Google
Translate to ensure accurate translation of agricultural terms.
"""

# ===============================================================
# COMMODITY MAPPINGS
# ===============================================================

COMMODITY_MAPPINGS = {
    "te": {  # Telugu → English
        # Vegetables
        "చెల్లి": "chilli",
        "మిర్చి": "chilli",
        "టమాటా": "tomato",
        "టమోటా": "tomato",
        "ఉల్లిపాయ": "onion",
        "ఉల్లి": "onion",
        "బంగాళదుంప": "potato",
        "బంగాళాదుంప": "potato",
        
        # Crops
        "పత్తి": "cotton",
        "వరి": "paddy",
        "వరిబియ్యం": "paddy",
        "మొక్కజొన్న": "maize",
        "జొన్న": "maize",
        "వేరుశెనగ": "groundnut",
        "వేరుసెనగ": "groundnut",
        "కందులు": "groundnut",
        "పసుపు": "turmeric",
        "అరటి": "banana",
        "కొబ్బరి": "coconut",
        "నిమ్మ": "lemon",
        "నిమ్మకాయ": "lemon",
        
        # Generic terms
        "వస్తువు": "commodity",
        "వస్తువులు": "commodities",
        "పంట": "crop",
        "పంటలు": "crops",
    },
    "hi": {  # Hindi → English
        # Vegetables
        "मिर्च": "chilli",
        "मिर्ची": "chilli",
        "टमाटर": "tomato",
        "प्याज": "onion",
        "आलू": "potato",
        
        # Crops
        "कपास": "cotton",
        "धान": "paddy",
        "चावल": "paddy",
        "मक्का": "maize",
        "मूंगफली": "groundnut",
        "हल्दी": "turmeric",
        "केला": "banana",
        "नारियल": "coconut",
        "नींबू": "lemon",
        
        # Generic terms
        "वस्तु": "commodity",
        "वस्तुएं": "commodities",
        "फसल": "crop",
        "फसलें": "crops",
    }
}

# ===============================================================
# LOCATION MAPPINGS
# ===============================================================

LOCATION_MAPPINGS = {
    "te": {  # Telugu → English
        # Districts
        "వరంగల్": "warangal",
        "వరంగల్లు": "warangal",
        "ఖమ్మం": "khammam",
        "ఖమ్మము": "khammam",
        "హైదరాబాద్": "hyderabad",
        "కరీంనగర్": "karimnagar",
        "నిజామాబాద్": "nizamabad",
        "మహబూబ్‌నగర్": "mahbubnagar",
        "ఆదిలాబాద్": "adilabad",
        "నల్గొండ": "nalgonda",
        "మేడక్": "medak",
        "హనుమకొండ": "hanamkonda",
        "నాక్రేకల్": "nakrekal",
        "రంగారెడ్డి": "rangareddy",
        "సూర్యాపేట": "suryapet",
        "విక్రమాబాద్": "vikarabad",
        "సిద్దిపేట": "siddipet",
        "జనగామ": "jangaon",
        "వంతిమామిడి": "vantimamidi",
        "బోవెన్‌పల్లి": "bowenpally",
        
        # Generic terms
        "జిల్లా": "district",
        "మార్కెట్": "market",
        "మండలం": "mandal",
        "స్థానం": "location",
    },
    "hi": {  # Hindi → English
        # Districts
        "वारंगल": "warangal",
        "खम्मम": "khammam",
        "हैदराबाद": "hyderabad",
        "करीमनगर": "karimnagar",
        "निजामाबाद": "nizamabad",
        "महबूबनगर": "mahbubnagar",
        "आदिलाबाद": "adilabad",
        "नलगोंडा": "nalgonda",
        "मेडक": "medak",
        "हनुमकोंडा": "hanamkonda",
        "नाक्रेकल": "nakrekal",
        "रंगारेड्डी": "rangareddy",
        "सूर्यापेट": "suryapet",
        "विकाराबाद": "vikarabad",
        "सिद्दीपेट": "siddipet",
        "जनगांव": "jangaon",
        
        # Generic terms
        "जिला": "district",
        "बाजार": "market",
        "मंडल": "mandal",
        "स्थान": "location",
    }
}

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def apply_mappings(text: str, source_lang: str) -> str:
    """
    Apply commodity and location mappings to text before translation.
    
    Args:
        text: Input text in Telugu or Hindi
        source_lang: Source language code ('te' or 'hi')
    
    Returns:
        Text with mapped terms replaced with English equivalents
    """
    if not text or source_lang not in ['te', 'hi']:
        return text
    
    result = text
    
    # Apply commodity mappings
    if source_lang in COMMODITY_MAPPINGS:
        for native_term, english_term in COMMODITY_MAPPINGS[source_lang].items():
            # Case-insensitive replacement
            result = result.replace(native_term, english_term)
    
    # Apply location mappings
    if source_lang in LOCATION_MAPPINGS:
        for native_term, english_term in LOCATION_MAPPINGS[source_lang].items():
            # Case-insensitive replacement
            result = result.replace(native_term, english_term)
    
    return result


def get_all_mapped_terms(source_lang: str) -> dict:
    """
    Get all mapped terms for a given language.
    
    Args:
        source_lang: Source language code ('te' or 'hi')
    
    Returns:
        Dictionary with 'commodities' and 'locations' keys
    """
    return {
        'commodities': COMMODITY_MAPPINGS.get(source_lang, {}),
        'locations': LOCATION_MAPPINGS.get(source_lang, {})
    }


# ===============================================================
# METRIC NAME TRANSLATIONS
# For translating backend metric names to user's language
# ===============================================================

METRIC_NAME_TRANSLATIONS = {
    "en": {
        "Number of Arrivals": "Number of Arrivals",
        "Total Bags": "Total Bags",
        "Total Weight": "Total Weight",
        "Number of Lots": "Number of Lots",
        "Number of Farmers": "Number of Farmers",
        "Total Revenue": "Total Revenue",
        "Covering": "Covering",
        "days": "days",
        "day": "day",
        "arrivals": "arrivals"
    },
    "te": {
        "Number of Arrivals": "రాకల సంఖ్య",
        "Total Bags": "మొత్తం సంచులు",
        "Total Weight": "మొత్తం బరువు",
        "Number of Lots": "లాట్ల సంఖ్య",
        "Number of Farmers": "రైతుల సంఖ్య",
        "Total Revenue": "మొత్తం ఆదాయం",
        "Covering": "కవరింగ్",
        "days": "రోజులు",
        "day": "రోజు",
        "arrivals": "రాకలు"
    },
    "hi": {
        "Number of Arrivals": "आगमन की संख्या",
        "Total Bags": "कुल बोरियां",
        "Total Weight": "कुल वजन",
        "Number of Lots": "लॉट की संख्या",
        "Number of Farmers": "किसानों की संख्या",
        "Total Revenue": "कुल राजस्व",
        "Covering": "कवरिंग",
        "days": "दिन",
        "day": "दिन",
        "arrivals": "आगमन"
    }
}


def translate_metric_name(metric_name: str, target_lang: str) -> str:
    """
    Translate metric name to target language.
    
    Args:
        metric_name: English metric name
        target_lang: Target language code ('en', 'te', or 'hi')
    
    Returns:
        Translated metric name
    """
    if not metric_name or target_lang == "en":
        return metric_name
    
    if target_lang not in METRIC_NAME_TRANSLATIONS:
        return metric_name
    
    return METRIC_NAME_TRANSLATIONS[target_lang].get(metric_name, metric_name)


# ===============================================================
# TESTING
# ===============================================================

if __name__ == "__main__":
    # Test Telugu mappings
    print("Testing Telugu mappings:")
    telugu_text = "రేపు వరంగల్లు చెల్లి ధర ఎంత?"
    print(f"Original: {telugu_text}")
    print(f"Mapped: {apply_mappings(telugu_text, 'te')}")
    
    # Test Hindi mappings
    print("\nTesting Hindi mappings:")
    hindi_text = "कल वारंगल में मिर्च की कीमत क्या होगी?"
    print(f"Original: {hindi_text}")
    print(f"Mapped: {apply_mappings(hindi_text, 'hi')}")
    
    # Test metric name translation
    print("\nTesting metric name translation:")
    print(f"English: Number of Arrivals")
    print(f"Telugu: {translate_metric_name('Number of Arrivals', 'te')}")
    print(f"Hindi: {translate_metric_name('Number of Arrivals', 'hi')}")
