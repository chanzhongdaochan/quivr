from quivr_core.language.models import Language


def detect_language(text: str, low_memory: bool = True) -> Language:
    try:
        from ftlangdetect import detect
    except ImportError:
        return Language.UNKNOWN

    detected_lang = detect(text=text, low_memory=low_memory)
    try:
        detected_language = Language(detected_lang["lang"])
    except ValueError:
        return Language.UNKNOWN

    return detected_language
