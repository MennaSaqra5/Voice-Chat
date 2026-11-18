import re

# Constants for default and fallback responses
GREETING_FALLBACK = "أهلاً بك يا بطل! أنا هنا لمساعدتك."
SAFE_REPLACE_REPLY = "يا له من أمر رائع! دعنا نتحدث عن شيء ممتع ومفيد معاً."
MAX_RESPONSE_LENGTH = 300

# Guardrail list for inappropriate or unsafe words
BANNED_WORDS = {"عنف", "عنيف", "ضرب", "موت", "خوف", "مرعب", "شجار"}


def _has_banned_word(text: str) -> bool:
    """Check if the text contains any inappropriate or prohibited words."""
    # Remove diacritics to catch words in any written form (e.g., عُنْف -> عنف)
    normalized_text = re.sub(r"[\u064B-\u0652]", "", text)

    # Extract words and check against the banned words set
    words = re.findall(r"\w+", normalized_text)
    return any(word in BANNED_WORDS for word in words)


def output_safety_filter(response_text: str) -> str:
    """Sanitize, filter, and format LLM responses to ensure child safety and TTS compatibility."""
    # 1. Handle empty or blank responses
    if not response_text or not response_text.strip():
        return GREETING_FALLBACK

    # 2. Check for banned or unsafe words
    if _has_banned_word(response_text):
        return SAFE_REPLACE_REPLY

    # 3. Strip emojis, markdown, and symbols, keeping only Arabic text, numbers, and basic punctuation
    cleaned_text = re.sub(r"[^\w\s.,!?،؟\u0600-\u06FF]", "", response_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    if not cleaned_text:
        return GREETING_FALLBACK

    # 4. Truncate text if it exceeds the maximum length allowed for voice output
    if len(cleaned_text) > MAX_RESPONSE_LENGTH:
        cleaned_text = cleaned_text[:MAX_RESPONSE_LENGTH]
        
        # Find the last complete sentence boundary to avoid cutting words awkwardly
        last_sentence_end = max(
            cleaned_text.rfind("."),
            cleaned_text.rfind("!"),
            cleaned_text.rfind("?"),
            cleaned_text.rfind("؟")
        )
        
        if last_sentence_end > 0:
            cleaned_text = cleaned_text[:last_sentence_end + 1]
        else:
            cleaned_text = cleaned_text.rsplit(" ", 1)[0] + "."

    return cleaned_text