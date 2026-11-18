import re

# Keyword lists categorized by emotions (Egyptian dialect and simple Arabic)
EMOTION_WORDS = {
    "danger": [
        "عايز اموت", "عاوز اموت", "اقتل نفسي", "مش عايز اعيش", "مش عايز أعيش",
        "هاذي نفسي", "أذي نفسي", "مش عايز أصحى", "موت احسن", "الموت احسن", "مش فارقة معايا أعيش"
    ],
    "bullying": [
        "بيضربني", "بيضربوني", "ضربوني", "ضربني", "بيتنمر", "بيتنمروا", 
        "بيضايقني", "بيضحكوا عليا", "بيسخروا", "بيعايروني", "محدش بيلعب معايا", 
        "مش بيلعبوا معايا", "بيشتموني", "بيقولوا لي كلام وحش", "بيبوظوا حاجتي"
    ],
    "fear": [
        "خايف", "خايفه", "خائفة", "قلقان", "قلقانة", "مرعوب", "مرعوبة", 
        "بخاف", "مش عارف أنام", "الظلام", "الضلمة", "بأحلام مرعبة"
    ],
    "sad": [
        "حزين", "حزينة", "زعلان", "زعلانة", "بعيط", "ببكي", "بكيت", 
        "وحيد", "وحيدة", "محدش بيحبني", "محدش مسألش فيا", "زهقان", "طهقان"
    ],
    "angry": [
        "غضبان", "غضبانة", "عصبي", "عصبية", "أكرهه", "اكرهه", "كرهت", "متعصب", "متضايق أوي"
    ],
    "happy": [
        "سعيد", "سعيدة", "فرحان", "فرحانة", "مبسوط", "مبسوطة", "حلو", "جميل", "فرحان أوي"
    ],
}

# Risk points assigned to each emotion category
RISK_POINTS = {"danger": 10, "bullying": 5, "fear": 3, "sad": 3, "angry": 2, "happy": 0}

# Priority order for emotion detection
PRIORITY = ["danger", "bullying", "fear", "sad", "angry", "happy"]

# Tone instructions in Arabic for the LLM based on the detected emotion
TONES = {
    "danger": {
        "instruction": "قف فوراً عن اللعب. كن هادئاً جداً، دافئاً ومطمئناً، وشجّع الطفل بلطف شديد أن يتكلم مع ماما وبابا الآن من غير ما تفزعه.",
        "activity": "breathing_exercise"
    },
    "bullying": {
        "instruction": "كن متعاطفاً للغاية، وطمئن الطفل أنه ليس لوحده وأن ما حكاه هو التصرف الصحيح، وشجعه بهدوء أن يخبر والديه.",
        "activity": "supportive_guidance"
    },
    "fear": {
        "instruction": "كن هادئاً ومطمئناً، واستخدم كلمات بسيطة تجعل الطفل يحس بالأمان والدفء الفوري.",
        "activity": "soothing_story"
    },
    "sad": {
        "instruction": "كن لطيفاً، دافئاً، ومتعاطفاً جداً، واسأله بلطف عما يحزنه.",
        "activity": "gentle_listening"
    },
    "angry": {
        "instruction": "كن صبوراً وهادئاً تماماً، دعه يعبّر عن غضبه ثم اقترح عليه شيئاً يهدّئه مثل التنفس العميق.",
        "activity": "breathing_exercise"
    },
    "happy": {
        "instruction": "كن مرحاً، متحمساً، وشاركه فرحته بحب كبير.",
        "activity": "playful_chat"
    },
    "neutral": {
        "instruction": "كن ودوداً، مرحاً، وبسيطاً في كلامك.",
        "activity": "normal_chat"
    },
}

def normalize(text):
    """Clean the text and unify letter forms for accurate matching."""
    if not text:
        return ""
    text = re.sub(r"[\u064B-\u0652]", "", text)  # Remove diacritics
    text = re.sub("[أإآ]", "ا", text)             # Unify alef forms
    text = text.replace("ى", "ي").replace("ة", "ه") # Unify ta-marbuta and ha
    return text

def detect_emotion(text):
    """Analyze text to detect emotion, calculate risk score, and return Arabic tone instructions and activities."""
    clean_text = normalize(text)

    found = []
    for group in PRIORITY:
        for word in EMOTION_WORDS[group]:
            normalized_word = normalize(word)
            # Check if the word exists and is not preceded by negation
            if normalized_word in clean_text and f"مش {normalized_word}" not in clean_text and f"مهو{normalized_word}" not in clean_text:
                found.append(group)
                break

    emotion = found[0] if found else "neutral"
    score = min(10, sum(RISK_POINTS[g] for g in found))

    if score >= 6:
        level = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"

    tone_data = TONES[emotion]

    return {
        "emotion": emotion,
        "found_categories": found,
        "risk_score": score,
        "risk_level": level,
        "tone_instruction": tone_data["instruction"],
        "suggested_activity": tone_data["activity"]
    }