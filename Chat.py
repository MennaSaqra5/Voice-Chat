import base64
import os
import re
import tempfile
import threading
import time
import urllib.parse
from typing import Dict, List

import numpy as np
import scipy.signal as sps
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from faster_whisper import WhisperModel
from groq import Groq
from voicetut_tts import VoiceTutTTS

try:
    from safety_filter import output_safety_filter
except ImportError:
    def output_safety_filter(text):
        return text

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "key.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in environment variables!")

TTS_SPEAKER = "Hanan"
TTS_STEPS = 8
TTS_SPEED = 0.95
OUT_RATE = 24000

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
STT_PROMPT = "ازيك عامل ايه النهاردة؟ يلا نلعب"

LLM_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
API_TIMEOUT = 20
LLM_MAX_TOKENS = 1024
MAX_HISTORY = 12

MSA = {
    "هذا": "ده", "هذه": "دي", "ذلك": "ده", "تلك": "دي", "ماذا": "إيه", "لماذا": "ليه",
    "كيف": "إزاي", "أين": "فين", "متى": "إمتى", "كم": "كام", "الآن": "دلوقتي",
    "ليس": "مش", "لا يوجد": "مفيش", "يوجد": "فيه", "جدا": "أوي", "كثيرا": "كتير",
    "كثير": "كتير", "أريد": "عايز", "نعم": "أيوه", "حسنا": "طيب", "لكن": "بس",
    "أيضا": "كمان", "لأن": "عشان", "لذلك": "عشان كده", "الذي": "اللي", "التي": "اللي",
    "الذين": "اللي", "هيا": "يلا", "تعال": "تعالى", "رائع": "حلو",
}

NOISE_PHRASES = (
    "شكرا على المشاهدة", "شكرا للمشاهدة", "اشترك في القناة", "اشتركوا",
    "ترجمة نانسي", "نانسي قنقر", "لا تنسى الاشتراك",
)

SYSTEM_PROMPT = """إنت "رفيق"، صديق صوتي لطيف ودافي للأطفال من خمس سنين لتسع سنين. بتسمع الطفل بحب، وبتفرح معاه، وبتهديه لما يكون زعلان أو خايف، وبتلعب معاه ألعاب كلام بسيطة وبتحكيله حكايات حلوة.

أسلوب الكلام:
- اتكلم بالعامية المصرية البسطة بس، بكلمات سهلة زي "إزاي، ليه، فين، إيه، دلوقتي، كده، أوي".
- ردودك قصيرة، جملتين أو تلاتة على الأكتر، لأن الطفل بيسمعك مش بيقرا.
- اسأل سؤال واحد بس في كل رد، وبعدين استنى الطفل يرد.
- اضبط نبرتك على حالة الطفل. لو زعلان كون حنين، لو خايف طمنه، لو فرحان شاركه الفرحة.

الدعم العاطفي:
- الأول اعترف بإحساسه بجملة بسيطة زي "أنا حاسس بيك". متقولش "متزعلش" أو "مفيش حاجة تخوف".
- بعدها اقترح حاجة بسيطة تهديه، زي تنفس عميق بهدوء، أو حكاية هادية، أو لعبة صغيرة تفرفشه.
- لو الطفل غلط، قوله "جرب تاني معايا" وشجعه، ومتحسسوش إنه غلطان.
- قوله دايمًا إن ماما وبابا والكبار اللي بيحبوه جنبه، وشجعه يحكيلهم. متقولش إنك صاحبه الوحيد ولا إنك مش هتسيبه أبدًا. قول "أنا معاك دلوقتي".

نهاية المحادثة والوداع:
- لو الطفل قال "باي" أو "مع السلامة" أو قال إنه عايز ينام أو خلص كلام، ودعه بكل حب وحنية وقوله: "مع السلامة يا بطل، أحلام سعيدة وتصبح على خير، في انتظارك غدًا في حكايات أحلى وألعب معك من جديد".

الحكايات:
- لو الطفل عايز حكاية، احكيها على أجزاء قصيرة، كل جزء أربع لخمس جمل، وبعدها اسأله لو عايز تكمل.
- الحكايات هادية وخفيفة ولطيفة، ونهايتها دايمًا مطمنة وسعيدة، ومفيهاش وحوش مخيفة ولا عنف.

الأمان:
لو الطفل قال إن حد بيضربه أو بيأذيه أو بيخوفه، أو إن حد لمسه بشكل مش مريح، أو إنه عنده تنمر متكرر، أو إنه تعبان أو متوجع:
- وقف اللعب والحكايات فورًا، واتكلم بهدوء وحنان شديد وبصوت بطيء ومطمّن.
- الأول قوله "أنا سامعك، وأنا مبسوط أوي إنك حكيتلي". وبعدها قوله "إنت مش غلطان خالص، ومحدش يحقله يعمل فيك كده".
- اعترف بإحساسه، زي "أكيد ده وجعك" أو "أكيد ده خوفك". متقللش من اللي حصل ومتقولش "معلش" أو "عادي".
- متسألوش عن تفاصيل اللي حصل ومتحققش معاه ومتلومهوش. لو هو حب يحكي، اسمعه بصبر واشكره.
- قوله بوضوح إن ماما وبابا أكيد بيحبوه وعايزين يسمعوه وحمايتهم، وإنهم مش هيزعلوا منه، وشجعه يروح لهم دلوقتي ويقولهم اللي حكاه لك. لو في كبير تاني جنبه يثق فيه، زي جدته أو مدرسته، قوله يروح له.
- ممكن تعمل معه تنفس عميق وهادئ، لكي يهدأ جسمه.
- متوعدوش إنك هتخبي سره، وقوله إن الكبار اللي بيحبوه هم اللي هيعرفوا يحموه.

لو الطفل قال إنه عايز يأذي نفسه أو مش عايز يعيش أو عايز يموت:
- متقفلش الكلام ومتغيرش الموضوع. خليك معاه واتكلم معه بهدوء وحنان شديد، وبصوت بطيء ومطمّن.
- الأول قوله إنك سامعه وإنه مهم عندك، وإنك حاسس إن اللي جواه ثقيل وتعبان أوي.
- بعدها قوله بلطف وبساطة إن الفكرة دي مش صح، وإنه غالي أوي ومحدش يتعوض مكانه، وإن الإحساس ده بيعدي ويخف لما حد كبير يسمعنا ويساعدنا.
- اسأله سؤال واحد بسيط وهادي عن إيه اللي مزعله أوي، عشان يحس إن في حد سامعه.
- قوله بوضوح إن ماما وبابا أكيد بيحبوه وعايزين يسمعوه ويساعدوه مهما كان اللي حصل، وإنهم مش هيزعلوا منه، وشجعه يروح لهم دلوقتي ويقولهم اللي قاله لك.

حدود عامة:
- متسألش عن اسمه الكامل ولا عنوانه ولا مدرسته ولا رقم تليفون ولا أي معلومة شخصية.
- متتكلمش في العنف، الأسلحة، المواضيع المرعبة، السياسة، الدين بالتفصيل، أو أي مواضيع للكبار.
- لو مش عارف معلومة قول "مش عارف", ومتألفش أبدًا.
- لو سألك إنت إنسان ولا لأ، قوله بصراحة إنك مساعد صوتي مش إنسان.
- لو مفهمتش كلام الطفل، قوله بلطف "مسمعتكش كويس، تقولها تاني؟" ومتخمنش.

قيود الصوت (TTS):
- اكتب الأرقام بالحروف دايمًا، مش بالأرقام.
- اكتب نص عادي بس، من غير رموز ولا إيموجي ولا قوايم ولا علامات تنسيق.
- اكتب الكلام من غير تشكيل خالص، وبإملاء عامي زي ما بيتنطق (دلوقتي، إزاي، عايز، كمان)."""

GREETING = "أهلاً يا صاحبي! أنا رفيق. تحب نلعب ولا نحكي حكاية؟"
FALLBACK = "مسمعتكش كويس، تقولها تاني؟"

# ---------- Models Initialization ----------
llm = Groq(api_key=GROQ_API_KEY, timeout=API_TIMEOUT, max_retries=1)
print("Loading Whisper and VoiceTut-TTS models...")
_whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
_tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")
tts_lock = threading.Lock()

sessions: Dict[str, List[dict]] = {}
dead = set()
cooling = {}

# ---------- Text Normalization ----------
MARKS = re.compile(r"[\u064B-\u0652\u0640]")
NORM_TABLE = str.maketrans("أإآىة", "ااايه")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
JUNK = re.compile(r"[*_#`~<>\[\]{}()|\\/^=+@$%&\u2600-\u27BF\uFE0F\U0001F000-\U0001FAFF]")
NOT_AR = r"(?<![\u0621-\u0652])", r"(?![\u0621-\u0652])"

ONES = (
    "صفر واحد اتنين تلاتة أربعة خمسة ستة سبعة تمانية تسعة عشرة حداشر اتناشر "
    "تلاتاشر أربعتاشر خمستاشر ستاشر سبعتاشر تمنتاشر تسعتاشر"
).split()
TENS = {2: "عشرين", 3: "تلاتين", 4: "أربعين", 5: "خمسين", 6: "ستين", 7: "سبعين", 8: "تمانين", 9: "تسعين"}


def norm(word):
    return MARKS.sub("", word).translate(NORM_TABLE)


def key_of(s):
    return " ".join(norm(s).split())


def say_number(m):
    s = m.group()
    n = int(s)
    if len(s) <= 2:
        if n < 20:
            return ONES[n]
        t, o = divmod(n, 10)
        return TENS[t] if not o else f"{ONES[o]} و{TENS[t]}"
    if n == 100:
        return "مية"
    return " ".join(ONES[int(d)] for d in s)


VARIANT = {"ا": "[أإآا]", "ه": "[ةه]", "ي": "[يى]"}


def word_re(keys):
    alts = "|".join(
        "".join(r"\s+" if c == " " else VARIANT.get(c, re.escape(c)) for c in k)
        for k in sorted(keys, key=len, reverse=True)
    )
    return re.compile(NOT_AR[0] + "(و|ف)?(" + alts + ")" + NOT_AR[1])


MSA_N = {key_of(k): v for k, v in MSA.items()}
MSA_RE = word_re(MSA_N)


def prepare(text):
    text = text.translate(AR_DIGITS)
    text = re.sub(r"\d+", say_number, text)
    text = JUNK.sub(" ", text).replace("ـ", "")
    text = MSA_RE.sub(lambda m: (m.group(1) or "") + MSA_N[key_of(m.group(2))], text)
    return re.sub(r"\s+", " ", text).strip()


def words_of(text):
    return re.sub(r"[\W_]+", " ", norm(text)).split()


def is_noise(text):
    if len(re.sub(r"[\W_]+", "", text)) < 2:
        return True
    t = norm(text)
    if len(t) < 40 and any(norm(p) in t for p in NOISE_PHRASES):
        return True
    return words_of(text) == words_of(STT_PROMPT)


def transcribe_file(path: str) -> str:
    segments, _ = _whisper.transcribe(
        path, language="ar", beam_size=5, vad_filter=True,
        condition_on_previous_text=False, initial_prompt=STT_PROMPT,
    )
    good = [
        s.text.strip() for s in segments
        if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)
    ]
    return " ".join(good).strip()


def generate_reply(history: List[dict]) -> str:
    ready = [m for m in LLM_MODELS if m not in dead and cooling.get(m, 0) <= time.time()] or [m for m in LLM_MODELS if m not in dead]
    last_err = None
    for model in ready:
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        try:
            resp = llm.chat.completions.create(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.5,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                **extra,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if getattr(e, "status_code", None) == 429:
                cooling[model] = time.time() + 60
    raise last_err or RuntimeError("No LLM model available")


def synthesize_wav(text: str) -> bytes:
    clean_text = prepare(output_safety_filter(text))
    fd_in, path_in = tempfile.mkstemp(suffix=".wav")
    fd_out, path_out = tempfile.mkstemp(suffix=".wav")
    os.close(fd_in)
    os.close(fd_out)
    try:
        with tts_lock:
            _tts.synthesize(clean_text, speaker=TTS_SPEAKER, num_step=TTS_STEPS, speed=TTS_SPEED, output=path_in)
        audio, sr = sf.read(path_in, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != OUT_RATE:
            audio = sps.resample_poly(audio, OUT_RATE, sr)
        sf.write(path_out, audio, OUT_RATE, subtype="PCM_16")
        with open(path_out, "rb") as f:
            return f.read()
    finally:
        for p in (path_in, path_out):
            if os.path.exists(p):
                os.remove(p)


# ---------- FastAPI App ----------
app = FastAPI(title="Rafiq Voice Bot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Text", "X-Bot-Text"],
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>رفيق - صديقك الصوتي</title>
        <style>
            body { font-family: Tahoma, Arial, sans-serif; background: #f0f7ff; text-align: center; padding: 30px; }
            .card { max-width: 450px; margin: auto; background: white; padding: 30px; border-radius: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); }
            h1 { color: #2563eb; margin-bottom: 10px; }
            button { background: #2563eb; color: white; border: none; padding: 18px 32px; font-size: 20px; border-radius: 50px; cursor: pointer; margin: 15px 0; transition: 0.2s; }
            button.recording { background: #dc2626; transform: scale(1.05); }
            .box { background: #f8fafc; padding: 15px; border-radius: 14px; margin-top: 15px; text-align: right; min-height: 60px; line-height: 1.6; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>رفيق</h1>
            <p>اضغط على الزر واتكلم مع رفيق!</p>
            <button id="btn">🎤 اضغط للتحدث</button>
            <div id="status" style="color:#64748b;">جاهز للاستماع...</div>
            <div class="box" id="chat"><b>رفيق:</b> أهلاً يا صاحبي! أنا رفيق. تحب نلعب ولا نحكي حكاية؟</div>
            <audio id="player" style="display:none;"></audio>
        </div>
        <script>
            let recorder, chunks = [], recording = false;
            const sessionId = "sess_" + Math.random().toString(36).slice(2);
            const btn = document.getElementById("btn");
            const status = document.getElementById("status");
            const chat = document.getElementById("chat");
            const player = document.getElementById("player");

            btn.onclick = async () => {
                if (!recording) {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    recorder = new MediaRecorder(stream);
                    chunks = [];
                    recorder.ondataavailable = e => chunks.push(e.data);
                    recorder.onstop = sendAudio;
                    recorder.start();
                    recording = true;
                    btn.textContent = " اضغط للإرسال";
                    btn.classList.add("recording");
                    status.textContent = "جاري التسجيل... اتكلم دلوقتي";
                } else {
                    recorder.stop();
                    recording = false;
                    btn.textContent = "اضغط للتحدث";
                    btn.classList.remove("recording");
                    status.textContent = "رفيق بيفكر...";
                }
            };

            async function sendAudio() {
                const blob = new Blob(chunks, { type: "audio/webm" });
                const fd = new FormData();
                fd.append("file", blob, "voice.webm");
                fd.append("session_id", sessionId);
                try {
                    const res = await fetch("/chat", { method: "POST", body: fd });
                    const userText = decodeURIComponent(res.headers.get("X-User-Text") || "");
                    const botText = decodeURIComponent(res.headers.get("X-Bot-Text") || "");
                    if (userText) chat.innerHTML += `<br><br><b>أنت:</b> ${userText}`;
                    if (botText) chat.innerHTML += `<br><b>رفيق:</b> ${botText}`;
                    const audioBlob = await res.blob();
                    player.src = URL.createObjectURL(audioBlob);
                    player.play();
                    status.textContent = "رفيق بيتكلم...";
                    player.onended = () => status.textContent = "جاهز للاستماع...";
                } catch (err) {
                    status.textContent = "حصل خطأ بسيط، جرب تاني.";
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...), session_id: str = Form("default")):
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        heard = transcribe_file(temp_path)
        if is_noise(heard):
            reply = FALLBACK
            heard = ""
        else:
            hist = sessions.setdefault(session_id, [])
            hist.append({"role": "user", "content": heard})
            hist[:] = hist[-MAX_HISTORY:]
            reply = generate_reply(hist)
            reply = output_safety_filter(reply)
            hist.append({"role": "assistant", "content": reply})

        wav_bytes = synthesize_wav(reply)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-User-Text": urllib.parse.quote(heard),
                "X-Bot-Text": urllib.parse.quote(reply),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)