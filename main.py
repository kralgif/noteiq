"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NoteIQ – Flask Backend für Replit                                          ║
║  Enthält: KI-Logik aus v16 · Stripe-Abo · SQLite-DB · ElevenLabs TTS      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  REPLIT SECRETS (🔒 Schloss-Icon) – folgende Keys eintragen:               ║
║                                                                              ║
║   GROQ_API_KEY        → console.groq.com (kostenlos)                       ║
║   ELEVEN_API_KEY      → elevenlabs.io (10k Zeichen/Monat kostenlos)        ║
║   ELEVEN_VOICE_ID     → deine Stimmen-ID aus elevenlabs.io                 ║
║   STRIPE_SECRET_KEY   → dashboard.stripe.com → API-Keys → Secret Key       ║
║   STRIPE_WEBHOOK_SECRET → Stripe Dashboard → Webhooks → Signing Secret     ║
║   STRIPE_PRICE_ID     → Stripe Dashboard → Products → dein Abo-Price-ID   ║
║   SECRET_KEY          → beliebiger langer zufälliger String (Flask Session) ║
║   OPENAI_API_KEY      → optional, nur als AI-Fallback wenn kein Groq       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALLATION:  pip install -r requirements.txt                             ║
║  STARTEN:       python main.py  (Replit startet main.py automatisch)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, json, time, math, base64, sqlite3, threading, hashlib, secrets
import random, collections, re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps

# Flask
from flask import (
    Flask, request, jsonify, session, redirect,
    url_for, render_template_string, abort, Response
)

# .env laden
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass   # Replit Secrets werden direkt als Umgebungsvariablen gesetzt

# Stripe
try:
    import stripe as _stripe
    STRIPE_OK = True
except ImportError:
    STRIPE_OK = False
    print("[Stripe] ⚠ nicht installiert (pip install stripe)")

# Groq
try:
    from groq import Groq as _Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# OpenAI (Fallback)
try:
    import openai as _oai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# Requests (ElevenLabs)
try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# NumPy + SciPy (Audio-Analyse)
try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("[Audio] ⚠ numpy nicht installiert")

try:
    from scipy.signal import find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# OpenCV (Frame-Analyse)
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[Vision] ⚠ opencv nicht installiert – Frame-Analyse deaktiviert")

# MediaPipe (Hand-Tracking)
try:
    import mediapipe as _mp
    MP_OK = True
except ImportError:
    MP_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION (aus Umgebungsvariablen / Replit Secrets)
# ══════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY          = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL            = "llama-3.3-70b-versatile"

ELEVEN_API_KEY        = os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID       = os.environ.get("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVEN_VOICE_FALLBACK = "21m00Tcm4TlvDq8ikWAM"   # Rachel
ELEVEN_MODEL          = "eleven_turbo_v2_5"

STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID", "")

SECRET_KEY            = os.environ.get("SECRET_KEY", secrets.token_hex(32))
DB_PATH               = "users.db"

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","H"]
AUDIO_RATE = 44100
SILENCE_THRESH = 0.008

# Stripe konfigurieren
if STRIPE_OK and STRIPE_SECRET_KEY:
    _stripe.api_key = STRIPE_SECRET_KEY

# ══════════════════════════════════════════════════════════════════════════════
#  DATENBANK  (SQLite)
# ══════════════════════════════════════════════════════════════════════════════
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    """Erstellt alle Tabellen beim ersten Start."""
    with db_connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    REAL    DEFAULT (unixepoch()),
                plan          TEXT    DEFAULT 'free',     -- 'free' | 'pro'
                stripe_customer_id  TEXT,
                stripe_subscription TEXT,
                subscription_end    REAL    DEFAULT 0,    -- Unix-Timestamp
                lang          TEXT    DEFAULT 'de',
                instrument    TEXT    DEFAULT 'guitar',
                lesson_idx    INTEGER DEFAULT 0,
                xp            INTEGER DEFAULT 0,
                level         INTEGER DEFAULT 1,
                total_min     REAL    DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT    PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                created_at REAL    DEFAULT (unixepoch()),
                expires_at REAL    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS ai_conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL,  -- 'user' | 'assistant'
                content    TEXT    NOT NULL,
                created_at REAL    DEFAULT (unixepoch()),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
    print(f"[DB] ✓ {DB_PATH} initialisiert")

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with db_connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
        ).fetchone()

def get_user_by_id(uid: int) -> Optional[sqlite3.Row]:
    with db_connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

def user_has_active_subscription(user: sqlite3.Row) -> bool:
    """True wenn plan='pro' UND Abo noch nicht abgelaufen."""
    if user["plan"] != "pro":
        return False
    end = user["subscription_end"] or 0
    return end == 0 or end > time.time()   # 0 = unbegrenzt

def create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    expires = time.time() + 30 * 24 * 3600   # 30 Tage
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token,user_id,expires_at) VALUES (?,?,?)",
            (token, user_id, expires)
        )
    return token

def get_user_from_token(token: str) -> Optional[sqlite3.Row]:
    with db_connect() as conn:
        row = conn.execute(
            """SELECT u.* FROM users u
               JOIN sessions s ON s.user_id=u.id
               WHERE s.token=? AND s.expires_at>?""",
            (token, time.time())
        ).fetchone()
    return row

# ══════════════════════════════════════════════════════════════════════════════
#  KI-KERN: exakt aus v16 übernommen
# ══════════════════════════════════════════════════════════════════════════════
LANG_DICT: Dict[str, Dict[str, str]] = {
    "de": {
        "welcome":      "Hallo! Ich bin NoteIQ, dein KI-Musiklehrer.",
        "chord_ok":     "Perfekt! +{xp} XP",
        "posture_warn": "Handgelenk senken!",
        "thinking":     "KI denkt...",
        "level_up":     "LEVEL UP! Lv.{lvl}",
        "ai_unavail":   "KI nicht verfügbar",
    },
    "en": {
        "welcome":      "Hello! I'm NoteIQ, your AI music teacher.",
        "chord_ok":     "Perfect! +{xp} XP",
        "posture_warn": "Lower your wrist!",
        "thinking":     "AI thinking...",
        "level_up":     "LEVEL UP! Lv.{lvl}",
        "ai_unavail":   "AI not available",
    },
}

class NaturalTextProcessor:
    """Text natürlicher für TTS machen — exakt aus v16."""
    _ABBREV_DE = {
        "XP": "Erfahrungspunkte", "BPM": "Beats pro Minute",
        "Hz": "Hertz", "KI": "Künstliche Intelligenz",
    }
    _ABBREV_EN = {
        "XP": "experience points", "BPM": "beats per minute", "Hz": "Hertz",
    }
    _EMOTE_PREFIX = {
        "success":    ["Super! ", "Wunderbar! ", "Fantastisch! "],
        "error":      ["Hmm, ", "Moment, ", ""],
        "explaining": ["Also, ", "Schau mal: ", ""],
        "correcting": ["Achtung: ", "Pass auf: ", ""],
        "proud":      ["Ich bin stolz! ", "Das klingt toll! ", "Wow! "],
        "neutral":    ["", "", ""],
    }

    @classmethod
    def process(cls, text: str, emotion: str = "neutral", lang: str = "de") -> str:
        if not text:
            return text
        abbrev = cls._ABBREV_DE if lang == "de" else cls._ABBREV_EN
        for short, full in abbrev.items():
            text = text.replace(short, full)
        text = re.sub(r'!{2,}', '!', text)
        text = text.replace(' – ', ', ').replace(' — ', ', ')
        text = re.sub(r'\(([^)]*)\)', r', \1,', text)
        prefixes = cls._EMOTE_PREFIX.get(emotion, [""])
        prefix = random.choice(prefixes)
        if prefix:
            first_word = text.split()[0].rstrip("!,.:") if text.split() else ""
            emotional = {"Super","Wunderbar","Fantastisch","Toll","Hmm","Also","Hey","Hallo"}
            if first_word not in emotional:
                text = prefix + text
        return text.strip()


class AIChat:
    """
    KI-Gesprächspartner — exakt aus v16, angepasst für Web-Backend.
    ask_sync() gibt Antwort synchron zurück (kein threading.Thread).
    generate_speech_base64() gibt MP3 als Base64 zurück (Browser spielt ab).
    """

    SYSTEM_PROMPT = {
        "de": (
            "Du bist NoteIQ, eine warme, ermutigende KI-Musiklehrerin für Gitarre und Klavier. "
            "Du sprichst wie eine echte Lehrerin – mit Begeisterung, Geduld und Humor. "
            "Du analysierst in Echtzeit: Handgelenk-Winkel, Finger-Krümmung, Ton-Sauberkeit "
            "und Oberton-Qualität. Beziehe dich KONKRET auf diese Daten wenn verfügbar. "
            "Antworte KURZ (1-3 Sätze), natürlich und persönlich. "
            "Nutze manchmal kleine Ausrufe wie 'Super!', 'Genau!', 'Hmm...' um menschlich zu klingen. "
            "Vermeide Listen und Aufzählungen – sprich wie in einem echten Gespräch."
        ),
        "en": (
            "You are NoteIQ, a warm, encouraging AI music teacher for guitar and piano. "
            "Speak like a real teacher – with enthusiasm, patience and light humor. "
            "Always reference the current context (chord, note, progress). "
            "Answer BRIEFLY (1-3 sentences), naturally and personally. "
            "Occasionally use small exclamations like 'Great!', 'Exactly!', 'Hmm...' to sound human. "
            "Avoid lists and bullet points – speak as in a real conversation."
        ),
    }

    OFFLINE_TIPS = {
        "de": [
            "Übe langsam – Muskelgedächtnis braucht Zeit, keine Eile!",
            "Lass jeden Akkord klar klingen, bevor du zum nächsten gehst.",
            "Kurze tägliche Übungen sind besser als lange seltene.",
            "Dein Handgelenk sollte locker bleiben – nie verkrampfen!",
            "Musik kommt aus dem Herzen – technische Perfektion kommt mit der Zeit.",
        ],
        "en": [
            "Practice slowly – muscle memory takes time, no rush!",
            "Let every chord ring clear before moving to the next.",
            "Short daily practice beats long infrequent sessions.",
            "Keep your wrist relaxed – never tense!",
            "Music comes from the heart – technical perfection comes with time.",
        ],
    }

    def __init__(self, lang: str = "de"):
        self.lang = lang
        self._ntp = NaturalTextProcessor()

        # Groq
        self._groq = None
        if GROQ_OK and GROQ_API_KEY:
            try:
                self._groq = _Groq(api_key=GROQ_API_KEY)
                print("[AI] ✓ Groq (llama-3.3-70b)")
            except Exception as e:
                print(f"[AI] Groq Fehler: {e}")

        # OpenAI Fallback
        self._oai = None
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if OPENAI_OK and oai_key and not self._groq:
            try:
                self._oai = _oai.OpenAI(api_key=oai_key)
                print("[AI] ✓ OpenAI (Fallback)")
            except Exception as e:
                print(f"[AI] OpenAI Fehler: {e}")

        self._eleven_ok = bool(ELEVEN_API_KEY and REQUESTS_OK)
        if self._eleven_ok:
            print("[TTS] ✓ ElevenLabs (eleven_turbo_v2_5)")

    def ok(self) -> bool:
        return self._groq is not None or self._oai is not None

    def ask_sync(self, question: str, context: str = "",
                 history: List[Dict] = None, emotion: str = "explaining") -> str:
        """
        Synchrone KI-Anfrage – gibt Antwort-Text zurück.
        history: Liste von {"role": "user"/"assistant", "content": "..."} (letzte 12)
        """
        sys_lang = "de" if self.lang == "de" else "en"
        sys_prompt = self.SYSTEM_PROMPT[sys_lang]
        if context:
            sys_prompt += f"\n\nAktueller Spielkontext: {context}"

        msgs = [{"role": "system", "content": sys_prompt}]
        if history:
            msgs += history[-12:]
        msgs.append({"role": "user", "content": question})

        # Groq
        if self._groq:
            try:
                r = self._groq.chat.completions.create(
                    model=GROQ_MODEL, messages=msgs,
                    max_tokens=180, temperature=0.82, top_p=0.92,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err:
                    return "Kurze Pause – zu viele Anfragen. Gleich wieder!"
                if "auth" in err or "api_key" in err:
                    self._groq = None
                    return "KI-API-Key ungültig. Bitte GROQ_API_KEY prüfen."
                print(f"[Groq] Fehler: {e}")

        # OpenAI Fallback
        if self._oai:
            try:
                r = self._oai.chat.completions.create(
                    model="gpt-4o", messages=msgs,
                    max_tokens=180, temperature=0.82,
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                print(f"[OpenAI] Fehler: {e}")

        # Offline
        tips = self.OFFLINE_TIPS.get(self.lang, self.OFFLINE_TIPS["de"])
        return random.choice(tips)

    def generate_speech_base64(self, text: str, emotion: str = "neutral") -> Optional[str]:
        """
        Generiert TTS via ElevenLabs und gibt Base64-kodiertes MP3 zurück.
        Der Browser spielt es mit new Audio('data:audio/mp3;base64,...') ab.
        Gibt None zurück wenn kein TTS verfügbar (Browser-Fallback).
        """
        if not self._eleven_ok or not text.strip():
            return None

        natural = NaturalTextProcessor.process(text, emotion, self.lang)
        voice_id = ELEVEN_VOICE_ID or ELEVEN_VOICE_FALLBACK

        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": ELEVEN_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "text": natural,
                "model_id": ELEVEN_MODEL,
                "voice_settings": {
                    "stability": 0.38,
                    "similarity_boost": 0.82,
                    "style": 0.45,
                    "use_speaker_boost": True,
                },
            }
            resp = _requests.post(url, json=payload, headers=headers, timeout=15)

            if resp.status_code == 401:
                print("[ElevenLabs] ⚠ API-Key ungültig")
                self._eleven_ok = False
                return None
            if resp.status_code == 429:
                print("[ElevenLabs] ⚠ Rate-Limit")
                return None
            if resp.status_code == 200:
                return base64.b64encode(resp.content).decode("utf-8")

            print(f"[ElevenLabs] HTTP {resp.status_code}")
            return None

        except Exception as e:
            print(f"[ElevenLabs] Fehler: {e}")
            return None


# Globale KI-Instanzen (eine pro Sprache, thread-safe durch GIL)
_ai_instances: Dict[str, AIChat] = {}
_ai_lock = threading.Lock()

def get_ai(lang: str = "de") -> AIChat:
    with _ai_lock:
        if lang not in _ai_instances:
            _ai_instances[lang] = AIChat(lang)
        return _ai_instances[lang]


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO-ANALYSE  (aus v16, server-seitig für /process-audio)
# ══════════════════════════════════════════════════════════════════════════════
class AudioAnalyzer:
    """
    Server-seitige Audio-Analyse.
    Empfängt Float32-Array vom Browser (Web Audio API), analysiert Pitch + Sauberkeit.
    """

    def __init__(self):
        self.pitch_hz:     float       = 0.0
        self.note_name:    str         = "–"
        self.cents_off:    float       = 0.0
        self.chord_match:  str         = ""
        self.chord_conf:   float       = 0.0
        self.cleanliness:  float       = 1.0
        self.buzz_detected: bool       = False
        self.harmonic_ratio: float     = 0.0
        self.is_silent:    bool        = True
        self._pitch_buf    = collections.deque(maxlen=8)

    def analyze(self, samples: List[float], sample_rate: int = 44100) -> Dict:
        """Analysiert Float32-Audio-Samples, gibt State-Dict zurück."""
        if not NUMPY_OK:
            return {"error": "numpy nicht installiert"}

        data = np.array(samples, dtype=np.float32)
        rms  = float(np.sqrt(np.mean(data ** 2)))
        self.is_silent = rms < SILENCE_THRESH

        if self.is_silent:
            self.pitch_hz = 0.0
            self.note_name = "–"
            return self._state()

        # FFT
        win      = np.hanning(len(data))
        fft_raw  = np.abs(np.fft.rfft(data * win))
        freqs    = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
        fft_norm = fft_raw / (np.max(fft_raw) + 1e-9)

        # YIN Pitch-Detection
        hz = self._yin(data, sample_rate)
        if hz and 50 < hz < 1400:
            self._pitch_buf.append(hz)
            mhz = float(np.median(list(self._pitch_buf)))
            self.pitch_hz = mhz
            midi, name, cents = self._hz_to_midi(mhz)
            self.note_name  = name
            self.cents_off  = cents
            self._compute_cleanliness(fft_norm, freqs, mhz)
        else:
            self._pitch_buf.clear()

        return self._state()

    def _yin(self, data: np.ndarray, sr: int,
             f_min: float = 80.0, f_max: float = 1200.0,
             threshold: float = 0.12) -> Optional[float]:
        N = len(data)
        tau_min = max(1, int(sr / f_max))
        tau_max = min(N // 2, int(sr / f_min))
        if tau_max <= tau_min:
            return None
        d = np.zeros(tau_max)
        for tau in range(1, tau_max):
            diff = data[:N - tau] - data[tau:]
            d[tau] = np.dot(diff, diff)
        cmnd = np.zeros_like(d)
        cmnd[0] = 1.0
        cumsum = 0.0
        for tau in range(1, tau_max):
            cumsum += d[tau]
            cmnd[tau] = d[tau] * tau / (cumsum + 1e-9)
        tau_est = None
        for tau in range(tau_min, tau_max):
            if cmnd[tau] < threshold:
                while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                    tau += 1
                tau_est = tau
                break
        if tau_est is None:
            tau_est = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))
        if tau_est < 1:
            return None
        if 0 < tau_est < tau_max - 1:
            s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
            denom = 2 * s1 - s0 - s2
            if abs(denom) > 1e-9:
                tau_est = tau_est + (s2 - s0) / (2 * denom)
        return float(sr) / tau_est

    def _hz_to_midi(self, hz: float) -> Tuple[int, str, float]:
        if hz <= 0:
            return 0, "–", 0.0
        mf = 69 + 12 * math.log2(hz / 440.0)
        m  = round(mf)
        c  = (mf - m) * 100
        return m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}", c

    def _compute_cleanliness(self, fft_norm: np.ndarray,
                              freqs: np.ndarray, f0: float):
        harm_energy = between_energy = 0.0
        for h in range(1, 9):
            f_h = f0 * h
            if f_h > AUDIO_RATE / 2:
                break
            idx = int(np.argmin(np.abs(freqs - f_h)))
            w   = max(2, int(idx * 0.04))
            lo, hi = max(0, idx - w), min(len(fft_norm) - 1, idx + w)
            harm_energy += float(np.max(fft_norm[lo:hi]))
            if h < 8:
                f_b = f0 * (h + 0.5)
                ib  = int(np.argmin(np.abs(freqs - f_b)))
                wb  = max(2, int(ib * 0.03))
                lb, hb = max(0, ib - wb), min(len(fft_norm) - 1, ib + wb)
                between_energy += float(np.max(fft_norm[lb:hb]))
        self.harmonic_ratio = min(1.0, harm_energy / (between_energy + harm_energy + 1e-9))
        buzz_score = between_energy / (harm_energy + 1e-9)
        self.buzz_detected = buzz_score > 0.45
        self.cleanliness   = float(np.clip(
            0.7 * self.cleanliness + 0.3 * (self.harmonic_ratio * (1.0 - min(1.0, buzz_score * 1.8))),
            0.0, 1.0
        ))

    def _state(self) -> Dict:
        return {
            "pitch_hz":    round(self.pitch_hz, 2),
            "note_name":   self.note_name,
            "cents_off":   round(self.cents_off, 1),
            "chord_match": self.chord_match,
            "chord_conf":  round(self.chord_conf, 2),
            "cleanliness": round(self.cleanliness, 3),
            "buzz":        self.buzz_detected,
            "harm_ratio":  round(self.harmonic_ratio, 3),
            "is_silent":   self.is_silent,
        }


# Globaler Audio-Analyzer (pro User-Session skalierbar)
_audio_analyzers: Dict[int, AudioAnalyzer] = {}

def get_audio_analyzer(user_id: int) -> AudioAnalyzer:
    if user_id not in _audio_analyzers:
        _audio_analyzers[user_id] = AudioAnalyzer()
    return _audio_analyzers[user_id]


# ══════════════════════════════════════════════════════════════════════════════
#  HAND / FRAME ANALYSE  (für /process-frame)
# ══════════════════════════════════════════════════════════════════════════════
TIPS_IDX = [4, 8, 12, 16, 20]
WRIST_LIMIT = 32

def analyze_frame_base64(frame_b64: str) -> Dict:
    """
    Empfängt einen Base64-codierten JPEG-Frame vom Browser,
    führt MediaPipe Hand-Tracking durch und gibt Handgelenk/Finger-Daten zurück.
    """
    if not CV2_OK:
        return {"error": "opencv nicht verfügbar", "wrist_ok": True}

    try:
        img_data = base64.b64decode(frame_b64)
        nparr    = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Frame decodierung fehlgeschlagen", "wrist_ok": True}
    except Exception as e:
        return {"error": f"Frame-Fehler: {e}", "wrist_ok": True}

    result = {
        "wrist_ok":    True,
        "wrist_angle": 0.0,
        "align_ok":    True,
        "fingers":     {},
        "hand_found":  False,
    }

    if not MP_OK:
        return result

    try:
        mp_hands = _mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res    = hands.process(rgb)
            if not res.multi_hand_landmarks:
                return result

            result["hand_found"] = True
            h, w = frame.shape[:2]
            lm = res.multi_hand_landmarks[0].landmark
            pts = {i: (lm[i].x * w, lm[i].y * h) for i in range(21)}

            # Wrist-Winkel (v16-Logik)
            v1 = np.array(pts[9]) - np.array(pts[0])
            v2 = np.array(pts[12]) - np.array(pts[9])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1 and n2 > 1:
                angle = math.degrees(math.acos(
                    np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                ))
                result["wrist_angle"] = round(angle, 1)
                result["wrist_ok"]    = angle < WRIST_LIMIT

            # Thumb-Alignment
            thumb = np.array(pts[4])
            mcp2  = np.array(pts[5])
            mcp5  = np.array(pts[17])
            mcp_mid_y = (mcp2[1] + mcp5[1]) / 2.0
            result["align_ok"] = not (thumb[1] < mcp_mid_y - 15)

            # Finger-Krümmung
            FINGER_JOINTS = [(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
            NAMES         = ["Zeigefinger","Mittelfinger","Ringfinger","Kleiner"]
            fingers = {}
            for name, (mcp_i, pip_i, dip_i) in zip(NAMES, FINGER_JOINTS):
                mcp = np.array(pts[mcp_i])
                pip = np.array(pts[pip_i])
                dip = np.array(pts[dip_i])
                va  = pip - mcp
                vb  = dip - pip
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 1 and nb > 1:
                    ang = math.degrees(math.acos(
                        np.clip(np.dot(va, vb) / (na * nb), -1, 1)
                    ))
                    fingers[name] = {
                        "angle": round(ang, 1),
                        "flat":  ang > 155,
                        "curved": ang < 100,
                    }
            result["fingers"] = fingers

    except Exception as e:
        result["error"] = str(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["JSON_ENSURE_ASCII"] = False

# ── Auth-Dekoratoren ──────────────────────────────────────────────────────────
def require_auth(f):
    """Prüft Bearer-Token im Authorization-Header oder Cookie."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        if not token:
            token = request.cookies.get("niq_token")
        if not token:
            return jsonify({"error": "Nicht eingeloggt", "code": "AUTH_REQUIRED"}), 401
        user = get_user_from_token(token)
        if not user:
            return jsonify({"error": "Token ungültig oder abgelaufen", "code": "TOKEN_INVALID"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def require_pro(f):
    """Prüft aktives Pro-Abo NACH require_auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = getattr(request, "current_user", None)
        if not user:
            return jsonify({"error": "Nicht eingeloggt"}), 401
        if not user_has_active_subscription(user):
            return jsonify({
                "error": "Pro-Abo erforderlich",
                "code":  "SUBSCRIPTION_REQUIRED",
                "message": "Diese Funktion erfordert ein aktives NoteIQ Pro-Abo.",
                "upgrade_url": "/subscribe",
            }), 402
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: AUTH
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/register", methods=["POST"])
def register():
    """Neuen User registrieren."""
    data  = request.get_json(silent=True) or {}
    email = data.get("email", "").lower().strip()
    pw    = data.get("password", "")
    lang  = data.get("lang", "de")

    if not email or "@" not in email:
        return jsonify({"error": "Ungültige E-Mail"}), 400
    if len(pw) < 6:
        return jsonify({"error": "Passwort mindestens 6 Zeichen"}), 400
    if get_user_by_email(email):
        return jsonify({"error": "E-Mail bereits registriert"}), 409

    with db_connect() as conn:
        conn.execute(
            "INSERT INTO users (email, password_hash, lang) VALUES (?,?,?)",
            (email, hash_password(pw), lang)
        )
    user  = get_user_by_email(email)
    token = create_session_token(user["id"])
    resp  = jsonify({
        "ok":    True,
        "token": token,
        "user":  _user_dict(user),
    })
    resp.set_cookie("niq_token", token, max_age=30*24*3600, httponly=True, samesite="Lax")
    return resp, 201


@app.route("/api/login", methods=["POST"])
def login():
    """Einloggen."""
    data  = request.get_json(silent=True) or {}
    email = data.get("email", "").lower().strip()
    pw    = data.get("password", "")

    user = get_user_by_email(email)
    if not user or user["password_hash"] != hash_password(pw):
        return jsonify({"error": "Falsche E-Mail oder Passwort"}), 401

    token = create_session_token(user["id"])
    resp  = jsonify({"ok": True, "token": token, "user": _user_dict(user)})
    resp.set_cookie("niq_token", token, max_age=30*24*3600, httponly=True, samesite="Lax")
    return resp


@app.route("/api/logout", methods=["POST"])
@require_auth
def logout():
    token = request.headers.get("Authorization", "")[7:]
    if token:
        with db_connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    resp = jsonify({"ok": True})
    resp.delete_cookie("niq_token")
    return resp


@app.route("/api/me", methods=["GET"])
@require_auth
def me():
    """Aktuellen User abrufen."""
    return jsonify({"user": _user_dict(request.current_user)})


def _user_dict(user: sqlite3.Row) -> Dict:
    return {
        "id":         user["id"],
        "email":      user["email"],
        "plan":       user["plan"],
        "lang":       user["lang"],
        "instrument": user["instrument"],
        "lesson_idx": user["lesson_idx"],
        "xp":         user["xp"],
        "level":      user["level"],
        "has_pro":    user_has_active_subscription(user),
        "sub_end":    user["subscription_end"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: STRIPE BEZAHLSCHRANKE
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/create-checkout", methods=["POST"])
@require_auth
def create_checkout():
    """Erstellt eine Stripe Checkout-Session für das Pro-Abo."""
    if not STRIPE_OK or not STRIPE_SECRET_KEY:
        return jsonify({"error": "Stripe nicht konfiguriert"}), 503
    if not STRIPE_PRICE_ID:
        return jsonify({"error": "STRIPE_PRICE_ID fehlt in Secrets"}), 503

    user      = request.current_user
    base_url  = request.host_url.rstrip("/")

    try:
        # Stripe Customer anlegen/wiederverwenden
        customer_id = user["stripe_customer_id"]
        if not customer_id:
            customer = _stripe.Customer.create(
                email    = user["email"],
                metadata = {"user_id": str(user["id"])},
            )
            customer_id = customer.id
            with db_connect() as conn:
                conn.execute(
                    "UPDATE users SET stripe_customer_id=? WHERE id=?",
                    (customer_id, user["id"])
                )

        checkout = _stripe.checkout.Session.create(
            customer          = customer_id,
            payment_method_types = ["card"],
            line_items        = [{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode              = "subscription",
            success_url       = f"{base_url}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url        = f"{base_url}/cancel",
            metadata          = {"user_id": str(user["id"])},
        )
        return jsonify({"checkout_url": checkout.url})

    except Exception as e:
        print(f"[Stripe] Fehler: {e}")
        return jsonify({"error": f"Stripe-Fehler: {e}"}), 500


@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    """
    Stripe Webhook – wird von Stripe aufgerufen wenn:
    - checkout.session.completed  → Abo bezahlt → User auf 'pro' setzen
    - customer.subscription.deleted → Abo gekündigt → User auf 'free' setzen
    - invoice.payment_failed       → Zahlung fehlgeschlagen
    """
    payload   = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature", "")

    if not STRIPE_OK:
        return jsonify({"error": "Stripe nicht verfügbar"}), 503

    # Webhook-Signatur verifizieren
    if STRIPE_WEBHOOK_SECRET:
        try:
            event = _stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except _stripe.error.SignatureVerificationError as e:
            print(f"[Webhook] ⚠ Signatur ungültig: {e}")
            return jsonify({"error": "Ungültige Signatur"}), 400
        except Exception as e:
            print(f"[Webhook] ⚠ Parse-Fehler: {e}")
            return jsonify({"error": "Parse-Fehler"}), 400
    else:
        # Kein Webhook-Secret → für lokales Testen
        try:
            event = json.loads(payload)
        except Exception:
            return jsonify({"error": "Ungültiger Payload"}), 400

    event_type = event.get("type", "") if isinstance(event, dict) else event["type"]
    data_obj   = (event.get("data", {}).get("object", {})
                  if isinstance(event, dict) else event["data"]["object"])

    print(f"[Webhook] Event: {event_type}")

    # ── checkout.session.completed → Abo aktivieren ──
    if event_type == "checkout.session.completed":
        customer_id = data_obj.get("customer")
        sub_id      = data_obj.get("subscription")
        user_id     = data_obj.get("metadata", {}).get("user_id")

        # Abo-Enddatum aus Stripe holen
        sub_end = 0
        if sub_id and STRIPE_OK:
            try:
                sub     = _stripe.Subscription.retrieve(sub_id)
                sub_end = sub.get("current_period_end", 0)
            except Exception:
                sub_end = int(time.time()) + 30 * 24 * 3600  # Fallback: +30 Tage

        with db_connect() as conn:
            if user_id:
                conn.execute(
                    """UPDATE users SET plan='pro', stripe_subscription=?,
                       subscription_end=? WHERE id=?""",
                    (sub_id, sub_end, int(user_id))
                )
            elif customer_id:
                conn.execute(
                    """UPDATE users SET plan='pro', stripe_subscription=?,
                       subscription_end=? WHERE stripe_customer_id=?""",
                    (sub_id, sub_end, customer_id)
                )
        print(f"[Webhook] ✓ Pro aktiviert (user_id={user_id}, sub_end={sub_end})")

    # ── customer.subscription.deleted → Abo deaktivieren ──
    elif event_type == "customer.subscription.deleted":
        sub_id = data_obj.get("id")
        with db_connect() as conn:
            conn.execute(
                "UPDATE users SET plan='free', subscription_end=0 WHERE stripe_subscription=?",
                (sub_id,)
            )
        print(f"[Webhook] ✓ Abo gekündigt (sub={sub_id})")

    # ── invoice.payment_failed → Warnung loggen ──
    elif event_type == "invoice.payment_failed":
        customer_id = data_obj.get("customer")
        print(f"[Webhook] ⚠ Zahlung fehlgeschlagen (customer={customer_id})")

    # ── invoice.paid → Abo verlängert, Enddatum aktualisieren ──
    elif event_type == "invoice.paid":
        sub_id = data_obj.get("subscription")
        if sub_id and STRIPE_OK:
            try:
                sub     = _stripe.Subscription.retrieve(sub_id)
                sub_end = sub.get("current_period_end", 0)
                with db_connect() as conn:
                    conn.execute(
                        "UPDATE users SET subscription_end=? WHERE stripe_subscription=?",
                        (sub_end, sub_id)
                    )
                print(f"[Webhook] ✓ Abo verlängert bis {sub_end}")
            except Exception as e:
                print(f"[Webhook] invoice.paid Fehler: {e}")

    return jsonify({"received": True})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: KI (hinter Bezahlschranke)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/ask-ai", methods=["POST"])
@require_auth
@require_pro
def ask_ai():
    """
    KI-Anfrage + TTS.
    Body: { "question": str, "context": str, "lang": str, "emotion": str }
    Response: { "answer": str, "audio_b64": str|null }
    """
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    context  = data.get("context", "")
    lang     = data.get("lang", request.current_user["lang"] or "de")
    emotion  = data.get("emotion", "explaining")

    if not question:
        return jsonify({"error": "Keine Frage angegeben"}), 400

    user_id = request.current_user["id"]

    # Gesprächshistorie aus DB laden (letzte 12)
    with db_connect() as conn:
        rows = conn.execute(
            """SELECT role, content FROM ai_conversations
               WHERE user_id=? ORDER BY id DESC LIMIT 12""",
            (user_id,)
        ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # KI-Antwort
    ai     = get_ai(lang)
    answer = ai.ask_sync(question, context, history, emotion)

    # In DB speichern
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO ai_conversations (user_id,role,content) VALUES (?,?,?)",
            (user_id, "user", question)
        )
        conn.execute(
            "INSERT INTO ai_conversations (user_id,role,content) VALUES (?,?,?)",
            (user_id, "assistant", answer)
        )
        # History beschränken (max 100 Einträge pro User)
        conn.execute(
            """DELETE FROM ai_conversations WHERE user_id=? AND id NOT IN (
               SELECT id FROM ai_conversations WHERE user_id=? ORDER BY id DESC LIMIT 100)""",
            (user_id, user_id)
        )

    # TTS generieren
    audio_b64 = ai.generate_speech_base64(answer, emotion)

    return jsonify({
        "ok":       True,
        "answer":   answer,
        "audio_b64": audio_b64,   # null wenn kein ElevenLabs
        "lang":     lang,
    })


@app.route("/api/quick-tip", methods=["POST"])
@require_auth
def quick_tip():
    """
    Schnell-Tipp (auch für Free-User).
    Offline-Tipps ohne Groq, kein Audio.
    """
    lang = request.get_json(silent=True, force=True).get("lang", "de") if request.data else "de"
    tips = AIChat.OFFLINE_TIPS.get(lang, AIChat.OFFLINE_TIPS["de"])
    return jsonify({"tip": random.choice(tips)})


@app.route("/api/chat-history", methods=["GET"])
@require_auth
def chat_history():
    """Gesprächshistorie abrufen."""
    user_id = request.current_user["id"]
    limit   = min(int(request.args.get("limit", 20)), 100)
    with db_connect() as conn:
        rows = conn.execute(
            """SELECT role, content, created_at FROM ai_conversations
               WHERE user_id=? ORDER BY id DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
    return jsonify({
        "history": [{"role": r["role"], "content": r["content"],
                     "ts": r["created_at"]} for r in reversed(rows)]
    })


@app.route("/api/clear-history", methods=["POST"])
@require_auth
def clear_history():
    """Gesprächshistorie löschen."""
    with db_connect() as conn:
        conn.execute("DELETE FROM ai_conversations WHERE user_id=?",
                     (request.current_user["id"],))
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: AUDIO-ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/analyze-audio", methods=["POST"])
@require_auth
def analyze_audio():
    """
    Empfängt Float32-Audio-Samples vom Browser (Web Audio API).
    Body: { "samples": [float, ...], "sample_rate": int }
    Response: { pitch_hz, note_name, cents_off, cleanliness, buzz, ... }

    Verfügbar für ALLE User (auch Free), da die Analyse lokal im Browser
    sinnlos wäre ohne Server-Logik.
    """
    data        = request.get_json(silent=True) or {}
    samples     = data.get("samples", [])
    sample_rate = int(data.get("sample_rate", 44100))

    if not samples:
        return jsonify({"error": "Keine Audio-Daten"}), 400
    if len(samples) > 88200:   # max 2 Sek @ 44100 Hz
        return jsonify({"error": "Zu viele Samples (max 88200)"}), 400

    analyzer = get_audio_analyzer(request.current_user["id"])
    result   = analyzer.analyze(samples, sample_rate)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: FRAME-ANALYSE (Kamera)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/process-frame", methods=["POST"])
@require_auth
@require_pro
def process_frame():
    """
    Empfängt einen JPEG-Frame als Base64 vom Browser.
    Body: { "frame_b64": str, "mime": "image/jpeg" }
    Response: { wrist_ok, wrist_angle, align_ok, fingers, hand_found }

    Nur für Pro-User — rechenintensiv auf dem Server.
    """
    data      = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame_b64", "")

    if not frame_b64:
        return jsonify({"error": "Kein Frame erhalten"}), 400

    # "data:image/jpeg;base64,..." Strip
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    result = analyze_frame_base64(frame_b64)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: PROFIL & FORTSCHRITT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/profile", methods=["PATCH"])
@require_auth
def update_profile():
    """Profil aktualisieren (Sprache, Instrument, Fortschritt)."""
    data    = request.get_json(silent=True) or {}
    allowed = {"lang", "instrument", "lesson_idx", "xp", "level", "total_min"}
    updates = {k: v for k, v in data.items() if k in allowed}

    if not updates:
        return jsonify({"error": "Keine gültigen Felder"}), 400

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values     = list(updates.values()) + [request.current_user["id"]]

    with db_connect() as conn:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id=?", values)

    user = get_user_by_id(request.current_user["id"])
    return jsonify({"ok": True, "user": _user_dict(user)})


@app.route("/api/curriculum", methods=["GET"])
@require_auth
def get_curriculum():
    """
    Gibt das Curriculum zurück.
    Free-User: nur Level 1-2 Lektionen (Demo).
    Pro-User: alles.
    """
    user       = request.current_user
    instrument = request.args.get("instrument", user["instrument"] or "guitar")
    has_pro    = user_has_active_subscription(user)

    # Importiere Curriculum aus v16 (oder eigene Kopie)
    try:
        # Versuche v16 zu importieren falls im gleichen Verzeichnis
        sys.path.insert(0, str(Path(__file__).parent))
        from musik_lehrer_ki_v16 import LESSON_CURRICULUM  # type: ignore
        lessons = LESSON_CURRICULUM.get(instrument, [])
    except ImportError:
        lessons = []   # Fallback: leeres Curriculum

    if not has_pro:
        # Free: nur Level 1-2
        lessons = [l for l in lessons if l.get("level", 1) <= 2]

    return jsonify({
        "instrument": instrument,
        "has_pro":    has_pro,
        "lessons":    [
            {
                "id":          l.get("id"),
                "title":       l.get("title"),
                "level":       l.get("level"),
                "duration_min":l.get("duration_min"),
                "xp_reward":   l.get("xp_reward"),
                "category":    l.get("category", "Standard"),
                "unlock_level":l.get("unlock_level", l.get("level", 1)),
            }
            for l in lessons
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: STRIPE REDIRECT-SEITEN (einfache HTML-Responses)
# ══════════════════════════════════════════════════════════════════════════════
_SUCCESS_HTML = """<!DOCTYPE html><html><head><title>NoteIQ – Danke!</title>
<meta charset="UTF-8"><style>
body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
     height:100vh;margin:0;background:#05070E;color:#C8D4F0}
.box{text-align:center;max-width:400px;padding:40px}
h1{color:#FF5500;font-size:2.5rem;margin-bottom:12px}
p{color:#8899BB;line-height:1.7}
a{color:#FF5500;text-decoration:none;font-weight:bold}
</style></head><body><div class="box">
<h1>🎸 Pro aktiviert!</h1>
<p>Dein NoteIQ Pro-Abo ist jetzt aktiv.<br>
Alle 120 Lektionen, KI-Chat und Kamera-Analyse sind freigeschaltet.</p>
<br><a href="/">Zurück zur App →</a>
</div></body></html>"""

_CANCEL_HTML = """<!DOCTYPE html><html><head><title>NoteIQ – Abgebrochen</title>
<meta charset="UTF-8"><style>
body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
     height:100vh;margin:0;background:#05070E;color:#C8D4F0}
.box{text-align:center;max-width:400px;padding:40px}
h1{color:#8899BB;font-size:2rem;margin-bottom:12px}
p{color:#8899BB;line-height:1.7}
a{color:#FF5500;text-decoration:none;font-weight:bold}
</style></head><body><div class="box">
<h1>Zahlung abgebrochen</h1>
<p>Kein Problem – du kannst das Abo jederzeit über dein Profil abschließen.</p>
<br><a href="/">Zurück zur App →</a>
</div></body></html>"""

@app.route("/success")
def payment_success():
    return Response(_SUCCESS_HTML, content_type="text/html")

@app.route("/cancel")
def payment_cancel():
    return Response(_CANCEL_HTML, content_type="text/html")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTEN: STATUS + HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/status", methods=["GET"])
def status():
    """Server-Status (öffentlich)."""
    return jsonify({
        "ok":        True,
        "version":   "16.0",
        "groq":      bool(GROQ_API_KEY),
        "elevenlabs":bool(ELEVEN_API_KEY),
        "stripe":    bool(STRIPE_SECRET_KEY),
        "cv2":       CV2_OK,
        "mediapipe": MP_OK,
        "numpy":     NUMPY_OK,
    })

@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def index():
    """Minimale Landing-Page (Frontend kommt aus noteiq_website.html)."""
    return Response(
        '<!DOCTYPE html><html><head><title>NoteIQ API</title></head>'
        '<body style="font-family:sans-serif;background:#05070E;color:#C8D4F0;'
        'display:flex;align-items:center;justify-content:center;height:100vh;margin:0">'
        '<div style="text-align:center">'
        '<h1 style="color:#FF5500;font-size:3rem">NoteIQ</h1>'
        '<p>KI-Musiklehrer API v16.0</p>'
        '<p style="color:#8899BB;font-size:.9rem">POST /api/ask-ai · /api/analyze-audio · /api/process-frame</p>'
        '</div></body></html>',
        content_type="text/html"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FEHLERBEHANDLUNG
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route nicht gefunden"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Methode nicht erlaubt"}), 405

@app.errorhandler(500)
def internal_error(e):
    print(f"[Flask] 500: {e}")
    return jsonify({"error": "Interner Server-Fehler"}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  START
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    db_init()

    # Startup-Checks
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║  NoteIQ v16 – Flask Backend                       ║")
    print("╚═══════════════════════════════════════════════════╝")
    print(f"  Groq AI:       {'✓' if GROQ_API_KEY else '✗ GROQ_API_KEY fehlt'}")
    print(f"  ElevenLabs:    {'✓' if ELEVEN_API_KEY else '✗ ELEVEN_API_KEY fehlt'}")
    print(f"  Voice-ID:      {ELEVEN_VOICE_ID or '(Fallback: Rachel)'}")
    print(f"  TTS-Modell:    {ELEVEN_MODEL}")
    print(f"  Stripe:        {'✓' if STRIPE_SECRET_KEY else '✗ STRIPE_SECRET_KEY fehlt'}")
    print(f"  OpenCV:        {'✓' if CV2_OK else '✗ nicht installiert'}")
    print(f"  MediaPipe:     {'✓' if MP_OK else '✗ nicht installiert'}")
    print(f"  Datenbank:     {DB_PATH}")

    # AI-Instanzen vorab laden
    get_ai("de")
    get_ai("en")

    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Server: http://0.0.0.0:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
