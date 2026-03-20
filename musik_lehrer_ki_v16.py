#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎸  NoteIQ v16.0  –  3D KI-Assistentin NoteIQ  ·  Guitar & Piano  🎹          ║
║  Natürliche KI-Stimme · Groq LLM · ElevenLabs TTS · Aktives Zuhören      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  NEU in v13.0:                                                              ║
║  ▸ Groq API (kostenlos) – llama-3.3-70b – ultra-schnelle KI-Antworten     ║
║  ▸ ElevenLabs TTS – menschliche, natürliche Stimme mit Betonung            ║
║  ▸ Natürlicher Text-Prozessor – Pausen, Betonung, Emotionen in Sprache     ║
║  ▸ Aktives Zuhören – NoteIQ hört kontinuierlich und antwortet kontextuell   ║
║  ▸ Intelligentes Gesprächsgedächtnis (letzte 12 Nachrichten)               ║
║  ▸ Automatischer Fallback: Groq → OpenAI → Browser-TTS                    ║
║  ▸ ALLE v12-Funktionen vollständig erhalten                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALLATION:  pip install opencv-python mediapipe numpy pyaudio scipy    ║
║                 pip install groq elevenlabs requests                        ║
║  OPTIONAL:      pip install openai                                          ║
║  API-KEYS:      GROQ_API_KEY    = kostenlos auf console.groq.com           ║
║                 ELEVEN_API_KEY  = kostenlos auf elevenlabs.io (10k Zeichen) ║
║  STARTEN:       python3 musik_lehrer_ki_v13.py → http://localhost:7878     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import cv2, numpy as np
import json, os, time, math, threading, datetime, random, collections, sys
import base64, socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:    import openai as _oai;          OPENAI_OK  = True
except: OPENAI_OK  = False
try:    from groq import Groq as _Groq;  GROQ_OK    = True
except: GROQ_OK    = False
try:    import requests as _requests;    REQUESTS_OK = True
except: REQUESTS_OK = False
try:    from gtts import gTTS as _gTTS; import subprocess as _sub; TTS_OK = True
except: TTS_OK = False
try:    import mediapipe as _mp;        MP_OK      = True
except: MP_OK      = False
try:    import pyaudio as _pa;          PYAUDIO_OK = True
except: PYAUDIO_OK = False
try:    from scipy.signal import butter, lfilter, find_peaks; SCIPY_OK = True
except: SCIPY_OK   = False

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
VERSION        = "16.0"
WEB_PORT       = 7878
PROGRESS_FILE  = "user_progress.json"
HOLD_TIME      = 3.0
WRIST_LIMIT    = 32
CV_WINDOW      = "NoteIQ v16.0"
OPENAI_MODEL   = "gpt-4o"

# ── Groq (kostenlos, schnell) ──────────────────────────────────────────────
GROQ_MODEL     = "llama-3.3-70b-versatile"   # kostenlos, sehr gut
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")

# ── ElevenLabs TTS (10.000 Zeichen/Monat kostenlos) ──────────────────────
ELEVEN_API_KEY  = os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_DE = "DEZHhPbmb8LVZmWufkCh"  # Anna – young natürlich, Deutsch-fähig
ELEVEN_VOICE_EN = "cNYrMw9glwJZXR8RwbuR"  # Belle – freundlich, natürlich
ELEVEN_MODEL    = "eleven_multilingual_v2"  # unterstützt DE, EN, ES, FR

AUDIO_RATE     = 44100
AUDIO_CHUNK    = 2048
AUDIO_CHANNELS = 1
PITCH_BUF_SIZE = 8
SILENCE_THRESH = 0.008
NOTE_NAMES     = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","H"]

# Gitarren-Saiten Frequenzen (Standard-Stimmung E2–e4)
GUITAR_STRINGS_HZ = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]  # E2 A2 D3 G3 H3 e4

# Orange-Studio Palette (BGR für OpenCV)
OR = {
    "bg":        (244, 250, 255), "panel":     (230, 240, 250),
    "orange":    (  0, 107, 255), "orangeL":   ( 53, 140, 255),
    "orangeD":   (  0,  79, 212), "white":     (255, 255, 255),
    "coal":      (  8,  16,  26), "muted":     (130, 116, 160),
    "success":   ( 78, 171,  61), "danger":    ( 62,  62, 224),
    "border":    (180, 160, 200), "skeleton":  (130, 230, 150),
    "fingertip": (  0,  90, 255), "accent":    (160,  80, 255),
    "info":      (200, 160,  20), "song":      ( 60, 180, 255),
}

# ══════════════════════════════════════════════════════════════════════════════
#  SPRACHEN
# ══════════════════════════════════════════════════════════════════════════════
LANG: Dict[str,Dict[str,str]] = {
    "de": {
        "welcome":       "Hallo! Ich bin NoteIQ, dein KI-Musiklehrer.",
        "chord_hold":    "Halten! Noch {s:.1f}s",
        "chord_ok":      "Perfekt! +{xp} XP",
        "posture_warn":  "Handgelenk senken!",
        "flair_unlock":  "🌹 Flamenco & Latin freigeschaltet! Schau in den Flair-Tab!",
        "flair_tip":     "Tipp: Im Flair-Tab warten Flamenco und Bossa Nova auf dich!",
        "lesson_done":   "Lektion bestanden!",
        "calibrate":     "Klicke 4 Griffbrett-Ecken",
        "cal_done":      "Kalibrierung fertig!",
        "thinking":      "KI denkt...",
        "level_up":      "LEVEL UP! Lv.{lvl}",
        "no_hand":       "Hand zeigen",
        "next_chord":    "Naechster: {chord}",
        "guitar":        "Gitarre",         "piano":       "Klavier",
        "wrist_ok":      "Haltung OK",      "wrist_bad":   "Handgelenk!",
        "hint":          "Q:Ende H:Hilfe L:Sprache 1:Git 2:Piano M:Metro C:Kalib A:KI B:Bibliothek G:Song",
        "audio_ok":      "Ton erkannt",     "audio_bad":   "Kein Ton",
        "pitch_match":   "Ton passt!",      "pitch_close": "Fast richtig",
        "pitch_wrong":   "Falscher Ton",    "silence":     "Stille",
        "ai_unavail":    "KI nicht verfuegbar",
        "song_start":    "Song-Modus: {name}",
        "song_change":   "Wechsel! → {chord}",
        "song_done":     "Song gemeistert! +{xp} XP",
        "strum_up":      "Aufschlag ↑",     "strum_down":  "Abschlag ↓",
        "strum_ok":      "Strumming OK!",   "strum_bad":   "Rhythmus halten!",
        "poly_ok":       "Akkord klingt!",  "poly_bad":    "Saite gedämpft?",
        "tut_wrist":     "Tutorial: Handgelenk-Haltung",
        "tut_barre":     "Tutorial: Barré-Technik",
        "tut_strum":     "Tutorial: Anschlag-Technik",
        "tut_finger":    "Tutorial: Fingerkuppen",
    },
    "en": {
        "welcome":       "Welcome! I'm your AI Music Teacher.",
        "chord_hold":    "Hold! {s:.1f}s left",       "chord_ok":     "Perfect! +{xp} XP",
        "posture_warn":  "Lower your wrist!",
        "flair_unlock":  "🌹 Flamenco & Latin unlocked! Check the Flair tab!",
        "flair_tip":     "Tip: Flamenco and Bossa Nova are waiting in the Flair tab!",          "lesson_done":  "Lesson complete!",
        "calibrate":     "Click 4 fretboard corners",  "cal_done":     "Calibration done!",
        "thinking":      "AI thinking...",             "level_up":     "LEVEL UP! Lv.{lvl}",
        "no_hand":       "Show your hand",             "next_chord":   "Next: {chord}",
        "guitar":        "Guitar",                     "piano":        "Piano",
        "wrist_ok":      "Posture OK",                 "wrist_bad":    "Wrist angle!",
        "hint":          "Q:Quit H:Help L:Lang 1:Gtr 2:Piano M:Metro C:Calib A:AI B:Lib G:Song",
        "audio_ok":      "Sound detected",             "audio_bad":    "No sound",
        "pitch_match":   "Pitch matches!",             "pitch_close":  "Almost right",
        "pitch_wrong":   "Wrong note",                 "silence":      "Silence",
        "ai_unavail":    "AI not available",
        "song_start":    "Song Mode: {name}",          "song_change":  "Change! → {chord}",
        "song_done":     "Song mastered! +{xp} XP",
        "strum_up":      "Up-stroke ↑",                "strum_down":   "Down-stroke ↓",
        "strum_ok":      "Strumming OK!",              "strum_bad":    "Keep the rhythm!",
        "poly_ok":       "Chord sounds good!",         "poly_bad":     "String muted?",
        "tut_wrist":     "Tutorial: Wrist posture",    "tut_barre":    "Tutorial: Barré technique",
        "tut_strum":     "Tutorial: Strumming",        "tut_finger":   "Tutorial: Fingertips",
    },
    "es": {
        "welcome":       "¡Bienvenido! Soy tu profesor de música IA.",
        "chord_hold":    "¡Mantén! {s:.1f}s",          "chord_ok":     "¡Perfecto! +{xp} XP",
        "posture_warn":  "¡Baja la muñeca!",            "lesson_done":  "¡Lección completa!",
        "calibrate":     "Clic en 4 esquinas",          "cal_done":     "¡Calibración lista!",
        "thinking":      "IA pensando...",              "level_up":     "¡NIVEL SUPERIOR! Nv.{lvl}",
        "no_hand":       "Muestra tu mano",             "next_chord":   "Siguiente: {chord}",
        "guitar":        "Guitarra",                    "piano":        "Piano",
        "wrist_ok":      "Postura OK",                  "wrist_bad":    "¡Muñeca!",
        "hint":          "Q:Salir H:Ayuda B:Biblioteca G:Canción",
        "audio_ok":      "Sonido detectado",            "audio_bad":    "Sin sonido",
        "pitch_match":   "¡Tono correcto!",             "pitch_close":  "Casi",
        "pitch_wrong":   "Tono incorrecto",             "silence":      "Silencio",
        "ai_unavail":    "IA no disponible",
        "song_start":    "Modo canción: {name}",        "song_change":  "¡Cambia! → {chord}",
        "song_done":     "¡Canción dominada! +{xp} XP",
        "strum_up":      "Subida ↑",                    "strum_down":   "Bajada ↓",
        "strum_ok":      "¡Ritmo OK!",                  "strum_bad":    "¡Mantén el ritmo!",
        "poly_ok":       "¡Acorde suena!",              "poly_bad":     "¿Cuerda apagada?",
        "tut_wrist":     "Tutorial: Muñeca",            "tut_barre":    "Tutorial: Cejilla",
        "tut_strum":     "Tutorial: Rasgueo",           "tut_finger":   "Tutorial: Yemas",
    },
    "fr": {
        "welcome":       "Bienvenue ! Je suis votre professeur IA.",
        "chord_hold":    "Tenez ! {s:.1f}s",            "chord_ok":     "Parfait ! +{xp} XP",
        "posture_warn":  "Abaissez le poignet !",       "lesson_done":  "Leçon réussie !",
        "calibrate":     "Cliquez 4 coins",             "cal_done":     "Calibration faite !",
        "thinking":      "L'IA réfléchit...",           "level_up":     "NIVEAU SUP ! Nv.{lvl}",
        "no_hand":       "Montrez votre main",          "next_chord":   "Prochain : {chord}",
        "guitar":        "Guitare",                     "piano":        "Piano",
        "wrist_ok":      "Posture OK",                  "wrist_bad":    "Poignet !",
        "hint":          "Q:Quitter H:Aide B:Bibliothèque G:Chanson",
        "audio_ok":      "Son détecté",                 "audio_bad":    "Pas de son",
        "pitch_match":   "Note correcte !",             "pitch_close":  "Presque",
        "pitch_wrong":   "Mauvaise note",               "silence":      "Silence",
        "ai_unavail":    "IA non disponible",
        "song_start":    "Mode chanson : {name}",       "song_change":  "Changez ! → {chord}",
        "song_done":     "Chanson maîtrisée ! +{xp} XP",
        "strum_up":      "Montée ↑",                    "strum_down":   "Descente ↓",
        "strum_ok":      "Rythme OK !",                 "strum_bad":    "Gardez le rythme !",
        "poly_ok":       "Accord sonne !",              "poly_bad":     "Corde étouffée ?",
        "tut_wrist":     "Tuto : Poignet",              "tut_barre":    "Tuto : Barré",
        "tut_strum":     "Tuto : Grattage",             "tut_finger":   "Tuto : Doigts",
    },
    "zh": {
        "welcome":       "欢迎！我是您的AI音乐老师。",
        "chord_hold":    "保持！{s:.1f}秒",             "chord_ok":     "完美！+{xp} XP",
        "posture_warn":  "放低手腕！",                   "lesson_done":  "课程完成！",
        "calibrate":     "点击4个角点",                  "cal_done":     "校准完成！",
        "thinking":      "AI思考中...",                  "level_up":     "升级！{lvl}级",
        "no_hand":       "展示您的手",                   "next_chord":   "下一个：{chord}",
        "guitar":        "吉他",                        "piano":        "钢琴",
        "wrist_ok":      "姿势OK",                      "wrist_bad":    "手腕！",
        "hint":          "Q:退出 H:帮助 B:曲库 G:歌曲",
        "audio_ok":      "检测到声音",                   "audio_bad":    "无声音",
        "pitch_match":   "音高正确！",                   "pitch_close":  "接近",
        "pitch_wrong":   "错误音符",                     "silence":      "静音",
        "ai_unavail":    "AI不可用",
        "song_start":    "歌曲模式: {name}",             "song_change":  "换和弦！→ {chord}",
        "song_done":     "歌曲完成！+{xp} XP",
        "strum_up":      "上拨 ↑",                      "strum_down":   "下拨 ↓",
        "strum_ok":      "节奏OK！",                     "strum_bad":    "保持节奏！",
        "poly_ok":       "和弦发声！",                   "poly_bad":     "琴弦被闷？",
        "tut_wrist":     "教程：手腕姿势",               "tut_barre":    "教程：横按技巧",
        "tut_strum":     "教程：拨弦",                   "tut_finger":   "教程：指尖",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  VOLLSTÄNDIGE AKKORD-BIBLIOTHEK  (109 Akkorde)
# ══════════════════════════════════════════════════════════════════════════════
CHORDS: Dict[str,Dict[str,Dict]] = {
    "guitar": {
        "E":      {"name":"E",      "fingers":{1:(1,3),2:(2,4),3:(2,5)},             "open":[1,2,6],"mute":[],"diff":1,"group":"Dur",   "tip":"Vollgriff – alle 6 Saiten anschlagen"},
        "Em":     {"name":"Em",     "fingers":{1:(2,4),2:(2,5)},                     "open":[1,2,3,6],"mute":[],"diff":1,"group":"Moll", "tip":"Einfachster Akkord – 2 Finger"},
        "A":      {"name":"A",      "fingers":{1:(2,2),2:(2,3),3:(2,4)},             "open":[1,5],"mute":[6],"diff":1,"group":"Dur",   "tip":"A-Dur: 3 Finger in Reihe, Saite 6 stumm"},
        "Am":     {"name":"Am",     "fingers":{1:(2,1),2:(2,2),3:(2,3)},             "open":[5],"mute":[6],"diff":1,"group":"Moll",  "tip":"A-Moll: Basis vieler Songs"},
        "D":      {"name":"D",      "fingers":{1:(2,1),2:(2,3),3:(3,2)},             "open":[],"mute":[5,6],"diff":2,"group":"Dur",   "tip":"D-Dur: Saiten 1–4, leichtes Dreieck"},
        "Dm":     {"name":"Dm",     "fingers":{1:(1,1),2:(2,3),3:(3,2)},             "open":[],"mute":[5,6],"diff":2,"group":"Moll",  "tip":"D-Moll: Traurig klingend"},
        "G":      {"name":"G",      "fingers":{1:(2,5),2:(3,6),3:(3,1)},             "open":[2,3,4],"mute":[],"diff":2,"group":"Dur",   "tip":"G-Dur: Finger weit spreizen"},
        "C":      {"name":"C",      "fingers":{1:(1,2),2:(2,4),3:(3,5)},             "open":[1,3],"mute":[6],"diff":2,"group":"Dur",   "tip":"C-Dur: Häufigster Akkord überhaupt"},
        "F":      {"name":"F",      "fingers":{1:(1,1),2:(2,3),3:(3,4),4:(3,5)},     "open":[],"mute":[],"barre_fret":1,"diff":3,"group":"Dur",   "tip":"F-Dur: Erster Barré – übe täglich!"},
        "Bm":     {"name":"Bm",     "fingers":{1:(2,1),2:(4,2),3:(4,3),4:(4,4)},     "open":[],"mute":[6],"barre_fret":2,"diff":3,"group":"Moll",  "tip":"H-Moll: Barré im 2. Bund"},
        "E7":     {"name":"E7",     "fingers":{1:(1,3),2:(2,5)},                     "open":[1,2,4,6],"mute":[],"diff":1,"group":"Septime","tip":"E7: Blues-Klang"},
        "A7":     {"name":"A7",     "fingers":{1:(2,4),2:(2,2)},                     "open":[1,3,5],"mute":[6],"diff":1,"group":"Septime","tip":"A7: Einfacher Blues-Akkord"},
        "D7":     {"name":"D7",     "fingers":{1:(1,2),2:(2,3),3:(2,1)},             "open":[],"mute":[5,6],"diff":2,"group":"Septime","tip":"D7: leitet zu G"},
        "G7":     {"name":"G7",     "fingers":{1:(1,1),2:(2,5),3:(3,6)},             "open":[2,3,4],"mute":[],"diff":2,"group":"Septime","tip":"G7: Leitet zu C-Dur"},
        "C7":     {"name":"C7",     "fingers":{1:(1,2),2:(2,4),3:(3,3),4:(3,5)},     "open":[1],"mute":[6],"diff":3,"group":"Septime","tip":"C7: Jazz und Blues"},
        "B7":     {"name":"B7",     "fingers":{1:(1,4),2:(2,1),3:(2,3),4:(2,5)},     "open":[2],"mute":[6],"diff":3,"group":"Septime","tip":"B7: Leitet zu Em"},
        "Am7":    {"name":"Am7",    "fingers":{1:(2,2),2:(2,3)},                     "open":[1,4,5],"mute":[6],"diff":1,"group":"Septime","tip":"Am7: Weicher Moll-Sept"},
        "Em7":    {"name":"Em7",    "fingers":{1:(2,4),2:(2,5)},                     "open":[1,2,3,6],"mute":[],"diff":1,"group":"Septime","tip":"Em7: Flüssiger Sound"},
        "Dm7":    {"name":"Dm7",    "fingers":{1:(1,1),2:(2,3)},                     "open":[4],"mute":[5,6],"diff":2,"group":"Septime","tip":"Dm7: Jazz-Farbe"},
        "Cmaj7":  {"name":"Cmaj7",  "fingers":{1:(2,4),2:(3,5)},                     "open":[1,2,3],"mute":[6],"diff":2,"group":"Maj7",  "tip":"Cmaj7: Romantischer Klang"},
        "Gmaj7":  {"name":"Gmaj7",  "fingers":{1:(2,5),2:(3,6)},                     "open":[1,2,3,4],"mute":[],"diff":2,"group":"Maj7",  "tip":"Gmaj7: Schwebender Sound"},
        "Amaj7":  {"name":"Amaj7",  "fingers":{1:(1,3),2:(2,2),3:(2,4)},             "open":[1,5],"mute":[6],"diff":2,"group":"Maj7",  "tip":"Amaj7: Häufig in Pop"},
        "Fmaj7":  {"name":"Fmaj7",  "fingers":{1:(1,1),2:(2,3),3:(3,4)},             "open":[],"mute":[],"barre_fret":1,"diff":3,"group":"Maj7",  "tip":"Fmaj7: F mit weicher Septime"},
        "Emaj7":  {"name":"Emaj7",  "fingers":{1:(1,3),2:(1,4),3:(2,5)},             "open":[1,2,6],"mute":[],"diff":2,"group":"Maj7",  "tip":"Emaj7: Jazzig"},
        "Bm7":    {"name":"Bm7",    "fingers":{1:(2,1),2:(2,3),3:(2,4)},             "open":[2],"mute":[6],"barre_fret":2,"diff":3,"group":"m7",    "tip":"Bm7: Sanfter als Bm"},
        "Fm7":    {"name":"Fm7",    "fingers":{1:(1,1),2:(1,2),3:(3,4)},             "open":[],"mute":[],"barre_fret":1,"diff":3,"group":"m7",    "tip":"Fm7: Moll-Sept im 1. Bund"},

        # ── Spanish / Flamenco / Latin (freigeschaltet ab Level 3) ────────────
        "E_Phryg": {"name":"E Phrygisch","fingers":{1:(1,3),2:(2,4),3:(3,5)},       "open":[1,2,6],"mute":[],"diff":3,"group":"Spanish","tip":"E-Phrygisch: Basis des Flamenco – öffnet die spanische Klangwelt","min_level":3},
        "F_Maj":   {"name":"F (Flamenco)","fingers":{1:(1,1),2:(2,3),3:(3,4),4:(3,5)},"open":[],"mute":[],"barre_fret":1,"diff":3,"group":"Spanish","tip":"F als Barré – im Flamenco folgt es immer auf E","min_level":3},
        "Am_Flam": {"name":"Am (Flamenco)","fingers":{1:(2,2),2:(2,3),3:(2,4)},     "open":[5],"mute":[6],"diff":2,"group":"Spanish","tip":"Am Flamenco: weich und ausdrucksvoll zupfen","min_level":3},
        "G_Flam":  {"name":"G (Rasgueado)","fingers":{1:(2,5),2:(3,6),3:(3,1)},     "open":[2,3,4],"mute":[],"diff":3,"group":"Spanish","tip":"G für Rasgueado – alle 4 Finger schnell fächern","min_level":3},
        "Dm_Flam": {"name":"Dm (Flamenco)","fingers":{1:(1,1),2:(2,3),3:(3,2)},    "open":[],"mute":[5,6],"diff":2,"group":"Spanish","tip":"Dm im Flamenco – traurig und leidenschaftlich","min_level":3},
        "E7_Span": {"name":"E7 (Spanish)", "fingers":{1:(1,3),2:(2,5)},             "open":[1,2,4,6],"mute":[],"diff":2,"group":"Spanish","tip":"E7 Spanish – erzeugt Spannung die sich zu Am auflöst","min_level":3},
        "Am7_Lat":  {"name":"Am7 (Latin)",  "fingers":{1:(2,2),2:(2,3)},             "open":[1,4,5],"mute":[6],"diff":1,"group":"Latin",  "tip":"Am7 Latin: Bossa-Nova & Samba Basis","min_level":3},
        "D9":      {"name":"D9 (Bossa)",   "fingers":{1:(2,1),2:(2,3),3:(3,2),4:(1,4)},"open":[],"mute":[5,6],"diff":3,"group":"Latin","tip":"D9: Jazz-Bossa Klang – harmonisch reich","min_level":4},
        "G7_Lat":  {"name":"G7 (Latin)",   "fingers":{1:(1,1),2:(2,5),3:(3,6)},     "open":[2,3,4],"mute":[],"diff":2,"group":"Latin",  "tip":"G7 Latin: Entspannte lateinische Atmosphäre","min_level":3},
        "Cmaj9":   {"name":"Cmaj9",        "fingers":{1:(3,5),2:(3,4)},              "open":[1,2,3],"mute":[6],"diff":3,"group":"Latin",  "tip":"Cmaj9: Bossa-Nova Traumklang","min_level":4},
        "Fmaj9":   {"name":"Fmaj9",        "fingers":{1:(1,1),2:(3,4),3:(3,3)},     "open":[1,2],"mute":[],"barre_fret":1,"diff":3,"group":"Latin","tip":"Fmaj9: Weicher Latin-Jazz-Akkord","min_level":4},
        "Asus2":  {"name":"Asus2",  "fingers":{1:(2,2),2:(2,4)},                     "open":[1,4,5],"mute":[6],"diff":2,"group":"Sus",   "tip":"Asus2: Schwebend, ohne Terz"},
        "Asus4":  {"name":"Asus4",  "fingers":{1:(2,2),2:(2,3),3:(2,4)},             "open":[1,5],"mute":[6],"diff":1,"group":"Sus",   "tip":"Asus4: Spannung vor Auflösung"},
        "Dsus2":  {"name":"Dsus2",  "fingers":{1:(2,1),2:(3,2)},                     "open":[3],"mute":[5,6],"diff":2,"group":"Sus",   "tip":"Dsus2: Offen und hell"},
        "Dsus4":  {"name":"Dsus4",  "fingers":{1:(2,1),2:(3,2),3:(3,3)},             "open":[],"mute":[5,6],"diff":2,"group":"Sus",   "tip":"Dsus4: Typisch Rock"},
        "Esus4":  {"name":"Esus4",  "fingers":{1:(2,3),2:(2,4),3:(2,5)},             "open":[1,2,6],"mute":[],"diff":2,"group":"Sus",   "tip":"Esus4: Power-Rock Sound"},
        "Gsus4":  {"name":"Gsus4",  "fingers":{1:(1,1),2:(2,5),3:(3,6)},             "open":[2,3,4],"mute":[],"diff":2,"group":"Sus",   "tip":"Gsus4: Schwebend"},
        "Cadd9":  {"name":"Cadd9",  "fingers":{1:(2,4),2:(3,2),3:(3,5)},             "open":[1,3],"mute":[6],"diff":2,"group":"Add",   "tip":"Cadd9: Modern und voll"},
        "Gadd9":  {"name":"Gadd9",  "fingers":{1:(2,5),2:(3,6),3:(3,1),4:(2,3)},     "open":[2,4],"mute":[],"diff":3,"group":"Add",   "tip":"Gadd9: Oasis, Coldplay"},
        "Dadd9":  {"name":"Dadd9",  "fingers":{1:(2,1),2:(2,3),3:(3,2),4:(3,4)},     "open":[],"mute":[5,6],"diff":3,"group":"Add",   "tip":"Dadd9: Reicher D-Klang"},
        "E5":     {"name":"E5",     "fingers":{1:(2,4),2:(2,5)},                     "open":[6],"mute":[1,2,3],"diff":1,"group":"Power","tip":"E5: Punk & Rock Basis"},
        "A5":     {"name":"A5",     "fingers":{1:(2,4),2:(2,5)},                     "open":[5],"mute":[1,2,3,6],"diff":1,"group":"Power","tip":"A5: Power Chord auf A"},
        "D5":     {"name":"D5",     "fingers":{1:(2,3),2:(2,4)},                     "open":[4],"mute":[1,2,5,6],"diff":1,"group":"Power","tip":"D5: Rock D-Dur Power"},
        "G5":     {"name":"G5",     "fingers":{1:(2,5),2:(2,4)},                     "open":[6],"mute":[1,2,3],"diff":1,"group":"Power","tip":"G5: Power Rock G"},
        "C5":     {"name":"C5",     "fingers":{1:(3,5),2:(3,4)},                     "open":[],"mute":[1,2,3,6],"diff":2,"group":"Power","tip":"C5: C Power Chord"},
        "F5":     {"name":"F5",     "fingers":{1:(1,4),2:(3,4),3:(3,5)},             "open":[],"mute":[1,2,3,6],"barre_fret":1,"diff":2,"group":"Power","tip":"F5: Power im 1. Bund"},
        "Bdim":   {"name":"Bdim",   "fingers":{1:(1,4),2:(2,3),3:(3,1)},             "open":[],"mute":[5,6],"diff":3,"group":"Dim",   "tip":"Bdim: Verminderter Akkord"},
        "Fdim":   {"name":"Fdim",   "fingers":{1:(1,4),2:(2,2),3:(3,3)},             "open":[],"mute":[5,6],"diff":3,"group":"Dim",   "tip":"Fdim: Dramatischer Klang"},
        "Adim":   {"name":"Adim",   "fingers":{1:(1,3),2:(2,1),3:(3,2)},             "open":[],"mute":[5,6],"diff":3,"group":"Dim",   "tip":"Adim: Leitet zu Bb"},
        "Eaug":   {"name":"Eaug",   "fingers":{1:(1,3),2:(2,4),3:(2,5),4:(2,2)},     "open":[1,6],"mute":[],"diff":3,"group":"Aug",   "tip":"Eaug: Übermäßig"},
        "F#m":    {"name":"F#m",    "fingers":{1:(2,1),2:(4,2),3:(4,3),4:(4,4)},     "open":[],"mute":[],"barre_fret":2,"diff":3,"group":"Barré","tip":"F#m: Barré im 2. Bund"},
        "Bb":     {"name":"Bb",     "fingers":{1:(1,1),2:(2,3),3:(3,4),4:(3,5)},     "open":[],"mute":[],"barre_fret":1,"diff":3,"group":"Barré","tip":"Bb: B-Dur Barré"},
        "B":      {"name":"B",      "fingers":{1:(2,1),2:(4,2),3:(4,3),4:(4,4)},     "open":[],"mute":[6],"barre_fret":2,"diff":3,"group":"Barré","tip":"B-Dur: Barré im 2. Bund"},
        "C#m":    {"name":"C#m",    "fingers":{1:(4,1),2:(6,2),3:(6,3),4:(6,4)},     "open":[],"mute":[6],"barre_fret":4,"diff":3,"group":"Barré","tip":"C#m: Barré im 4. Bund"},
        "Gm":     {"name":"Gm",     "fingers":{1:(3,1),2:(5,2),3:(5,3),4:(5,4)},     "open":[],"mute":[],"barre_fret":3,"diff":3,"group":"Barré","tip":"G-Moll: Barré im 3. Bund"},
        "Em7open":{"name":"Em7*",   "fingers":{1:(2,4)},                             "open":[1,2,3,5,6],"mute":[],"diff":1,"group":"Fingerpick","tip":"Em7 offen: Fingerpicking"},
        "Dsus2x": {"name":"Dsus2*", "fingers":{1:(2,1)},                             "open":[2,3],"mute":[5,6,4],"diff":1,"group":"Fingerpick","tip":"Dsus2 offen: Minimalismus"},
        "E7b9":   {"name":"E7b9",   "fingers":{1:(1,3),2:(2,5),3:(2,2),4:(3,4)},     "open":[1,6],"mute":[],"diff":3,"group":"Jazz",  "tip":"E7b9: Jazz Dominante"},
        "Am9":    {"name":"Am9",    "fingers":{1:(2,2)},                             "open":[1,3,4,5],"mute":[6],"diff":2,"group":"Jazz",  "tip":"Am9: Weicher Jazz-Moll"},
        "Dm9":    {"name":"Dm9",    "fingers":{1:(1,1),2:(2,4)},                     "open":[3],"mute":[5,6],"diff":3,"group":"Jazz",  "tip":"Dm9: Üppiger Jazz-Klang"},
    },
    "piano": {
        "C-Dur":  {"name":"C-Dur",  "keys":[0,4,7],     "fingers_rh":{1:0,3:1,5:2},"diff":1,"group":"Dur",   "tip":"C–E–G: Daumen(1), Mittelfinger(3), Kleiner(5)"},
        "D-Dur":  {"name":"D-Dur",  "keys":[2,6,9],     "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Dur",   "tip":"D–F#–A: Beachte F#"},
        "E-Dur":  {"name":"E-Dur",  "keys":[4,8,11],    "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Dur",   "tip":"E–G#–H: Zwei schwarze Tasten"},
        "F-Dur":  {"name":"F-Dur",  "keys":[5,9,0],     "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Dur",   "tip":"F–A–C: Häufiger Dur-Akkord"},
        "G-Dur":  {"name":"G-Dur",  "keys":[7,11,2],    "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Dur",   "tip":"G–H–D: Klassischer Begleitakkord"},
        "A-Dur":  {"name":"A-Dur",  "keys":[9,1,4],     "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Dur",   "tip":"A–C#–E: Beachte C#"},
        "H-Dur":  {"name":"H-Dur",  "keys":[11,3,6],    "fingers_rh":{1:0,3:1,5:2},"diff":3,"group":"Dur",   "tip":"H–D#–F#: Zwei schwarze Tasten"},
        "C-Moll": {"name":"C-Moll", "keys":[0,3,7],     "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Moll",  "tip":"C–Eb–G: Kleines Terz-Intervall"},
        "D-Moll": {"name":"D-Moll", "keys":[2,5,9],     "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Moll",  "tip":"D–F–A: Traurig klingend"},
        "E-Moll": {"name":"E-Moll", "keys":[4,7,11],    "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Moll",  "tip":"E–G–H: Sehr häufig"},
        "F-Moll": {"name":"F-Moll", "keys":[5,8,0],     "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Moll",  "tip":"F–Ab–C: Tiefer und dunkel"},
        "G-Moll": {"name":"G-Moll", "keys":[7,10,2],    "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Moll",  "tip":"G–Bb–D: Dramatisch"},
        "A-Moll": {"name":"A-Moll", "keys":[9,0,4],     "fingers_rh":{1:0,2:1,5:2},"diff":1,"group":"Moll",  "tip":"A–C–E: Trauriger Grundakkord"},
        "H-Moll": {"name":"H-Moll", "keys":[11,2,6],    "fingers_rh":{1:0,2:1,5:2},"diff":3,"group":"Moll",  "tip":"H–D–F#: Oft in Molltonarten"},
        "C7":     {"name":"C7",     "keys":[0,4,7,10],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"C–E–G–Bb: Dominant-Sept leitet zu F"},
        "D7":     {"name":"D7",     "keys":[2,6,9,0],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"D–F#–A–C: Leitet zu G"},
        "E7":     {"name":"E7",     "keys":[4,8,11,2],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"E–G#–H–D: Blues-Feeling"},
        "F7":     {"name":"F7",     "keys":[5,9,0,3],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"F–A–C–Eb: Swing und Jazz"},
        "G7":     {"name":"G7",     "keys":[7,11,2,5],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"G–H–D–F: Häufigste Dominante"},
        "A7":     {"name":"A7",     "keys":[9,1,4,7],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Sept","tip":"A–C#–E–G: Blues-Rock"},
        "H7":     {"name":"H7",     "keys":[11,3,6,9],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":3,"group":"Sept","tip":"H–D#–F#–A: Leitet zu Em"},
        "Cmaj7":  {"name":"Cmaj7",  "keys":[0,4,7,11],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Maj7","tip":"C–E–G–H: Romantisch"},
        "Dmaj7":  {"name":"Dmaj7",  "keys":[2,6,9,1],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Maj7","tip":"D–F#–A–C#: Hell"},
        "Emaj7":  {"name":"Emaj7",  "keys":[4,8,11,3],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":3,"group":"Maj7","tip":"E–G#–H–D#: Jazzig"},
        "Fmaj7":  {"name":"Fmaj7",  "keys":[5,9,0,4],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Maj7","tip":"F–A–C–E: Warm"},
        "Gmaj7":  {"name":"Gmaj7",  "keys":[7,11,2,6],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Maj7","tip":"G–H–D–F#: Bossa Nova"},
        "Amaj7":  {"name":"Amaj7",  "keys":[9,1,4,8],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"Maj7","tip":"A–C#–E–G#: Pop"},
        "Cm7":    {"name":"Cm7",    "keys":[0,3,7,10],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"C–Eb–G–Bb: Blues/Funk"},
        "Dm7":    {"name":"Dm7",    "keys":[2,5,9,0],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"D–F–A–C: Klassischer Jazz"},
        "Em7":    {"name":"Em7",    "keys":[4,7,11,2],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"E–G–H–D: Weich"},
        "Fm7":    {"name":"Fm7",    "keys":[5,8,0,3],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"F–Ab–C–Eb: Soul"},
        "Gm7":    {"name":"Gm7",    "keys":[7,10,2,5],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"G–Bb–D–F: Modal"},
        "Am7":    {"name":"Am7",    "keys":[9,0,4,7],   "fingers_rh":{1:0,2:1,3:2,5:3},"diff":2,"group":"m7",  "tip":"A–C–E–G: Beliebt"},
        "Hm7":    {"name":"Hm7",    "keys":[11,2,6,9],  "fingers_rh":{1:0,2:1,3:2,5:3},"diff":3,"group":"m7",  "tip":"H–D–F#–A: In Dur-Tonarten"},
        "Cdim":   {"name":"Cdim",   "keys":[0,3,6],     "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"C–Eb–Gb: Spannung"},
        "Ddim":   {"name":"Ddim",   "keys":[2,5,8],     "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"D–F–Ab: Dissonant"},
        "Edim":   {"name":"Edim",   "keys":[4,7,10],    "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"E–G–Bb: Führungston"},
        "Fdim":   {"name":"Fdim",   "keys":[5,8,11],    "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"F–Ab–H: Leitet zu E"},
        "Gdim":   {"name":"Gdim",   "keys":[7,10,1],    "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"G–Bb–Db: Dramatisch"},
        "Adim":   {"name":"Adim",   "keys":[9,0,3],     "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"A–C–Eb: Spannungsgeladen"},
        "Hdim":   {"name":"Hdim",   "keys":[11,2,5],    "fingers_rh":{1:0,2:1,4:2},"diff":2,"group":"Dim",  "tip":"H–D–F: Halbvermindert"},
        "Caug":   {"name":"Caug",   "keys":[0,4,8],     "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Aug",  "tip":"C–E–G#: Traumhaft"},
        "Faug":   {"name":"Faug",   "keys":[5,9,1],     "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Aug",  "tip":"F–A–C#: Magisch"},
        "Gaug":   {"name":"Gaug",   "keys":[7,11,3],    "fingers_rh":{1:0,3:1,5:2},"diff":2,"group":"Aug",  "tip":"G–H–D#: Romantisch"},
        "Csus2":  {"name":"Csus2",  "keys":[0,2,7],     "fingers_rh":{1:0,2:1,5:2},"diff":2,"group":"Sus",  "tip":"C–D–G: Schwebend"},
        "Csus4":  {"name":"Csus4",  "keys":[0,5,7],     "fingers_rh":{1:0,4:1,5:2},"diff":2,"group":"Sus",  "tip":"C–F–G: Spannung"},
        "Dsus4":  {"name":"Dsus4",  "keys":[2,7,9],     "fingers_rh":{1:0,4:1,5:2},"diff":2,"group":"Sus",  "tip":"D–G–A: Rock/Pop"},
        "Gsus4":  {"name":"Gsus4",  "keys":[7,0,2],     "fingers_rh":{1:0,4:1,5:2},"diff":2,"group":"Sus",  "tip":"G–C–D: Rock"},
        "Asus4":  {"name":"Asus4",  "keys":[9,2,4],     "fingers_rh":{1:0,4:1,5:2},"diff":2,"group":"Sus",  "tip":"A–D–E: Leitet zu A"},
        "Cmaj9":  {"name":"Cmaj9",  "keys":[0,4,7,11,2],"fingers_rh":{1:0,2:1,3:2,4:3,5:4},"diff":3,"group":"Jazz","tip":"C–E–G–H–D: Luxuriös"},
        "Am9":    {"name":"Am9",    "keys":[9,0,4,7,11],"fingers_rh":{1:0,2:1,3:2,4:3,5:4},"diff":3,"group":"Jazz","tip":"A–C–E–G–H: Sehr weich"},
        "Dm9":    {"name":"Dm9",    "keys":[2,5,9,0,4], "fingers_rh":{1:0,2:1,3:2,4:3,5:4},"diff":3,"group":"Jazz","tip":"D–F–A–C–E: Jazz-Standard"},
        "G13":    {"name":"G13",    "keys":[7,11,2,5,9],"fingers_rh":{1:0,2:1,3:2,4:3,5:4},"diff":3,"group":"Jazz","tip":"G–H–D–F–A: Reicher Klang"},
        "E7b9":   {"name":"E7b9",   "keys":[4,8,11,2,5],"fingers_rh":{1:0,2:1,3:2,4:3,5:4},"diff":3,"group":"Jazz","tip":"E–G#–H–D–F: Dissonant"},
    },
}

LESSONS: Dict[str,List[List[str]]] = {
    "guitar": [
        ["Em","Am","E"],
        ["D","C","G","A","E5","A5"],
        ["Dm","Bm","E7","A7","G7","D7"],
        ["Asus2","Asus4","Dsus4","Cadd9","Cmaj7","Amaj7"],
        ["F","Bb","F#m","Gm","Am9","E7b9"],
    ],
    "piano": [
        ["C-Dur","A-Moll","G-Dur"],
        ["F-Dur","D-Moll","E-Moll","E-Dur"],
        ["G7","C7","D7","Am7","Em7"],
        ["Cmaj7","Fmaj7","Gmaj7","Adim","Csus4"],
        ["Dm9","Am9","Cmaj9","G13","E7b9"],
    ],
}

TIPS: Dict[str,List[str]] = {
    "guitar": [
        "Nagel kurz halten für saubereres Greifen.",
        "Fingerkuppen senkrecht auf die Saiten – nicht flach!",
        "Daumen hinter dem Hals, nicht darüber.",
        "Nach 20 Min Üben: 5 Min Pause machen.",
        "Langsam üben → schnell spielen.",
        "Gitarre täglich stimmen – das Ohr trainiert mit!",
        "Power-Chords: Zeigefinger + Ringfinger genügen.",
        "Metronom ab Tag 1 benutzen – Timing ist alles.",
        "Bei Barré: Finger nah am Bund, nicht auf dem Bund.",
        "Handgelenk locker – Verkrampfung kostet Klang.",
    ],
    "piano": [
        "Handgelenke auf Tastatur-Höhe – nicht hängen.",
        "Finger leicht gebogen, wie eine Kugel haltend.",
        "Jede Hand separat üben, dann zusammen.",
        "Legato: Taste halten bis zur nächsten.",
        "Das Metronom ist dein bester Freund!",
        "Pianissimo üben – Kontrolle vor Geschwindigkeit.",
        "Finger 1 = Daumen. Fingersatz immer gleich.",
        "Schwarze Tasten mit leicht gebogenen Fingern.",
        "Pedal erst benutzen wenn Noten sauber klingen.",
        "Beide Hände gleichzeitig sehr langsam üben.",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  SONG-MODUS – Lied-Bibliothek mit Tabulatur-Timelines
# ══════════════════════════════════════════════════════════════════════════════
# Format: beats_per_bar, bars pro Akkord, dann Liste von (akkord, bars, strumPattern)
# strumPattern: "DDUUDU" D=Down U=Up -=Pause

SONGS: Dict[str, Dict] = {
    "guitar": {
        "Knockin' on Heaven's Door": {
            "bpm": 68, "key": "G", "diff": 1, "bars_per_chord": 4,
            "timeline": [
                ("G",  4, "DDUUDU"), ("D",  4, "DDUUDU"),
                ("Am", 4, "DDUUDU"), ("Am", 4, "DDUUDU"),
                ("G",  4, "DDUUDU"), ("D",  4, "DDUUDU"),
                ("C",  4, "DDUUDU"), ("C",  4, "DDUUDU"),
            ],
            "tip": "Bob Dylan – G D Am Akkord-Wechsel üben",
        },
        "House of the Rising Sun": {
            "bpm": 76, "key": "Am", "diff": 2, "bars_per_chord": 4,
            "timeline": [
                ("Am", 4, "D-D-D-"), ("C",  4, "D-D-D-"),
                ("D",  4, "D-D-D-"), ("F",  4, "D-D-D-"),
                ("Am", 4, "D-D-D-"), ("C",  4, "D-D-D-"),
                ("E7", 4, "D-D-D-"), ("E7", 4, "D-D-D-"),
            ],
            "tip": "The Animals – Fingerpicking-Muster in 6/8",
        },
        "Wonderwall": {
            "bpm": 87, "key": "Em", "diff": 2, "bars_per_chord": 2,
            "timeline": [
                ("Em7", 4, "DD-UDU"), ("G",    4, "DD-UDU"),
                ("Dsus4",4,"DD-UDU"), ("A7",   4, "DD-UDU"),
                ("Em7", 4, "DD-UDU"), ("G",    4, "DD-UDU"),
                ("Dsus4",4,"DD-UDU"), ("A7",   4, "DD-UDU"),
                ("C",   4, "DD-UDU"), ("G",    4, "DD-UDU"),
                ("Dsus4",4,"DD-UDU"), ("Em",   4, "DD-UDU"),
            ],
            "tip": "Oasis – Capo 2, DD-UDU Strumming-Pattern",
        },
        "Sweet Home Chicago (Blues)": {
            "bpm": 92, "key": "E", "diff": 3, "bars_per_chord": 2,
            "timeline": [
                ("E",  8, "D-DU-U"), ("A",  4, "D-DU-U"),
                ("E",  4, "D-DU-U"), ("B7", 4, "D-DU-U"),
                ("A",  4, "D-DU-U"), ("E",  4, "D-DU-U"),
            ],
            "tip": "12-Bar Blues in E – E A B7",
        },
        "Nothing Else Matters (Intro)": {
            "bpm": 69, "key": "Em", "diff": 2, "bars_per_chord": 8,
            "timeline": [
                ("Em", 8, "D-----"), ("Em", 8, "D-----"),
                ("Am", 8, "D-----"), ("C",  8, "D-----"),
            ],
            "tip": "Metallica – Fingerpicking, langsam und sauber",
        },
        "Let Her Go": {
            "bpm": 100, "key": "C", "diff": 2, "bars_per_chord": 2,
            "timeline": [
                ("C",  4, "DDUUDU"), ("G",  4, "DDUUDU"),
                ("Am", 4, "DDUUDU"), ("F",  4, "DDUUDU"),
                ("C",  4, "DDUUDU"), ("G",  4, "DDUUDU"),
                ("Dm", 4, "DDUUDU"), ("F",  4, "DDUUDU"),
            ],
            "tip": "Passenger – C G Am F Grundprogression",
        },

        # ── Spanish / Flamenco Songs (freigeschaltet ab Level 3) ─────────────
        "Malagueña (Flamenco)": {
            "bpm": 80, "key": "Am", "diff": 3, "bars_per_chord": 2,
            "category": "Spanish", "min_level": 3,
            "timeline": [
                ("Am_Flam", 2, "DU-DU"), ("G_Flam",  2, "DU-DU"),
                ("F_Maj",   2, "DU-DU"), ("E_Phryg", 4, "D-DU-D"),
                ("Am_Flam", 2, "DU-DU"), ("G_Flam",  2, "DU-DU"),
                ("F_Maj",   2, "DU-DU"), ("E_Phryg", 4, "D---D-"),
            ],
            "tip": "Malagueña – A-Phrygisch, der Herzschlag des Flamenco",
        },
        "Oye Como Va (Santana)": {
            "bpm": 126, "key": "Am", "diff": 3, "bars_per_chord": 4,
            "category": "Latin", "min_level": 3,
            "timeline": [
                ("Am7_Lat", 4, "DDUUDU"), ("D9",      4, "DDUUDU"),
                ("Am7_Lat", 4, "DDUUDU"), ("D9",      4, "DDUUDU"),
                ("Am7_Lat", 4, "DDUUDU"), ("G7_Lat",  4, "DDUUDU"),
                ("Am7_Lat", 4, "DDUUDU"), ("E7_Span", 4, "D-DU-D"),
            ],
            "tip": "Santana – Am-Dorisch, der Latin-Rock Klassiker",
        },
        "La Bamba (Traditional)": {
            "bpm": 92, "key": "G", "diff": 2, "bars_per_chord": 2,
            "category": "Latin", "min_level": 3,
            "timeline": [
                ("G", 2, "DDUUDU"), ("C",  2, "DDUUDU"), ("D", 2, "DDUUDU"), ("G", 2, "DDUUDU"),
                ("G", 2, "DDUUDU"), ("C",  2, "DDUUDU"), ("D", 2, "D-D-D-"), ("G", 2, "DDUUDU"),
            ],
            "tip": "La Bamba – G C D, mexikanischer Folk-Rock",
        },
        "Besame Mucho (Bossa)": {
            "bpm": 68, "key": "Dm", "diff": 3, "bars_per_chord": 4,
            "category": "Latin", "min_level": 4,
            "timeline": [
                ("Dm_Flam", 4, "D-UDU-"), ("G7_Lat",  4, "D-UDU-"),
                ("Am7_Lat", 4, "D-UDU-"), ("E7_Span", 4, "D-D-D-"),
                ("Am_Flam", 4, "D-UDU-"), ("E7_Span", 4, "D-UDU-"),
                ("Dm_Flam", 4, "D-UDU-"), ("Am_Flam", 4, "D---D-"),
            ],
            "tip": "Besame Mucho – Bolero-Bossa, romantischer Klassiker",
        },
        "Guantanamera (Son Cubano)": {
            "bpm": 100, "key": "G", "diff": 2, "bars_per_chord": 2,
            "category": "Latin", "min_level": 3,
            "timeline": [
                ("G",  2, "D-UDU-"), ("C",  2, "D-UDU-"), ("D", 4, "D-DUDU"),
                ("G",  2, "D-UDU-"), ("Am7_Lat", 2, "D-UDU-"), ("D", 2, "D-DU-U"), ("G", 2, "D---D-"),
            ],
            "tip": "Guantanamera – kubanischer Son, mitreißendes Strumming",
        },
        "Concierto de Aranjuez (Klassisch)": {
            "bpm": 56, "key": "Am", "diff": 3, "bars_per_chord": 4,
            "category": "Spanish", "min_level": 4,
            "timeline": [
                ("Am_Flam", 4, "D-U-D-"), ("E7_Span", 4, "D-U-D-"),
                ("Am_Flam", 4, "D-U-D-"), ("Dm_Flam", 4, "D-U-D-"),
                ("G_Flam",  4, "D-U-D-"), ("E7_Span", 4, "D-U-D-"),
                ("Am_Flam", 4, "D-U-DU"), ("E_Phryg", 4, "D-----"),
            ],
            "tip": "Aranjuez – klassisches Thema, Fingerpicking Arpeggio",
        },
    },
    "piano": {
        "Für Elise (Vereinfacht)": {
            "bpm": 60, "key": "Am", "diff": 1, "bars_per_chord": 4,
            "timeline": [
                ("A-Moll",4,"---"), ("E-Dur",4,"---"),
                ("A-Moll",4,"---"), ("C-Dur",4,"---"),
                ("G-Dur", 4,"---"), ("A-Moll",4,"---"),
            ],
            "tip": "Beethoven – sehr langsam beginnen",
        },
        "Let It Be": {
            "bpm": 72, "key": "C", "diff": 1, "bars_per_chord": 4,
            "timeline": [
                ("C-Dur",4,"---"), ("G-Dur",4,"---"),
                ("A-Moll",4,"---"), ("F-Dur",4,"---"),
                ("C-Dur",4,"---"), ("G-Dur",4,"---"),
                ("F-Dur",4,"---"), ("C-Dur",4,"---"),
            ],
            "tip": "Beatles – C G Am F Classic",
        },
        "Clocks (Coldplay)": {
            "bpm": 130, "key": "Eb", "diff": 2, "bars_per_chord": 4,
            "timeline": [
                ("E-Moll",4,"---"), ("D-Dur",4,"---"),
                ("A-Moll",8,"---"), ("E-Moll",4,"---"),
                ("D-Dur",4,"---"),  ("A-Moll",8,"---"),
            ],
            "tip": "Coldplay – Arpeggio-Muster, Tempo aufbauen",
        },
        "Bohemian Rhapsody (Ballade)": {
            "bpm": 66, "key": "Bb", "diff": 3, "bars_per_chord": 2,
            "timeline": [
                ("G-Dur",4,"---"),  ("G7",4,"---"),
                ("C-Dur",4,"---"),  ("G-Dur",4,"---"),
                ("A-Moll",4,"---"), ("D7",4,"---"),
                ("G-Dur",4,"---"),  ("G7",4,"---"),
            ],
            "tip": "Queen – Langsam, Dynamik beachten",
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO-TUTORIAL-DATEN  (SVG-animiert, kein echter Download nötig)
# ══════════════════════════════════════════════════════════════════════════════
# Jedes Tutorial enthält: title, trigger_condition, steps (text + SVG-Hint)

TUTORIALS: Dict[str, Dict] = {
    "wrist": {
        "id": "wrist", "icon": "🖐",
        "title": {"de":"Handgelenk-Haltung","en":"Wrist Posture"},
        "trigger": "posture_bad",
        "steps": [
            {"text":{"de":"Das Handgelenk sollte UNTER dem Hals bleiben – nicht oben drüber!","en":"Keep wrist BELOW the neck, not above!"},
             "svg_hint":"wrist_correct"},
            {"text":{"de":"Stell dir vor, du hältst einen Apfel unter dem Hals.","en":"Imagine holding an apple under the neck."},
             "svg_hint":"wrist_grip"},
            {"text":{"de":"Entspann den Arm – Verkrampfung ist der größte Feind.","en":"Relax your arm – tension is the enemy."},
             "svg_hint":"wrist_relax"},
        ],
    },
    "barre": {
        "id": "barre", "icon": "🎸",
        "title": {"de":"Barré-Technik","en":"Barré Technique"},
        "trigger": "barre_chord",
        "steps": [
            {"text":{"de":"Zeigefinger flach über ALLE Saiten – nah am Bund, nicht darauf.","en":"Index finger flat across ALL strings, close to fret."},
             "svg_hint":"barre_position"},
            {"text":{"de":"Drück mit dem Daumen HINTER den Hals für mehr Kraft.","en":"Press thumb BEHIND neck for more power."},
             "svg_hint":"barre_thumb"},
            {"text":{"de":"Prüfe jede Saite einzeln – eine gedämpfte Saite verrät den Fehler.","en":"Check each string – muted = wrong position."},
             "svg_hint":"barre_check"},
        ],
    },
    "strumming": {
        "id": "strumming", "icon": "🎵",
        "title": {"de":"Anschlag-Technik","en":"Strumming Technique"},
        "trigger": "strum_wrong",
        "steps": [
            {"text":{"de":"Das Plektrum schräg halten – 45° zur Saite.","en":"Hold pick at 45° to the strings."},
             "svg_hint":"pick_angle"},
            {"text":{"de":"Abschlag (↓): Aus dem Handgelenk, nicht vom Ellenbogen.","en":"Down-stroke: wrist motion, not elbow."},
             "svg_hint":"strum_down"},
            {"text":{"de":"Aufschlag (↑): Leichter Kontakt – nicht so viel Druck.","en":"Up-stroke: lighter contact, less pressure."},
             "svg_hint":"strum_up"},
            {"text":{"de":"Halte den Rhythmus mit dem Fuß – tipp im Takt.","en":"Keep rhythm with your foot – tap the beat."},
             "svg_hint":"strum_rhythm"},
        ],
    },
    "fingertips": {
        "id": "fingertips", "icon": "👆",
        "title": {"de":"Fingerkuppen-Technik","en":"Fingertip Technique"},
        "trigger": "accuracy_low",
        "steps": [
            {"text":{"de":"Greife mit der KUPPE, nicht mit der Fläche – Finger senkrecht!","en":"Use FINGERTIP, not finger pad – vertical fingers!"},
             "svg_hint":"fingertip_correct"},
            {"text":{"de":"Nagel kurz schneiden – langer Nagel = flacher Griff.","en":"Keep nails short – long nail = flat finger."},
             "svg_hint":"fingertip_nail"},
            {"text":{"de":"Finger direkt hinter den Bund, nicht auf den Bund.","en":"Finger just behind fret, not on top of it."},
             "svg_hint":"fingertip_fret"},
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATENMODELLE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Profile:
    name:        str
    instrument:  str         = "guitar"
    level:       int         = 1
    xp:          int         = 0
    total_min:   float       = 0.0
    acc_history: List[float] = field(default_factory=list)
    mastered:    List[str]   = field(default_factory=list)
    sessions:    List[Dict]  = field(default_factory=list)
    lesson_idx:  int         = 0
    best_streak: int         = 0
    created:     str         = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def xp_next(self): return self.level * 120
    def add_xp(self, n):
        self.xp += n; lv = False
        while self.xp >= self.xp_next():
            self.xp -= self.xp_next(); self.level += 1; lv = True
        return lv
    def avg_acc(self):
        return float(np.mean(self.acc_history[-20:])) if self.acc_history else 0.

@dataclass
class Msg:
    text:     str
    color:    tuple
    born:     float = field(default_factory=time.time)
    duration: float = 3.5
    bold:     bool  = False
    def alive(self): return time.time() - self.born < self.duration
    def alpha(self):
        t = time.time() - self.born; s = self.duration * 0.65
        return 1.0 if t < s else max(0.0, 1.0 - (t - s) / (self.duration - s))

# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class ProgressMgr:
    def __init__(self):
        self.profiles: Dict[str,Profile] = {}; self._load()

    def _load(self):
        if not Path(PROGRESS_FILE).exists(): return
        try:
            for n, d in json.loads(Path(PROGRESS_FILE).read_text("utf-8")).items():
                for k in ["lesson_idx","best_streak"]: d.setdefault(k,0)
                for old,new in [("total_minutes","total_min"),("accuracy_history","acc_history"),
                                 ("chords_mastered","mastered"),("created_at","created")]:
                    if old in d: d[new] = d.pop(old)
                valid = set(Profile.__dataclass_fields__)
                d = {k: v for k,v in d.items() if k in valid}
                self.profiles[n] = Profile(**d)
        except Exception as e: print(f"[Progress] {e}")

    def save(self):
        Path(PROGRESS_FILE).write_text(
            json.dumps({n: asdict(p) for n,p in self.profiles.items()},
                       ensure_ascii=False, indent=2), "utf-8")

    def get(self, name, instr="guitar"):
        if name not in self.profiles:
            self.profiles[name] = Profile(name=name, instrument=instr); self.save()
        return self.profiles[name]

    def record(self, p, acc, mins, chords):
        p.sessions.append({"date":datetime.datetime.now().isoformat(),
            "accuracy":round(acc,1),"minutes":round(mins,1),"chords":chords})
        p.acc_history.append(round(acc,1)); p.total_min += mins
        p.add_xp(int(acc/10+mins*4)); self.save()

# ══════════════════════════════════════════════════════════════════════════════
#  POLYPHONER AUDIO-ANALYZER  (NEU v5)
# ══════════════════════════════════════════════════════════════════════════════
class PolyphonicAnalyzer:
    """
    Erkennt mehrere gleichzeitige Töne (Polyphone Analyse) via:
    - HPS (Harmonic Product Spectrum): für jeden der 6 Saiten-Frequenzbereiche
    - Iteratives Peak-Picking über das gesamte Spektrum
    - String-Präsenz-Map: Welche Saite klingt, welche ist gedämpft?
    - Akkord-Vollständigkeits-Score: Sind alle erwarteten Töne da?
    """
    # Frequenzbänder pro Saite (mit Toleranz ±20%)
    STRING_BANDS = [(int(f*0.8), int(f*1.2)) for f in GUITAR_STRINGS_HZ]
    # Note-Klassen der offenen Saiten (E2=E, A2=A, D3=D, G3=G, H3=H, e4=E)
    OPEN_STRING_NOTES = ["E","A","D","G","H","E"]

    def __init__(self):
        self.string_presence: List[float] = [0.]*6   # 0-1 pro Saite
        self.string_notes:    List[str]   = ["–"]*6  # Erkannte Note
        self.poly_notes:      List[str]   = []        # Alle erkannten Noten
        self.chord_score:     float       = 0.0       # Akkord-Vollständigkeit
        self.muted_strings:   List[int]   = []        # Saiten die gedämpft klingen
        self._spec_smooth     = np.zeros(2048)

    def analyze(self, fft: np.ndarray, freqs: np.ndarray, target_chord: Dict) -> Dict:
        """
        Analysiert FFT gegen Ziel-Akkord.
        Gibt zurück: {string_presence, poly_notes, chord_score, muted_strings}
        """
        if len(fft) < 100 or freqs is None or len(freqs) != len(fft):
            return self._empty_state()

        # Sanfte Spektrum-Glättung
        fft_n = fft / (np.max(fft) + 1e-9)
        if len(fft_n) == len(self._spec_smooth):
            self._spec_smooth = self._spec_smooth * 0.6 + fft_n * 0.4
        else:
            self._spec_smooth = fft_n.copy()

        # ── Saiten-Präsenz: prüfe Energie im Frequenzband jeder Saite ───────
        presence = []
        notes_found = []
        for si, (f_lo, f_hi) in enumerate(self.STRING_BANDS):
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if mask.any():
                band_energy = float(np.mean(fft_n[mask]))
                presence.append(min(1.0, band_energy * 12))
                # Feinere Noten-Erkennung: Peak im Band
                band_fft = fft_n.copy(); band_fft[~mask] = 0
                peak_idx = int(np.argmax(band_fft))
                if freqs[peak_idx] > 0 and band_energy > 0.02:
                    midi = round(69 + 12 * math.log2(freqs[peak_idx] / 440.0))
                    note = NOTE_NAMES[midi % 12]
                    notes_found.append(note)
                    self.string_notes[si] = note
                else:
                    notes_found.append("–")
                    self.string_notes[si] = "–"
            else:
                presence.append(0.0); notes_found.append("–")

        # ── HPS (Harmonic Product Spectrum) für Gesamtspektrum ──────────────
        hps = fft_n.copy()
        for h in range(2, 5):
            ds = fft_n[::h]
            hps[:len(ds)] *= ds

        # Peak-Picking im HPS (bis 6 gleichzeitige Töne)
        poly = []
        hps_work = hps.copy()
        for _ in range(6):
            idx = int(np.argmax(hps_work))
            if hps_work[idx] < 0.01: break
            f = freqs[idx] if idx < len(freqs) else 0
            if 60 < f < 4000:
                midi = round(69 + 12 * math.log2(f / 440.0))
                note = NOTE_NAMES[midi % 12]
                if note not in poly: poly.append(note)
            # Lösche Peak + Umgebung
            w = max(1, int(idx * 0.08))
            hps_work[max(0, idx-w):idx+w+1] = 0
        self.poly_notes = poly

        # ── Akkord-Vollständigkeits-Score ───────────────────────────────────
        tgt_keys = target_chord.get("keys", [])
        if tgt_keys:  # Piano: Vergleich über Halbtöne
            expected = set(NOTE_NAMES[k % 12] for k in tgt_keys)
        else:          # Gitarre: aus Fingerposition schätzen
            expected = self._guitar_chord_notes(target_chord)
        if expected:
            found = set(poly) | set(n for n in notes_found if n != "–")
            hits = len(found & expected)
            self.chord_score = min(1.0, hits / len(expected))
        else:
            self.chord_score = float(len(poly) > 1)

        # ── Gedämpfte Saiten (erwartet klingend, aber leise) ────────────────
        target_strings = set(target_chord.get("open", []) +
                             [v[1] for v in target_chord.get("fingers", {}).values()])
        self.muted_strings = [si+1 for si, p in enumerate(presence)
                               if p < 0.12 and (si+1) in target_strings]
        self.string_presence = presence
        return self.get_state()

    def _guitar_chord_notes(self, chord: Dict) -> set:
        """Schätze erwartete Noten aus Gitarren-Fingersatz."""
        # Saiten-Grundton-MIDI (EADGHE)
        open_midi = [40, 45, 50, 55, 59, 64]
        notes = set()
        # Offene Saiten
        for s in chord.get("open", []):
            if 1 <= s <= 6: notes.add(NOTE_NAMES[(open_midi[s-1]) % 12])
        # Gegriffene Saiten
        for fid, (fret, string) in chord.get("fingers", {}).items():
            if 1 <= string <= 6:
                notes.add(NOTE_NAMES[(open_midi[string-1] + fret) % 12])
        return notes

    def _empty_state(self):
        self.string_presence = [0.]*6; self.string_notes = ["–"]*6
        self.poly_notes = []; self.chord_score = 0.0; self.muted_strings = []
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "string_presence": [round(p, 3) for p in self.string_presence],
            "string_notes":    self.string_notes,
            "poly_notes":      self.poly_notes,
            "chord_score":     round(self.chord_score, 3),
            "muted_strings":   self.muted_strings,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STRUMMING DETEKTOR  (NEU v5)
# ══════════════════════════════════════════════════════════════════════════════
class StrummingDetector:
    """
    Erkennt Strumming-Events in Echtzeit:
    - Richtung: Abschlag (↓) oder Aufschlag (↑) via Spektral-Schwerpunkt-Bewegung
    - Timing: ms zwischen Schlägen, gemessenes BPM
    - Pattern-Erkennung: Vergleich mit Ziel-Pattern (z.B. "DDUUDU")
    - Dynamik: Anschlagsstärke (pp, mp, mf, f, ff)
    """
    DIR_DOWN  = "↓"
    DIR_UP    = "↑"
    DIR_NONE  = "–"
    DYNAMICS  = ["pp","mp","mf","f","ff"]

    def __init__(self):
        self._onset_times:   collections.deque = collections.deque(maxlen=32)
        self._onset_dirs:    collections.deque = collections.deque(maxlen=32)
        self._onset_dynams:  collections.deque = collections.deque(maxlen=32)
        self._prev_centroid: float = 0.0
        self._prev_rms:      float = 0.0
        self._onset_cd:      float = 0.0
        self._pattern_buf:   List[str] = []
        self._target_pattern: str = ""
        self._pattern_score:  float = 0.0

        # Public outputs
        self.last_dir:     str   = self.DIR_NONE
        self.last_dynamic: str   = "–"
        self.measured_bpm: float = 0.0
        self.pattern_live: str   = ""        # Was wurde gespielt
        self.pattern_match:float = 0.0       # Übereinstimmung 0-1
        self.onset_flash:  bool  = False
        self._flash_t:     float = 0.

    def update(self, fft: np.ndarray, freqs: np.ndarray, rms: float,
               target_pattern: str = "", bpm: float = 80.0):
        """Jeden Audio-Frame aufrufen."""
        now = time.time()
        self.onset_flash = now - self._flash_t < 0.06

        if freqs is None or len(fft) != len(freqs) or len(fft) < 10:
            return

        fft_n = fft / (np.max(fft) + 1e-9)

        # ── Spektral-Schwerpunkt (Centroid) für Richtungserkennung ───────────
        centroid = float(np.sum(freqs * fft_n) / (np.sum(fft_n) + 1e-9))

        # ── Onset-Detection via RMS-Anstieg ─────────────────────────────────
        drms = rms - self._prev_rms
        is_onset = (drms > 0.018 and rms > SILENCE_THRESH * 2
                    and now - self._onset_cd > 0.06)

        if is_onset:
            self._onset_cd = now; self._flash_t = now; self.onset_flash = True
            # Richtung: Abschlag → Centroid steigt (Tiefere Frequenzen zuerst)
            #           Aufschlag → Centroid sinkt (Höhere Frequenzen zuerst)
            centroid_delta = centroid - self._prev_centroid
            direction = self.DIR_DOWN if centroid_delta < -30 else self.DIR_UP

            # Dynamik: RMS → Dynamikstufe
            dyn_idx = min(4, int(rms / 0.06 * 5))
            dynamic = self.DYNAMICS[dyn_idx]

            self.last_dir = direction; self.last_dynamic = dynamic
            self._onset_times.append(now)
            self._onset_dirs.append(direction)
            self._onset_dynams.append(dynamic)
            self._pattern_buf.append(direction)

        self._prev_centroid = centroid * 0.7 + self._prev_centroid * 0.3
        self._prev_rms      = rms * 0.7 + self._prev_rms * 0.3

        # ── Gemessenes BPM aus Onset-Abständen ──────────────────────────────
        if len(self._onset_times) >= 4:
            times = list(self._onset_times)[-8:]
            diffs = [times[i+1]-times[i] for i in range(len(times)-1)]
            med_diff = float(np.median(diffs))
            if 0.1 < med_diff < 2.0:
                self.measured_bpm = round(60.0 / med_diff, 1)

        # ── Pattern-Matching: Vergleich mit Ziel ────────────────────────────
        self._target_pattern = target_pattern
        if target_pattern and len(self._pattern_buf) >= len(target_pattern):
            live = self._pattern_buf[-len(target_pattern):]
            score = 0
            for i, c in enumerate(target_pattern):
                if c == "D" and live[i] == self.DIR_DOWN: score += 1
                elif c == "U" and live[i] == self.DIR_UP:  score += 1
                elif c == "-": score += 1  # Pause immer als OK
            self.pattern_match = score / max(len(target_pattern), 1)
        self.pattern_live = "".join(
            ("D" if d==self.DIR_DOWN else "U") for d in list(self._onset_dirs)[-8:])

    def get_state(self) -> Dict:
        return {
            "last_dir":      self.last_dir,
            "last_dynamic":  self.last_dynamic,
            "measured_bpm":  self.measured_bpm,
            "pattern_live":  self.pattern_live,
            "pattern_match": round(self.pattern_match, 2),
            "onset_flash":   self.onset_flash,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYZER v5  (YIN + Polyphon + Strumming)
# ══════════════════════════════════════════════════════════════════════════════
class AudioAnalyzer:
    def __init__(self):
        self._pa = None; self._stream = None
        self._buf = collections.deque(maxlen=PITCH_BUF_SIZE)
        self._wave = collections.deque(maxlen=240)
        self.active = False
        self.error:      Optional[str] = None   # Fehlermeldung
        self.error_code: Optional[str] = None   # Maschinenlesbarer Code

        # Public outputs
        self.rms:          float       = 0.0
        self.pitch_hz:     float       = 0.0
        self.pitch_midi:   int         = 0
        self.note_name:    str         = "–"
        self.cents_off:    float       = 0.0
        self.spectrum:     List[float] = [0.]*240
        self.wave_hist:    List[float] = [0.]*240
        self.onset:        bool        = False
        self.bass_rms:     float       = 0.0
        self.mid_rms:      float       = 0.0
        self.high_rms:     float       = 0.0
        self.chord_match:  str         = ""
        self.chord_conf:   float       = 0.0
        self.detected_notes: List[str] = []
        self.is_silent:    bool        = True

        self._prev_rms   = 0.0
        self._onset_cd   = 0.0
        self._lock       = threading.Lock()
        self._raw_buf    = np.zeros(AUDIO_CHUNK * 4, dtype=np.float32)
        self._fft_cache  = np.zeros(AUDIO_CHUNK * 2)
        self._freqs_cache= np.zeros(AUDIO_CHUNK * 2)

        # ── Neue Analyse-Ausgaben ──────────────────────────────────────────
        self.attack_detected:  bool  = False   # Harter Anschlag (Transient)
        self.cleanliness:      float = 1.0     # 0=schnarren, 1=sauber
        self.buzz_detected:    bool  = False   # Schnarr-Erkennung
        self.harmonic_ratio:   float = 0.0     # Verhältnis Grund-/Oberton-Energie
        self.transient_rms:    float = 0.0     # RMS im Moment des Anschlags

        self._attack_history:  collections.deque = collections.deque(maxlen=8)
        self._harm_history:    collections.deque = collections.deque(maxlen=16)
        self._buzz_history:    collections.deque = collections.deque(maxlen=12)
        self._prev_fft:        np.ndarray = np.zeros(AUDIO_CHUNK * 2)

        # Sub-Analyzer
        self.poly    = PolyphonicAnalyzer()
        self.strum   = StrummingDetector()

        self._try_start()

    def _try_start(self):
        """Startet Mikrofon-Stream mit detailliertem Error-Handling."""
        if not PYAUDIO_OK:
            self.error = "PyAudio nicht installiert (pip install pyaudio)"
            self.error_code = "NO_PYAUDIO"
            print(f"[Audio] ⚠ {self.error} → Simulation aktiv")
            return

        # PyAudio-Geräte prüfen
        try:
            pa_tmp = _pa.PyAudio()
            n_in = sum(
                1 for i in range(pa_tmp.get_device_count())
                if pa_tmp.get_device_info_by_index(i).get('maxInputChannels', 0) > 0
            )
            pa_tmp.terminate()
            if n_in == 0:
                self.error = "Kein Mikrofon gefunden – bitte Mikrofon anschließen"
                self.error_code = "NO_MIC"
                print(f"[Audio] ⚠ {self.error}")
                return
        except Exception as e:
            self.error = f"PyAudio Initialisierungsfehler: {e}"
            self.error_code = "PYAUDIO_INIT_ERROR"
            print(f"[Audio] ⚠ {self.error}")
            return

        try:
            self._pa = _pa.PyAudio()
            self._stream = self._pa.open(
                format=_pa.paFloat32, channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE, input=True,
                frames_per_buffer=AUDIO_CHUNK,
                stream_callback=self._callback)
            self._stream.start_stream()
            self.active = True
            self.error = None
            self.error_code = None
            print(f"[Audio] ✓ Mikrofon @ {AUDIO_RATE}Hz")
        except OSError as e:
            if "Invalid sample rate" in str(e):
                self.error = f"Sampling-Rate {AUDIO_RATE}Hz nicht unterstützt"
                self.error_code = "BAD_SAMPLERATE"
            elif "No Default Input Device" in str(e):
                self.error = "Kein Standard-Mikrofon – bitte in Systemeinstellungen setzen"
                self.error_code = "NO_DEFAULT_MIC"
            else:
                self.error = f"Mikrofon-Fehler: {e}"
                self.error_code = "MIC_ERROR"
            print(f"[Audio] ⚠ {self.error} → Simulation")
        except Exception as e:
            self.error = f"Unbekannter Audio-Fehler: {e}"
            self.error_code = "AUDIO_UNKNOWN"
            print(f"[Audio] ⚠ {self.error} → Simulation")

    def _callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.float32).copy()
        with self._lock:
            n = len(samples)
            self._raw_buf = np.roll(self._raw_buf, -n)
            self._raw_buf[-n:] = samples
        return (None, _pa.paContinue)

    def update(self, target_chord: Dict = None, target_pattern: str = "",
               metro_bpm: float = 80.0):
        now = time.time()
        if self.active:
            with self._lock: data = self._raw_buf[-AUDIO_CHUNK*2:].copy()
        else:
            t = now; freq = 220.0
            data = (0.4*np.sin(2*np.pi*freq*np.linspace(t,t+AUDIO_CHUNK*2/AUDIO_RATE,AUDIO_CHUNK*2))
                    +0.2*np.sin(2*np.pi*freq*2*np.linspace(t,t+AUDIO_CHUNK*2/AUDIO_RATE,AUDIO_CHUNK*2))
                    +0.05*np.random.randn(AUDIO_CHUNK*2)).astype(np.float32)

        # ── RMS & Stille ────────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(data**2)))
        self.rms = min(1.0, rms/0.3)
        self.is_silent = rms < SILENCE_THRESH

        # ── Attack/Transient Detection (verschärft) ──────────────────────────
        # Klassischer Onset via RMS-Delta
        delta = rms - self._prev_rms
        basic_onset = (delta > 0.012 and now - self._onset_cd > 0.04 and not self.is_silent)

        # NEUER: Spectral Flux – wie stark ändert sich das FFT-Spektrum?
        # Hoher Flux = Transient (Anschlag), niedriger Flux = Sustain/Rauschen
        # Erst nach erster FFT (wird unten berechnet – wir nutzen den Cache)
        win = np.hanning(len(data))
        fft_raw = np.abs(np.fft.rfft(data * win))
        freqs   = np.fft.rfftfreq(len(data), 1.0 / AUDIO_RATE)
        fft_max = np.max(fft_raw) + 1e-9
        fft_norm = fft_raw / fft_max
        self._fft_cache = fft_norm; self._freqs_cache = freqs

        # Spectral Flux: positiver Anteil der FFT-Änderung (Anschlag-Energie)
        spec_flux = float(np.mean(np.maximum(0, fft_norm[:128] - self._prev_fft[:128])))
        self._prev_fft = fft_norm.copy()

        # Attack = starker Flux ODER hoher RMS-Delta
        hard_attack = spec_flux > 0.035 and not self.is_silent
        self.attack_detected = (basic_onset or hard_attack) and now - self._onset_cd > 0.06
        self.onset = self.attack_detected  # Rückwärtskompatibilität

        if self.attack_detected:
            self._onset_cd = now
            self.transient_rms = rms
            self._attack_history.append(rms)

        self._prev_rms = rms * 0.65 + self._prev_rms * 0.35

        # ── FFT (bereits in Attack-Detection berechnet) ─────────────────────
        fft = fft_raw  # Alias – oben schon berechnet

        # Spektrum-Bins für Visualisierung
        bins = np.logspace(np.log10(50), np.log10(8000), 240)
        spec = np.clip(np.interp(bins, freqs, fft_norm), 0, 1)
        if len(self.spectrum)==240:
            self.spectrum = list(np.array(self.spectrum)*0.55+spec*0.45)
        else:
            self.spectrum = list(spec)

        # Band-Energie
        def band_rms(f_lo, f_hi):
            mask = (freqs>=f_lo)&(freqs<=f_hi)
            return min(1.0, float(np.sqrt(np.mean(fft[mask]**2)))*8) if mask.any() else 0.0
        self.bass_rms = band_rms(20,300)
        self.mid_rms  = band_rms(300,3000)
        self.high_rms = band_rms(3000,16000)

        self._wave.append(self.rms)
        self.wave_hist = list(self._wave)+[0.]*(240-len(self._wave))

        # ── YIN Pitch ─────────────────────────────────────────────────────
        if not self.is_silent:
            hz = self._yin(data, AUDIO_RATE)
            if hz and 50 < hz < 1400:
                self._buf.append(hz)
                mhz = float(np.median(list(self._buf)))
                self.pitch_hz = mhz
                midi, name, cents = self._hz_to_midi(mhz)
                self.pitch_midi=midi; self.note_name=name; self.cents_off=cents
                self.detected_notes = self._detect_harmonics(fft_norm, freqs, mhz)
                self.chord_match, self.chord_conf = self._match_chord(self.detected_notes)
            else: self._buf.clear()
        else:
            self.pitch_hz=0.; self.note_name="–"; self.cents_off=0.
            self.detected_notes=[]; self.chord_match=""; self.chord_conf=0.

        # ── Polyphoner Analyzer ───────────────────────────────────────────
        self.poly.analyze(fft_norm, freqs, target_chord or {})

        # ── Strumming Detektor ────────────────────────────────────────────
        self.strum.update(fft_norm, freqs, rms, target_pattern, metro_bpm)

        # ══════════════════════════════════════════════════════════════════
        # NEUE AUDIO-INTELLIGENZ: Sauberkeit + Buzz-Detection
        # ══════════════════════════════════════════════════════════════════
        if not self.is_silent and self.pitch_hz > 50:
            self._compute_cleanliness(fft_norm, freqs, self.pitch_hz)
        else:
            self.cleanliness   = 1.0
            self.buzz_detected = False

    def _yin(self, data, sr, f_min=80., f_max=1200., threshold=0.12):
        N=len(data); tau_min=max(1,int(sr/f_max)); tau_max=min(N//2,int(sr/f_min))
        if tau_max<=tau_min: return None
        d=np.zeros(tau_max)
        for tau in range(1,tau_max):
            diff=data[:N-tau]-data[tau:]; d[tau]=np.dot(diff,diff)
        cmnd=np.zeros_like(d); cmnd[0]=1.0; cumsum=0.0
        for tau in range(1,tau_max):
            cumsum+=d[tau]; cmnd[tau]=d[tau]*tau/(cumsum+1e-9)
        tau_est=None
        for tau in range(tau_min,tau_max):
            if cmnd[tau]<threshold:
                while tau+1<tau_max and cmnd[tau+1]<cmnd[tau]: tau+=1
                tau_est=tau; break
        if tau_est is None: tau_est=tau_min+int(np.argmin(cmnd[tau_min:tau_max]))
        if tau_est<1: return None
        if 0<tau_est<tau_max-1:
            s0,s1,s2=cmnd[tau_est-1],cmnd[tau_est],cmnd[tau_est+1]
            denom=2*s1-s0-s2
            if abs(denom)>1e-9: tau_est=tau_est+(s2-s0)/(2*denom)
        return float(sr)/tau_est

    def _compute_cleanliness(self, fft_norm: np.ndarray, freqs: np.ndarray, f0: float):
        """
        Sauberkeits-Faktor: Misst wie "sauber" der Ton klingt.

        Methode:
        1. Harmonic Ratio: Energie der Grundfrequenz + Obertöne vs. Gesamt-Energie
           → niedrig = viel Rauschen/Schnarren
        2. Inter-Harmonic Noise: Energie ZWISCHEN den Obertönen
           → hoch = Schnarren (zusätzliche Frequenzen durch Saiten-Vibration an Bund)
        3. Oberton-Amplituden-Varianz: Schwankung zwischen Messungen
           → hoch = instabile Obertöne = Finger hält nicht fest genug
        """
        harm_energy   = 0.0
        between_energy = 0.0
        harm_peaks     = []

        for h in range(1, 9):  # 8 Obertöne
            f_h = f0 * h
            if f_h > AUDIO_RATE / 2: break

            # Peak um Oberton h
            idx = int(np.argmin(np.abs(freqs - f_h)))
            w   = max(2, int(idx * 0.04))   # 4% Fenster um Oberton
            lo, hi = max(0, idx - w), min(len(fft_norm) - 1, idx + w)
            peak_e = float(np.max(fft_norm[lo:hi]))
            harm_energy += peak_e
            harm_peaks.append(peak_e)

            # Energie ZWISCHEN Obertönen (Schnarr-Indikator)
            if h < 8:
                f_between = f0 * (h + 0.5)
                idx_b = int(np.argmin(np.abs(freqs - f_between)))
                wb = max(2, int(idx_b * 0.03))
                lo_b, hi_b = max(0, idx_b - wb), min(len(fft_norm) - 1, idx_b + wb)
                between_energy += float(np.max(fft_norm[lo_b:hi_b]))

        total_energy = float(np.mean(fft_norm)) * len(fft_norm) + 1e-9

        # Harmonic Ratio: wie viel Energie ist IN den Obertönen?
        self.harmonic_ratio = min(1.0, harm_energy / (between_energy + harm_energy + 1e-9))

        # Buzz-Score: hohe Energie zwischen Obertönen = Schnarren
        buzz_score = between_energy / (harm_energy + 1e-9)
        self._buzz_history.append(buzz_score)
        avg_buzz = float(np.mean(self._buzz_history))
        self.buzz_detected = avg_buzz > 0.45  # Schwelle: 45% Zwischen-Energie

        # Oberton-Varianz (nur bei ausreichend Peaks)
        if len(harm_peaks) >= 3:
            peak_var = float(np.std(harm_peaks[1:4]) / (np.mean(harm_peaks[1:4]) + 1e-9))
        else:
            peak_var = 0.0
        self._harm_history.append(peak_var)

        # Sauberkeits-Score: kombiniert Harmonic Ratio + Buzz-Abwesenheit + Stabilität
        harm_score    = self.harmonic_ratio              # hoch = gut
        buzz_penalty  = min(1.0, avg_buzz * 1.8)         # hoch = schlecht
        var_penalty   = min(1.0, float(np.mean(self._harm_history)) * 2.0)  # schlecht

        raw_clean = harm_score * (1.0 - buzz_penalty * 0.6) * (1.0 - var_penalty * 0.4)
        # Smooth über Zeit
        self.cleanliness = float(
            np.clip(0.7 * self.cleanliness + 0.3 * raw_clean, 0.0, 1.0)
        )

    def _hz_to_midi(self, hz):
        if hz<=0: return 0,"–",0.
        mf=69+12*math.log2(hz/440.); m=round(mf); c=(mf-m)*100
        return m, f"{NOTE_NAMES[m%12]}{m//12-1}", c

    def _detect_harmonics(self, fft, freqs, f0, n_harmonics=6):
        notes=set()
        for h in range(1, n_harmonics+1):
            f=f0*h
            if f>4000: break
            idx=int(np.argmin(np.abs(freqs-f)))
            w=max(1,int(idx*0.05)); lo,hi=max(0,idx-w),min(len(fft)-1,idx+w)
            if fft[lo:hi].max()>0.03*fft.max():
                midi=round(69+12*math.log2(f/440.)); notes.add(NOTE_NAMES[midi%12])
        return list(notes)

    def _match_chord(self, detected):
        if len(detected)<2: return "",0.
        det=set(detected); best_n,best_s="",0.
        templates = {
            "C-Dur":["C","E","G"],"D-Dur":["D","F#","A"],"E-Dur":["E","G#","H"],
            "F-Dur":["F","A","C"],"G-Dur":["G","H","D"],"A-Dur":["A","C#","E"],
            "C-Moll":["C","D#","G"],"D-Moll":["D","F","A"],"E-Moll":["E","G","H"],
            "A-Moll":["A","C","E"],"G-Moll":["G","A#","D"],
            "C7":["C","E","G","A#"],"G7":["G","H","D","F"],"A7":["A","C#","E","G"],
            "Em7":["E","G","H","D"],"Am7":["A","C","E","G"],
            "Cmaj7":["C","E","G","H"],"Fmaj7":["F","A","C","E"],
        }
        enharmonic = {"D#":"Eb","A#":"Bb","G#":"Ab","C#":"Db","F#":"Gb"}
        for name,template in templates.items():
            ts=set(enharmonic.get(t,t) for t in template)
            ds=set(enharmonic.get(d,d) for d in det)
            if not ts: continue
            sc=len(ds&ts)/len(ts)
            if sc>best_s: best_s=sc; best_n=name
        return best_n, best_s

    def stop(self):
        if self._stream:
            try: self._stream.stop_stream(); self._stream.close()
            except: pass
        if self._pa:
            try: self._pa.terminate()
            except: pass

    def get_state(self) -> Dict:
        base = {
            "rms":         round(self.rms,3),
            "pitch_hz":    round(self.pitch_hz,2),
            "pitch_midi":  self.pitch_midi,
            "note_name":   self.note_name,
            "cents_off":   round(self.cents_off,1),
            "spectrum":    [round(v,3) for v in self.spectrum[::4]],
            "wave":        [round(v,3) for v in self.wave_hist[::4]],
            "onset":       self.onset,
            "bass":        round(self.bass_rms,3),
            "mid":         round(self.mid_rms,3),
            "high":        round(self.high_rms,3),
            "chord_match": self.chord_match,
            "chord_conf":  round(self.chord_conf,2),
            "notes":       self.detected_notes[:6],
            "is_silent":   self.is_silent,
            "active":      self.active,
            "attack":      self.attack_detected,
            "cleanliness": round(self.cleanliness, 3),
            "buzz":        self.buzz_detected,
            "harm_ratio":  round(self.harmonic_ratio, 3),
        }
        base["poly"]  = self.poly.get_state()
        base["strum"] = self.strum.get_state()
        return base


# ══════════════════════════════════════════════════════════════════════════════
#  SONG-MODUS CONTROLLER  (NEU v5)
# ══════════════════════════════════════════════════════════════════════════════
class SongMode:
    """
    Spielt eine Song-Timeline ab:
    - Zeigt aktuellen Akkord, nächsten Akkord, Akkord-Wechsel-Countdown
    - Verfolgt Präzision des Wechselzeitpunkts
    - Berechnet Song-Score
    - Gibt Strumming-Pattern für aktuellen Akkord zurück
    """
    def __init__(self):
        self.active    = False
        self.song_key  = ""
        self.song_data : Dict = {}
        self.timeline  : List = []
        self.pos       = 0         # Aktuelle Position in Timeline
        self.bar_pos   = 0.        # Position innerhalb des aktuellen Blocks (0-1)
        self._block_start = 0.
        self._bar_duration = 2.0   # Sekunden pro Bar (wird aus BPM berechnet)
        self.cur_chord = ""
        self.next_chord = ""
        self.cur_pattern = ""
        self.bars_left = 0
        self.score     = 0.        # Song-Gesamtscore 0-1
        self.changes   = 0         # Anzahl Akkord-Wechsel
        self.good_changes = 0      # Pünktliche Wechsel
        self.finished  = False
        self._last_change_acc = 1.0

    def start(self, song_key: str, songs_dict: Dict, bpm_override: int = 0):
        songs = songs_dict
        if song_key not in songs: return False
        self.song_data = songs[song_key]; self.song_key = song_key
        bpm = bpm_override if bpm_override else self.song_data.get("bpm", 80)
        self._bar_duration = 60.0 / bpm * 4   # 4 Beats pro Bar
        self.timeline = self.song_data.get("timeline", [])
        if not self.timeline: return False
        self.pos=0; self.bar_pos=0.; self.score=0.; self.changes=0
        self.good_changes=0; self.finished=False; self.active=True
        self._block_start=time.time()
        self._refresh_cur()
        return True

    def stop(self):
        self.active=False; self.cur_chord=""; self.next_chord=""; self.cur_pattern=""

    def _refresh_cur(self):
        if self.pos >= len(self.timeline): self.finished=True; self.active=False; return
        chord, bars, pattern = self.timeline[self.pos]
        self.cur_chord=chord; self.cur_pattern=pattern; self.bars_left=bars
        npos=self.pos+1
        if npos<len(self.timeline): self.next_chord=self.timeline[npos][0]
        else: self.next_chord="(Ende)"

    def update(self, acc: float = 0.) -> Tuple[bool,bool]:
        """Returns: (chord_changed, song_finished)"""
        if not self.active: return False,False
        now=time.time()
        elapsed=now-self._block_start
        chord,bars,pattern=self.timeline[self.pos]
        block_dur=self._bar_duration*bars
        self.bar_pos=min(1.,elapsed/block_dur)
        changed=False
        if elapsed>=block_dur:
            self._block_start=now; self.pos+=1
            self.changes+=1
            timing_err=abs(elapsed-block_dur)
            if timing_err<self._bar_duration*0.25: self.good_changes+=1
            self.score=(self.good_changes/max(1,self.changes))*0.7 + acc*0.3
            self._refresh_cur(); changed=True
        return changed, self.finished

    def get_state(self) -> Dict:
        return {
            "active":       self.active,
            "song_key":     self.song_key,
            "cur_chord":    self.cur_chord,
            "next_chord":   self.next_chord,
            "cur_pattern":  self.cur_pattern,
            "bar_pos":      round(self.bar_pos,3),
            "bars_left":    self.bars_left,
            "pos":          self.pos,
            "total":        len(self.timeline),
            "score":        round(self.score,3),
            "changes":      self.changes,
            "good_changes": self.good_changes,
            "finished":     self.finished,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TUTORIAL MANAGER  (NEU v5)
# ══════════════════════════════════════════════════════════════════════════════
class TutorialManager:
    """
    Steuert Video-Tutorials:
    - Erkennt Trigger-Bedingungen (schlechte Haltung, Barré, niedr. Genauigkeit)
    - Zeigt Schritt-für-Schritt SVG-Animationen in Web-UI
    - Wartezeit zwischen Tutorials (kein Spam)
    - Notiert welche Tutorials schon gezeigt wurden
    """
    COOLDOWN = 45.   # Sekunden zwischen gleichen Tutorials

    def __init__(self, lang="de"):
        self.lang      = lang
        self.active_id : str   = ""
        self.step      : int   = 0
        self._shown    : Dict[str,float] = {}
        self._auto_adv_t = 0.
        self.AUTO_ADV  = 8.    # Sekunden pro Schritt (Auto-Weiter)

    def try_trigger(self, trigger: str) -> bool:
        """Versucht ein Tutorial zu starten. True wenn gestartet."""
        for tid, tdata in TUTORIALS.items():
            if tdata["trigger"] == trigger:
                now = time.time()
                if now - self._shown.get(tid, 0) > self.COOLDOWN:
                    self.active_id = tid; self.step = 0
                    self._shown[tid] = now; self._auto_adv_t = now
                    return True
        return False

    def next_step(self):
        if not self.active_id: return
        tut = TUTORIALS.get(self.active_id)
        if not tut: return
        self.step += 1; self._auto_adv_t = time.time()
        if self.step >= len(tut["steps"]): self.dismiss()

    def dismiss(self): self.active_id=""; self.step=0

    def update(self):
        """Auto-Advance."""
        if not self.active_id: return
        if time.time() - self._auto_adv_t > self.AUTO_ADV:
            self.next_step()

    @property
    def current_tut(self) -> Optional[Dict]:
        if not self.active_id: return None
        return TUTORIALS.get(self.active_id)

    @property
    def current_step_text(self) -> str:
        tut = self.current_tut
        if not tut or self.step >= len(tut["steps"]): return ""
        s = tut["steps"][self.step]
        return s["text"].get(self.lang, s["text"].get("de",""))

    @property
    def current_svg_hint(self) -> str:
        tut = self.current_tut
        if not tut or self.step >= len(tut["steps"]): return ""
        return tut["steps"][self.step].get("svg_hint","")

    def get_state(self) -> Dict:
        tut = self.current_tut
        return {
            "active_id":   self.active_id,
            "step":        self.step,
            "total_steps": len(tut["steps"]) if tut else 0,
            "title":       tut["title"].get(self.lang,"") if tut else "",
            "icon":        tut["icon"] if tut else "",
            "text":        self.current_step_text,
            "svg_hint":    self.current_svg_hint,
        }

# ══════════════════════════════════════════════════════════════════════════════
#  HAND TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class LM:
    __slots__=("x","y","z")
    def __init__(self,x,y,z=0.): self.x=x;self.y=y;self.z=z

class Hand:
    def __init__(self,pts): self.landmark=[LM(x,y) for x,y in pts]

SKEL=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
      (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),
      (5,9),(9,13),(13,17)]
TIPS_IDX=[4,8,12,16,20]

class HandTracker:
    def __init__(self):
        self._mode="skin";self._lm=None
        self._prev=[];self._sm=0.35
        self._wrist=collections.deque(maxlen=25)
        self._no_hand=0
        self.tracker_error: Optional[str] = None
        self.tracker_mode_info: str = "Skin-Farb-Erkennung (Standard)"
        self._try_mp()

    def _try_mp(self):
        """MediaPipe-Initialisierung mit detailliertem Fallback."""
        model = Path(__file__).parent / "hand_landmarker.task"
        if not MP_OK:
            self.tracker_error = "MediaPipe nicht installiert (pip install mediapipe)"
            self.tracker_mode_info = "Skin-Farb-Erkennung aktiv (weniger präzise)"
            print(f"[Tracker] ⚠ {self.tracker_error}")
            return
        if not model.exists():
            self.tracker_error = f"Modell-Datei fehlt: {model.name}"
            self.tracker_mode_info = (
                "Skin-Farb-Erkennung aktiv – für präzises Hand-Tracking "
                f"'{model.name}' ins Programm-Verzeichnis legen"
            )
            print(f"[Tracker] ⚠ {self.tracker_error} → Skin-Modus")
            return
        try:
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision as mpv
            opts = mpv.HandLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model)),
                running_mode=mpv.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self._lm = mpv.HandLandmarker.create_from_options(opts)
            self._mode = "mediapipe"
            self.tracker_error = None
            self.tracker_mode_info = "MediaPipe Hand-Landmarker aktiv (präzise)"
            print("[Tracker] ✓ MediaPipe Hand-Landmarker")
        except ImportError as e:
            self.tracker_error = f"MediaPipe-Modul fehlt: {e}"
            self.tracker_mode_info = "Skin-Farb-Erkennung als Fallback"
            print(f"[Tracker] ⚠ {self.tracker_error}")
        except Exception as e:
            self.tracker_error = f"MediaPipe-Fehler: {e}"
            self.tracker_mode_info = "Skin-Farb-Erkennung als Fallback"
            print(f"[Tracker] ⚠ {self.tracker_error}")

    def _skin(self,frame):
        h,w=frame.shape[:2]
        hsv=cv2.cvtColor(cv2.GaussianBlur(frame,(7,7),0),cv2.COLOR_BGR2HSV)
        m=cv2.bitwise_or(
            cv2.inRange(hsv,np.array([0,18,60]),np.array([22,255,255])),
            cv2.inRange(hsv,np.array([158,18,60]),np.array([180,255,255])))
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        m=cv2.dilate(cv2.morphologyEx(cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k),k,iterations=2)
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        hands=[]
        for cnt in sorted(cnts,key=cv2.contourArea,reverse=True)[:2]:
            if cv2.contourArea(cnt)<7500: continue
            bx,by,bw,bh=cv2.boundingRect(cnt)
            lm=self._est(bx,by,bw,bh,w,h)
            if lm: hands.append(Hand(lm))
        return hands

    def _est(self,bx,by,bw,bh,fw,fh):
        n=lambda x,y:(x/fw,y/fh)
        cx=bx+bw//2;sp=bw*0.44
        fb=[(cx-sp*.88,by+bh*.34),(cx-sp*.44,by+bh*.14),(cx-sp*.04,by+bh*.09),(cx+sp*.36,by+bh*.14),(cx+sp*.76,by+bh*.19)]
        ft=[(cx-sp*.6,by+bh*.04),(cx-sp*.5,by-bh*.06),(cx-sp*.04,by-bh*.11),(cx+sp*.4,by-bh*.01),(cx+sp*.8,by+bh*.04)]
        pts=[None]*21;pts[0]=n(cx,by+bh)
        for fi,(p1,p2,p3,p4) in enumerate([(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]):
            for ji,idx in enumerate([p1,p2,p3,p4]):
                t=ji/3.
                pts[idx]=n(fb[fi][0]+t*(ft[fi][0]-fb[fi][0]),fb[fi][1]+t*(ft[fi][1]-fb[fi][1]))
        return [(p if p else (cx/fw,(by+bh//2)/fh)) for p in pts]

    def _mp_detect(self,frame,ts):
        try:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=_mp.Image(image_format=_mp.ImageFormat.SRGB,data=rgb)
            res=self._lm.detect_for_video(img,ts)
            return [Hand([(lm.x,lm.y) for lm in h]) for h in (res.hand_landmarks or [])]
        except: self._mode="skin";return []

    def _smooth(self,new):
        if not self._prev or len(self._prev)!=len(new):
            self._prev=[[(lm.x,lm.y) for lm in h.landmark] for h in new];return new
        out=[]
        for i,h in enumerate(new):
            if i<len(self._prev):
                spts=[(self._prev[i][j][0]*self._sm+lm.x*(1-self._sm),
                       self._prev[i][j][1]*self._sm+lm.y*(1-self._sm))
                      if j<len(self._prev[i]) else (lm.x,lm.y)
                      for j,lm in enumerate(h.landmark)]
                self._prev[i]=spts;out.append(Hand(spts))
            else: out.append(h)
        return out

    def _draw(self,frame,hand):
        H,W=frame.shape[:2]
        pts={i:(int(lm.x*W),int(lm.y*H)) for i,lm in enumerate(hand.landmark)}
        for a,b in SKEL:
            if a in pts and b in pts:
                cv2.line(frame,pts[a],pts[b],OR["skeleton"],2,cv2.LINE_AA)
        for i,(x,y) in pts.items():
            if i in TIPS_IDX:
                cv2.circle(frame,(x,y),9,OR["fingertip"],-1)
                cv2.circle(frame,(x,y),11,(255,255,255),2)
            else: cv2.circle(frame,(x,y),4,OR["skeleton"],-1)

    def process(self,frame,ts=0):
        raw=self._mp_detect(frame,ts) if self._mode=="mediapipe" else self._skin(frame)
        hands=self._smooth(raw)
        self._no_hand=0 if hands else self._no_hand+1
        for h in hands: self._draw(frame,h)
        return frame,hands

    def px(self,hand,shape):
        H,W=shape[:2]
        return {i:(int(lm.x*W),int(lm.y*H)) for i,lm in enumerate(hand.landmark)}

    # ── Handgelenk-Winkel (verbessert: Daumen-Zeigefinger-Alignment) ────────
    def wrist_ang(self, hand, shape):
        """
        Verbesserte Handgelenk-Analyse:
        - Misst Winkel zwischen Handgelenk→Handfläche und Handfläche→Mittelfinger
        - Zusätzlich: Daumen-zu-Zeigefinger-Grundgelenk relative Position
          (wenn Daumen ÜBER MCP5→MCP2-Linie: Handgelenk zu hoch)
        Gibt zurück: (ist_ok, winkel_grad, alignment_ok)
        """
        p = self.px(hand, shape)
        # Klassischer Winkel Handgelenk→Palm→Mittelfinger
        v1 = np.array(p[9])  - np.array(p[0])   # Wrist→Index-MCP
        v2 = np.array(p[12]) - np.array(p[9])    # Index-MCP→Middle-Tip
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            return True, 0., True
        deg = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
        self._wrist.append(deg)
        avg = float(np.mean(self._wrist))

        # Wrist-Alignment: Daumen-Spitze (4) relativ zu Zeigefinger-MCP (5) und
        # kleiner-Finger-MCP (17) → wenn Daumen Y deutlich ÜBER MCP-Linie: Handgelenk zu hoch
        thumb_tip = np.array(p[4])
        mcp2      = np.array(p[5])
        mcp5      = np.array(p[17])
        # Gerade durch MCP2→MCP5, Daumen-Y relativ dazu
        mcp_mid_y = (mcp2[1] + mcp5[1]) / 2.0
        # Im Bild-Koordinatensystem: kleiner Y = höher im Bild
        # Wenn Daumen Y < MCP-Mitte Y: Daumen ist höher = Handgelenk zu hoch
        thumb_above_mcp = (thumb_tip[1] < mcp_mid_y - 15)  # 15px Toleranz
        alignment_ok = not thumb_above_mcp

        return avg < WRIST_LIMIT, avg, alignment_ok

    # ── Finger-Krümmungs-Analyse (NEU) ──────────────────────────────────────
    def finger_analysis(self, hand, shape):
        """
        Misst Krümmungswinkel jedes Fingers an den PIP-Gelenken.
        Ein zu flacher Finger (< FLAT_THRESHOLD) dämpft möglicherweise
        benachbarte Saiten.

        Gibt zurück: Dict mit pro-Finger {curved: bool, angle: float, flat: bool}
        """
        p = self.px(hand, shape)
        FLAT_THRESHOLD = 155  # Grad – über diesem Winkel gilt Finger als zu flach
        CURL_IDEAL     = 70   # Idealer Krümmungswinkel am PIP

        # Landmark-Gruppen: (MCP, PIP, DIP) je Finger (Index→Pinky)
        FINGER_JOINTS = [
            (5, 6, 7),    # Zeigefinger
            (9, 10, 11),  # Mittelfinger
            (13, 14, 15), # Ringfinger
            (17, 18, 19), # Kleiner Finger
        ]
        FINGER_NAMES = ["Zeigefinger", "Mittelfinger", "Ringfinger", "Kleiner"]

        results = {}
        for name, (mcp_i, pip_i, dip_i) in zip(FINGER_NAMES, FINGER_JOINTS):
            mcp = np.array(p[mcp_i])
            pip = np.array(p[pip_i])
            dip = np.array(p[dip_i])
            v1 = pip - mcp
            v2 = dip - pip
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1 or n2 < 1:
                results[name] = {"curved": True, "angle": 90.0, "flat": False}
                continue
            angle = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
            is_flat    = angle > FLAT_THRESHOLD
            is_curved  = angle < CURL_IDEAL + 30
            results[name] = {
                "curved": is_curved,
                "angle":  round(angle, 1),
                "flat":   is_flat,
            }
        return results

    # ── Fingerkuppen-Winkel: Nagel oder Kuppe? ────────────────────────────────
    def fingertip_angle(self, hand, shape):
        """
        Prüft ob die Fingerkuppe (nicht der Nagel) die Saite berührt.
        Vergleicht DIP→Tip-Winkel zum vorherigen Segment.
        Gibt zurück: {finger: is_tip_correct}
        """
        p = self.px(hand, shape)
        # (PIP, DIP, Tip) je Finger
        TIPS = [(6,7,8),(10,11,12),(14,15,16),(18,19,20)]
        NAMES = ["Zeigefinger","Mittelfinger","Ringfinger","Kleiner"]
        results = {}
        for name, (pip_i, dip_i, tip_i) in zip(NAMES, TIPS):
            pip = np.array(p[pip_i]); dip = np.array(p[dip_i]); tip = np.array(p[tip_i])
            v1 = dip - pip; v2 = tip - dip
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1 or n2 < 1:
                results[name] = True; continue
            angle = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
            # Zu gerader Winkel (>140°) = Nagel könnte Saite berühren
            results[name] = angle < 140
        return results

    def fret_map(self,hand,shape,cal):
        if not cal or len(cal)<4: return None
        p=self.px(hand,shape)
        try:
            M=cv2.getPerspectiveTransform(np.float32(cal),
                                          np.float32([[0,0],[600,0],[600,200],[0,200]]))
        except: return None
        out={}
        for fi,ti in enumerate(TIPS_IDX):
            if ti not in p: continue
            try:
                tr=cv2.perspectiveTransform(np.float32([[p[ti]]]).reshape(-1,1,2),M)[0][0]
                out[fi+1]={"fret":max(0,min(12,int(tr[0]/600*12)+1)),
                           "string":max(1,min(6,int(tr[1]/200*6)+1)),"px":p[ti]}
            except: pass
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  OPENAI / TTS
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  TEXT-PROZESSOR – macht Texte natürlicher für TTS
# ══════════════════════════════════════════════════════════════════════════════
class NaturalTextProcessor:
    """Wandelt Text in natürlich klingende TTS-Varianten um."""

    # Abkürzungen ausschreiben (DE)
    _ABBREV_DE = {
        "XP": "Erfahrungspunkte", "BPM": "Beats pro Minute",
        "Hz": "Hertz", "TTS": "Text zu Sprache",
        "KI": "Künstliche Intelligenz", "AI": "Künstliche Intelligenz",
        "ca.": "circa", "bzw.": "beziehungsweise", "z.B.": "zum Beispiel",
        "u.a.": "unter anderem", "d.h.": "das heißt",
    }
    _ABBREV_EN = {
        "XP": "experience points", "BPM": "beats per minute",
        "Hz": "Hertz", "AI": "artificial intelligence",
        "e.g.": "for example", "i.e.": "that is",
    }

    # Emotionale Präfixe je nach Kontext
    _EMOTE_PREFIX = {
        "success":    ["Super! ", "Wunderbar! ", "Fantastisch! ", "Ja! "],
        "error":      ["Hmm, ", "Moment, ", "Warte mal — ", ""],
        "explaining": ["Also, ", "Schau mal: ", "", "Weißt du was? "],
        "correcting": ["Achtung: ", "Kurze Korrektur — ", "Pass auf: ", ""],
        "proud":      ["Ich bin stolz auf dich! ", "Das klingt toll! ", "Wow! ", ""],
        "neutral":    ["", "", "", ""],
    }

    @classmethod
    def process(cls, text: str, emotion: str = "neutral", lang: str = "de") -> str:
        """Hauptfunktion: Text für natürliche Aussprache aufbereiten."""
        if not text:
            return text

        # Abkürzungen ersetzen
        abbrev = cls._ABBREV_DE if lang == "de" else cls._ABBREV_EN
        for short, full in abbrev.items():
            text = text.replace(short, full)

        # Zahlen natürlich einbetten (z.B. "80 XP" → "achtzig Erfahrungspunkte")
        # Übermäßige Ausrufezeichen dämpfen
        import re
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)

        # Em-Dash / Bindestriche als Sprechpause
        text = text.replace(' – ', ', ')
        text = text.replace(' — ', ', ')
        text = text.replace('…', '...')

        # Klammern weglassen (klingen unnatürlich)
        text = re.sub(r'\(([^)]*)\)', r', ,', text)

        # Emotionalen Prefix hinzufügen (zufällig, damit es nicht repetitiv klingt)
        prefixes = cls._EMOTE_PREFIX.get(emotion, cls._EMOTE_PREFIX["neutral"])
        prefix = random.choice(prefixes)
        if prefix and not text.startswith(prefix.strip()):
            # Nur hinzufügen wenn nicht schon ein emotionaler Start
            first_word = text.split()[0] if text.split() else ""
            emotional_starters = {"Super","Wunderbar","Fantastisch","Toll","Hmm","Also","Schau","Hey","Hallo","Ja"}
            if first_word.rstrip("!,.:") not in emotional_starters:
                text = prefix + text

        return text.strip()

    @classmethod
    def make_ssml(cls, text: str, emotion: str = "neutral") -> str:
        """Erstellt einfaches SSML für ElevenLabs (falls unterstützt)."""
        # ElevenLabs unterstützt kein vollständiges SSML, aber Pausen via <break>
        import re
        # Punkte = kurze Pause
        text = re.sub(r'\. ', '. <break time="300ms"/> ', text)
        # Komma = sehr kurze Pause
        text = re.sub(r', ', ', <break time="150ms"/> ', text)
        # Frage = leicht andere Intonation
        return text


# ══════════════════════════════════════════════════════════════════════════════
#  AI-CHAT + TTS – Groq (LLM) + ElevenLabs (Stimme) mit Fallbacks
# ══════════════════════════════════════════════════════════════════════════════
class AIChat:
    """
    KI-Gesprächspartner mit natürlicher Stimme.

    Priorität LLM:  Groq (kostenlos) → OpenAI → Offline-Antworten
    Priorität TTS:  ElevenLabs (natürlich) → OpenAI TTS → gTTS → Browser
    """

    SYSTEM_PROMPT = {
        "de": (
            "Du bist NoteIQ, eine warme, ermutigende KI-Musiklehrerin für Gitarre und Klavier. "
            "Du sprichst wie eine echte Lehrerin – mit Begeisterung, Geduld und Humor. "
            "Du analysierst in Echtzeit: Handgelenk-Winkel, Finger-Krümmung, Ton-Sauberkeit und Oberton-Qualität. Beziehe dich KONKRET auf diese Daten wenn verfügbar. "
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

    # Offline-Antworten wenn keine API verfügbar
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
        self.lang    = lang
        self._hist: List[Dict] = []
        self.last    = ""
        self.thinking = False
        self._spoken: Dict[str, float] = {}
        self._tts_queue: List[str] = []
        self._tts_lock  = threading.Lock()
        self._ntp = NaturalTextProcessor()

        # ── Groq Client ──
        self._groq = None
        if GROQ_OK and GROQ_API_KEY:
            try:
                self._groq = _Groq(api_key=GROQ_API_KEY)
                print("[AI] Groq ✓  (llama-3.3-70b)")
            except Exception as e:
                print(f"[AI] Groq Fehler: {e}")

        # ── OpenAI Fallback ──
        self._oai = None
        k = os.environ.get("OPENAI_API_KEY", "")
        if OPENAI_OK and k and not self._groq:
            try:
                self._oai = _oai.OpenAI(api_key=k)
                print("[AI] OpenAI ✓ (Fallback)")
            except Exception as e:
                print(f"[AI] OpenAI Fehler: {e}")

        # ── ElevenLabs ──
        self._eleven_ok = bool(ELEVEN_API_KEY and REQUESTS_OK)
        if self._eleven_ok:
            print("[TTS] ElevenLabs ✓ (natürliche Stimme)")
        elif self._oai:
            print("[TTS] OpenAI TTS (nova)")
        elif TTS_OK:
            print("[TTS] gTTS (Fallback)")
        else:
            print("[TTS] Browser Web Speech API")

    def ok(self) -> bool:
        return self._groq is not None or self._oai is not None

    # ── LLM Anfrage ────────────────────────────────────────────────────────
    def ask(self, q: str, ctx: str = "", emotion: str = "explaining"):
        """Frage an die KI stellen (async, Antwort in self.last)."""
        self.thinking = True
        self.last = LANG[self.lang].get("thinking", "...")

        def _run():
            try:
                sys_lang = "de" if self.lang == "de" else "en"
                sys_prompt = self.SYSTEM_PROMPT[sys_lang]
                if ctx:
                    sys_prompt += f"\n\nAktueller Kontext: {ctx}"

                self._hist.append({"role": "user", "content": q})
                msgs = [{"role": "system", "content": sys_prompt}] + self._hist[-12:]

                ans = ""

                # Groq (primär – kostenlos, schnell)
                if self._groq:
                    try:
                        r = self._groq.chat.completions.create(
                            model=GROQ_MODEL, messages=msgs,
                            max_tokens=180, temperature=0.82, top_p=0.92,
                        )
                        ans = r.choices[0].message.content.strip()
                    except Exception as groq_err:
                        err_str = str(groq_err)
                        if "rate_limit" in err_str.lower():
                            ans = "Kurze Pause – zu viele Anfragen. Gleich wieder!"
                        elif "api_key" in err_str.lower() or "auth" in err_str.lower():
                            ans = "KI-API-Key ungültig. Bitte GROQ_API_KEY prüfen."
                            self._groq = None  # Nicht mehr versuchen
                        elif "connection" in err_str.lower() or "timeout" in err_str.lower():
                            # Offline Fallback
                            tips = self.OFFLINE_TIPS.get(self.lang, self.OFFLINE_TIPS["de"])
                            ans = random.choice(tips) + " (KI offline)"
                        else:
                            tips = self.OFFLINE_TIPS.get(self.lang, self.OFFLINE_TIPS["de"])
                            ans = random.choice(tips)
                        print(f"[Groq] Fehler: {groq_err}")

                # OpenAI (Fallback)
                elif self._oai:
                    try:
                        r = self._oai.chat.completions.create(
                            model=OPENAI_MODEL, messages=msgs,
                            max_tokens=180, temperature=0.82,
                        )
                        ans = r.choices[0].message.content.strip()
                    except Exception as oai_err:
                        tips = self.OFFLINE_TIPS.get(self.lang, self.OFFLINE_TIPS["de"])
                        ans = random.choice(tips)
                        print(f"[OpenAI] Fehler: {oai_err}")

                # Offline
                else:
                    tips = self.OFFLINE_TIPS.get(self.lang, self.OFFLINE_TIPS["de"])
                    ans = random.choice(tips)

                self._hist.append({"role": "assistant", "content": ans})
                self.last = ans
                self.thinking = False

                # Natürlichen Text aufbereiten und sprechen
                natural = NaturalTextProcessor.process(ans, emotion, self.lang)
                self._speak_elevenlabs(natural)

            except Exception as e:
                self.last = f"[KI Fehler: {e}]"
                self.thinking = False
                print(f"[AI] Fehler: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── TTS: ElevenLabs (natürlichste Stimme) ─────────────────────────────
    def _speak_elevenlabs(self, text: str):
        """Spricht Text via ElevenLabs – klingt menschlich mit Betonung."""
        def _run():
            try:
                if self._eleven_ok:
                    voice_id = ELEVEN_VOICE_DE if self.lang == "de" else ELEVEN_VOICE_EN
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
                    headers = {
                        "xi-api-key": ELEVEN_API_KEY,
                        "Content-Type": "application/json",
                    }
                    # Voice Settings: stability niedrig = mehr Ausdruck/Betonung
                    payload = {
                        "text": text,
                        "model_id": ELEVEN_MODEL,
                        "voice_settings": {
                            "stability": 0.38,          # niedrig = mehr Emotion
                            "similarity_boost": 0.82,   # hoch = konsistente Stimme
                            "style": 0.45,              # Ausdrucksstärke
                            "use_speaker_boost": True,  # Klarheit
                        },
                    }
                    resp = _requests.post(url, json=payload, headers=headers,
                                          stream=True, timeout=12)
                    if resp.status_code == 401:
                        print("[ElevenLabs] ⚠ API-Key ungültig – verwende Fallback")
                        self._eleven_ok = False
                        raise Exception("ElevenLabs: Auth-Fehler")
                    elif resp.status_code == 429:
                        print("[ElevenLabs] ⚠ Rate-Limit – verwende Fallback")
                        raise Exception("ElevenLabs: Rate-Limit")
                    elif resp.status_code == 200:
                        out = "/tmp/noteiq_eleven.mp3"
                        with open(out, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=4096):
                                if chunk:
                                    f.write(chunk)
                        # Abspielen (mpg123 → afplay → aplay → mplayer)
                        for player in ["mpg123 -q", "afplay", "aplay", "mplayer -really-quiet"]:
                            parts = player.split()
                            try:
                                import subprocess as _sub2
                                _sub2.Popen(parts + [out],
                                           stdout=_sub2.DEVNULL, stderr=_sub2.DEVNULL)
                                return
                            except FileNotFoundError:
                                continue
                    else:
                        print(f"[ElevenLabs] HTTP {resp.status_code}: {resp.text[:200]}")
                        raise Exception("ElevenLabs Fehler")

                # Fallback: OpenAI TTS (nova)
                elif self._oai:
                    r = self._oai.audio.speech.create(
                        model="tts-1-hd",  # HD-Version für bessere Qualität
                        voice="nova",      # Freundlich, warm
                        input=text,
                        response_format="mp3",
                        speed=0.96,        # Leicht langsamer = natürlicher
                    )
                    out = "/tmp/noteiq_openai.mp3"
                    r.stream_to_file(out)
                    import subprocess as _sub2
                    _sub2.Popen(["mpg123", "-q", out],
                               stdout=_sub2.DEVNULL, stderr=_sub2.DEVNULL)

                # Fallback: gTTS (Roboterstimme, aber besser als nichts)
                elif TTS_OK:
                    out = "/tmp/noteiq_gtts.mp3"
                    _gTTS(text=text, lang=self.lang[:2], slow=False).save(out)
                    import subprocess as _sub2
                    _sub2.Popen(["mpg123", "-q", out],
                               stdout=_sub2.DEVNULL, stderr=_sub2.DEVNULL)

                # Kein Server-TTS → Browser macht es (lunaSay im JS)

            except Exception as e:
                print(f"[TTS] Fehler: {e}")
                # Stilles Fallback – Browser-TTS übernimmt

        threading.Thread(target=_run, daemon=True).start()

    # ── Kurzhinweise (Chord OK, Level Up etc.) ─────────────────────────────
    def speak_tip(self, text: str, cd: float = 6.0, emotion: str = "neutral"):
        """Spricht einen kurzen Tipp – mit Cooldown damit es nicht nervt."""
        now = time.time()
        k   = text[:30]
        if now - self._spoken.get(k, 0) > cd:
            self._spoken[k] = now
            natural = NaturalTextProcessor.process(text, emotion, self.lang)
            self._speak_elevenlabs(natural)


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class FeedbackEngine:
    def __init__(self,lang="de"):
        self.lang=lang;self._msgs:List[Msg]=[]

    def t(self,key,**kw):
        tmpl=LANG.get(self.lang,LANG["de"]).get(key,key)
        return tmpl.format(**kw) if kw else tmpl

    def push(self,text,color=None,dur=3.5,bold=False):
        if color is None: color=OR["orange"]
        for m in self._msgs:
            if m.text==text and m.alive(): m.born=time.time();return
        self._msgs.append(Msg(text,color,duration=dur,bold=bold))
        if len(self._msgs)>14: self._msgs=self._msgs[-10:]

    def active(self):
        self._msgs=[m for m in self._msgs if m.alive()]
        return self._msgs[-4:]


# ══════════════════════════════════════════════════════════════════════════════
#  CHORD TRAINER
# ══════════════════════════════════════════════════════════════════════════════
class ChordTrainer:
    def __init__(self,instr,lesson_idx=0):
        self.instr=instr
        pool=LESSONS.get(instr,LESSONS["guitar"])
        self.li=min(lesson_idx,len(pool)-1)
        self.keys=pool[self.li][:]
        self.pos=0;self.mastered=[]
        self._buf=collections.deque(maxlen=50)
        self.hold_start=None;self.hold_prog=0.;self.total=0.
        self.streak=0;self._acc=0.

    @property
    def cur_key(self): return self.keys[self.pos%len(self.keys)]
    @property
    def cur_chord(self): return CHORDS[self.instr].get(self.cur_key,{})

    def set_chord(self,key):
        if key in CHORDS[self.instr]:
            if key not in self.keys: self.keys.append(key)
            self.pos=self.keys.index(key)
            self.hold_start=None;self.hold_prog=0.;self._buf.clear()

    def eval(self,fret_data):
        if fret_data is None: return 0.
        if self.instr=="piano":
            req=len(self.cur_chord.get("keys",[1,2,3]))
            return min(1.,len(fret_data)/max(req,1))
        tgt=self.cur_chord.get("fingers",{})
        if not tgt: return 0.
        ok=sum(1 for fid,(tf,ts) in tgt.items()
               if fid in fret_data
               and abs(fret_data[fid]["fret"]-tf)<=1
               and abs(fret_data[fid]["string"]-ts)<=1)
        return ok/len(tgt)

    def update(self,acc,dt,fb,ai):
        self._buf.append(acc)
        avg=float(np.mean(self._buf)) if self._buf else 0.
        self._acc=avg;mastered=False
        if avg>0.68:
            if self.hold_start is None: self.hold_start=time.time()
            el=time.time()-self.hold_start
            self.hold_prog=min(1.,el/HOLD_TIME)
            rem=HOLD_TIME-el
            if rem>0: fb.push(fb.t("chord_hold",s=rem),OR["orange"],0.9)
            else: self._do_master(fb,ai);mastered=True
        else:
            self.hold_start=None
            self.hold_prog=max(0.,self.hold_prog-dt*0.6)
        return mastered

    def _do_master(self,fb,ai):
        k=self.cur_key;xp=20+self.cur_chord.get("diff",1)*15
        if k not in self.mastered: self.mastered.append(k);self.streak+=1
        fb.push(fb.t("chord_ok",xp=xp),OR["success"],4.,bold=True)
        ai.speak_tip(fb.t("chord_ok",xp=xp), cd=4.0, emotion="success")
        self.hold_start=None;self.hold_prog=0.;self.pos+=1
        self.total=min(1.,len(self.mastered)/max(len(self.keys),1))
        if len(self.mastered)>=len(self.keys):
            fb.push(fb.t("lesson_done"),OR["orangeD"],6.,bold=True)
        else:
            fb.push(fb.t("next_chord",chord=self.cur_key),OR["muted"])
        self._buf.clear()

    def accuracy(self): return self._acc


# ══════════════════════════════════════════════════════════════════════════════
#  METRONOME & CALIBRATOR
# ══════════════════════════════════════════════════════════════════════════════
class Metronome:
    def __init__(self,bpm=80):
        self.bpm=bpm;self.on=True
        self._last=time.time();self.pulse=0.;self._iv=60./bpm

    def update(self):
        if not self.on: self.pulse=0.;return False
        el=time.time()-self._last
        self.pulse=max(0.,1.-el/(self._iv*0.38))
        if el>=self._iv: self._last=time.time();return True
        return False

    def set(self,bpm): self.bpm=max(30,min(250,bpm));self._iv=60./self.bpm


class Calibrator:
    LABELS=["Oben-Links","Oben-Rechts","Unten-Rechts","Unten-Links"]
    def __init__(self): self.pts=[];self.active=False;self.done=False
    def start(self): self.pts=[];self.active=True;self.done=False
    def click(self,x,y):
        if not self.active or len(self.pts)>=4: return
        self.pts.append((x,y))
        if len(self.pts)==4: self.done=True;self.active=False
    def draw(self,frame):
        if not self.active: return
        for i,(px,py) in enumerate(self.pts):
            cv2.circle(frame,(px,py),10,OR["orange"],-1)
            cv2.circle(frame,(px,py),12,(255,255,255),1)
            _txt(frame,str(i+1),(px+14,py-6),0.55,OR["orange"])
        n=len(self.pts)
        if n<4: _txt(frame,f"Klicke {n+1}/4: {self.LABELS[n]}",(20,50),0.7,OR["orange"])


# ══════════════════════════════════════════════════════════════════════════════
#  OPENCV RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _txt(img,text,pos,scale,color,thick=2):
    x,y=pos
    cv2.putText(img,text,(x+2,y+2),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def _panel(img,x1,y1,x2,y2,alpha=0.88,border=True):
    ov=img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),(230,240,250),-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
    if border: cv2.rectangle(img,(x1,y1),(x2,y2),OR["border"],1)

def _bar(img,x,y,w,h,val,fg,bg=(210,220,235),label=""):
    cv2.rectangle(img,(x,y),(x+w,y+h),bg,-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),OR["border"],1)
    fill=max(0,min(w,int(w*val)))
    if fill>0:
        for i in range(fill):
            t=i/max(fill,1)
            r=int(OR["orangeD"][2]+t*(fg[2]-OR["orangeD"][2]))
            g=int(OR["orangeD"][1]+t*(fg[1]-OR["orangeD"][1]))
            b=int(OR["orangeD"][0]+t*(fg[0]-OR["orangeD"][0]))
            cv2.line(img,(x+i,y),(x+i,y+h),(b,g,r),1)
    _txt(img,f"{int(val*100)}%",(x+w//2-10,y+h-3),0.37,OR["coal"],1)
    if label: _txt(img,label,(x+w+6,y+h-3),0.38,OR["muted"],1)

def _ring(img,cx,cy,r,val,color,text=""):
    cv2.ellipse(img,(cx,cy),(r,r),0,0,360,(210,220,230),6)
    ang=int(360*val)
    if ang>0: cv2.ellipse(img,(cx,cy),(r,r),-90,0,ang,color,6,cv2.LINE_AA)
    if text: _txt(img,text,(cx-len(text)*4,cy+5),0.5,OR["coal"],1)

def _guitar_diagram(img,chord,ox,oy,fret_data=None):
    strings,frets=6,5;sg,fg=22,20;W=(strings-1)*sg
    cv2.rectangle(img,(ox-10,oy-28),(ox+W+10,oy+frets*fg+16),(248,252,255),-1)
    cv2.rectangle(img,(ox-10,oy-28),(ox+W+10,oy+frets*fg+16),OR["border"],1)
    _txt(img,chord.get("name","?"),(ox,oy-16),0.72,OR["orange"])
    if "barre_fret" in chord:
        bf=chord["barre_fret"];by2=oy+(bf-1)*fg+2
        cv2.rectangle(img,(ox,by2),(ox+W,by2+fg-4),(*OR["orangeD"][:2],180),-1)
        _txt(img,f"B{bf}",(ox+W+3,by2+fg//2+3),0.3,OR["orangeD"],1)
    cv2.rectangle(img,(ox,oy),(ox+W,oy+3),OR["coal"],-1)
    for s in range(strings): cv2.line(img,(ox+s*sg,oy+3),(ox+s*sg,oy+frets*fg),OR["muted"],1)
    for f in range(1,frets+1): cv2.line(img,(ox,oy+f*fg),(ox+W,oy+f*fg),OR["border"],1)
    fc=[OR["orange"],OR["orangeL"],OR["orangeD"],OR["accent"]]
    for fid,(fret_n,string_n) in chord.get("fingers",{}).items():
        sx=ox+(string_n-1)*sg;fy=oy+(fret_n-1)*fg+fg//2
        col=fc[(fid-1)%4]
        cv2.circle(img,(sx,fy),9,col,-1);cv2.circle(img,(sx,fy),9,(255,255,255),1)
        _txt(img,str(fid),(sx-4,fy+4),0.38,(255,255,255),1)
    for s_idx in range(1,strings+1):
        sx=ox+(s_idx-1)*sg
        if s_idx in chord.get("mute",[]): _txt(img,"x",(sx-4,oy-6),0.38,OR["danger"],1)
        elif s_idx in chord.get("open",[]): cv2.circle(img,(sx,oy-8),5,OR["success"],1)
    if fret_data:
        for fid,fd in fret_data.items():
            sx=ox+(fd["string"]-1)*sg
            fy=oy+(fd["fret"]-1)*fg+fg//2 if fd["fret"]>0 else oy+fg//2
            cv2.circle(img,(sx,fy),13,(255,255,255),2)

def _piano_diagram(img,chord,ox,oy):
    ww,wh,bw,bh=22,62,14,40;n=8
    active=set(chord.get("keys",[]))
    white_notes=[0,2,4,5,7,9,11]
    cv2.rectangle(img,(ox-6,oy-28),(ox+n*ww+6,oy+wh+14),(248,252,255),-1)
    cv2.rectangle(img,(ox-6,oy-28),(ox+n*ww+6,oy+wh+14),OR["border"],1)
    _txt(img,chord.get("name","?"),(ox,oy-16),0.72,OR["orange"])
    for i,note in enumerate(white_notes[:n]):
        col=(240,245,248) if note not in active else OR["orange"]
        cv2.rectangle(img,(ox+i*ww,oy),(ox+(i+1)*ww-2,oy+wh),col,-1)
        cv2.rectangle(img,(ox+i*ww,oy),(ox+(i+1)*ww-2,oy+wh),(160,160,160),1)
        if note in active:
            _txt(img,"*",(ox+i*ww+7,oy+wh-6),0.38,(255,255,255),1)
            fn=chord.get("fingers_rh",{})
            fnum=[k for k,v in fn.items() if v==i]
            if fnum: _txt(img,str(fnum[0]),(ox+i*ww+8,oy+wh-18),0.32,(255,255,255),1)
    black_map={1:0,3:1,6:3,8:4,10:5}
    for semitone,wi in black_map.items():
        if wi>=n-1: continue
        col=OR["orangeD"] if semitone in active else (40,35,30)
        bx=ox+(wi+1)*ww-bw//2
        cv2.rectangle(img,(bx,oy),(bx+bw,oy+bh),col,-1)
        cv2.rectangle(img,(bx,oy),(bx+bw,oy+bh),(100,90,80),1)

def _beat_dot(img,metro,cx,cy,r=20):
    p=metro.pulse
    lo=np.array(OR["border"],float);hi=np.array(OR["orange"],float)
    col=tuple(int(v) for v in (lo+p*(hi-lo)))
    cv2.circle(img,(cx,cy),r,col,-1);cv2.circle(img,(cx,cy),r,OR["border"],1)
    if p>0.7: cv2.circle(img,(cx,cy),int(r*1.4+p*8),(255,180,50),1)
    _txt(img,str(metro.bpm),(cx-13,cy+5),0.48,OR["coal"],1)

def _audio_bars(img,x,y,w,h,spectrum,onset=False):
    n=min(len(spectrum),w//3);bw2=max(1,w//n-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(230,238,245),-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),OR["border"],1)
    mid=y+h//2
    for i in range(n):
        amp=spectrum[int(i/n*len(spectrum))] if spectrum else 0.3
        amp=max(0.05,min(1.,amp));bh3=int(amp*(h//2-2))
        t_r=i/n
        r=int(OR["orangeD"][2]+t_r*(OR["orange"][2]-OR["orangeD"][2]))
        g=int(OR["orangeD"][1]+t_r*(OR["orange"][1]-OR["orangeD"][1]))
        b=int(OR["orangeD"][0]+t_r*(OR["orange"][0]-OR["orangeD"][0]))
        bx2=x+i*(bw2+1)
        cv2.rectangle(img,(bx2,mid-bh3),(bx2+bw2,mid+bh3),(b,g,r),-1)
    if onset: cv2.rectangle(img,(x,y),(x+w,y+h),OR["success"],2)

def _pitch_meter(img,x,y,w,h,cents,note_name):
    cv2.rectangle(img,(x,y),(x+w,y+h),(240,245,250),-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),OR["border"],1)
    cx=x+w//2;mx=cx+int(cents/50*(w//2-6))
    mx=max(x+4,min(x+w-4,mx))
    cv2.line(img,(cx,y+3),(cx,y+h-3),OR["muted"],1)
    for i in range(w):
        dist=abs(i-w//2)
        if dist<4: col=OR["success"]
        elif dist>w//3: col=OR["danger"]
        else:
            alpha=max(0.,1.-dist/(w//2))
            col=(int(OR["success"][0]*alpha),int(OR["success"][1]*alpha),int(OR["success"][2]*alpha))
        cv2.line(img,(x+i,y+h-4),(x+i,y+h-2),col,1)
    col=OR["success"] if abs(cents)<10 else (OR["info"] if abs(cents)<25 else OR["danger"])
    cv2.line(img,(mx,y+2),(mx,y+h-4),col,3)
    _txt(img,note_name,(cx-10,y+h//2+4),0.42,OR["coal"],1)
    if abs(cents)<50:
        sign="+" if cents>0 else ""
        _txt(img,f"{sign}{cents:.0f}ct",(x+2,y+h-2),0.3,col,1)

def _string_indicator(img, x, y, presence, notes, muted):
    """Zeigt 6 Saiten-Präsenz-Balken (polyphoner Analyzer)."""
    sw, sh, sg = 14, 40, 4
    labels=["E","A","D","G","H","e"]
    for si in range(6):
        bx=x+si*(sw+sg)
        p=presence[si] if si<len(presence) else 0.
        is_muted=(si+1) in muted
        col=OR["danger"] if is_muted else (OR["success"] if p>0.3 else OR["muted"])
        cv2.rectangle(img,(bx,y),(bx+sw,y+sh),(210,220,235),-1)
        fill=max(0,int(sh*p))
        if fill>0:
            cv2.rectangle(img,(bx,y+sh-fill),(bx+sw,y+sh),col,-1)
        cv2.rectangle(img,(bx,y),(bx+sw,y+sh),OR["border"],1)
        _txt(img,labels[si],(bx+3,y+sh+12),0.3,OR["coal"],1)
        if notes and si<len(notes) and notes[si]!="–":
            _txt(img,notes[si],(bx+1,y-3),0.28,OR["orangeD"],1)

def _strum_arrow(img, x, y, direction, dynamic, flash):
    """Zeigt Strumming-Richtungspfeil."""
    color=OR["success"] if flash else OR["muted"]
    if direction=="↓":
        pts=np.array([[x+10,y],[x+10,y+20],[x+15,y+20],[x+10,y+30],[x+5,y+20],[x+10,y+20]],np.int32)
        cv2.polylines(img,[pts.reshape((-1,1,2))],False,color,3,cv2.LINE_AA)
    elif direction=="↑":
        pts=np.array([[x+10,y+30],[x+10,y+10],[x+5,y+10],[x+10,y],[x+15,y+10],[x+10,y+10]],np.int32)
        cv2.polylines(img,[pts.reshape((-1,1,2))],False,color,3,cv2.LINE_AA)
    _txt(img,dynamic,(x,y+44),0.32,color,1)

def _song_timeline(img, x, y, w, h, song_state):
    """Zeigt Song-Timeline als horizontaler Balken."""
    if not song_state.get("active"): return
    cv2.rectangle(img,(x,y),(x+w,y+h),(230,238,245),-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),OR["border"],1)
    pos=song_state.get("pos",0); total=max(1,song_state.get("total",1))
    prog=pos/total; bar_pos=song_state.get("bar_pos",0)
    fill=int(w*prog); bar_fill=int((w/total)*bar_pos)
    if fill>0: cv2.rectangle(img,(x,y),(x+fill,y+h),OR["orangeD"],-1)
    if bar_fill>0: cv2.rectangle(img,(x+fill,y),(x+fill+bar_fill,y+h),OR["orange"],-1)
    # Akkord-Labels
    cur=song_state.get("cur_chord",""); nxt=song_state.get("next_chord","")
    _txt(img,cur,(x+4,y+h-3),0.45,OR["white"],1)
    if nxt: _txt(img,f"→{nxt}",(x+w//2,y+h-3),0.38,OR["info"],1)
    # Countdown-Overlay
    bar_left=1.-bar_pos
    if bar_left<0.25:
        cv2.rectangle(img,(x,y-18),(x+w,y-2),OR["danger"],-1)
        _txt(img,"WECHSEL!",(x+w//2-28,y-5),0.45,(255,255,255),1)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN INSTRUCTOR  (v5 – alle Systeme vereint)
# ══════════════════════════════════════════════════════════════════════════════
class Instructor:
    def __init__(self,profile,lang="de"):
        self.profile=profile;self.lang=lang
        self.fb       = FeedbackEngine(lang)
        self.tracker  = HandTracker()
        self.metro    = Metronome(80)
        self.trainer  = ChordTrainer(profile.instrument,profile.lesson_idx)
        self.cal      = Calibrator()
        self.ai       = AIChat(lang)
        self.audio    = AudioAnalyzer()
        self.song     = SongMode()
        self.tut      = TutorialManager(lang)
        self._start   = time.time();self._last_t=time.time()
        self._fps=30.;self._fc=0;self._fps_t=time.time()
        self._show_help=False;self._show_stats=False;self._show_lib=False;self._show_song=False
        self._posture_t=0.;self._tip_t=time.time();self._tip_i=0
        self._wrist_ok=True;self._wrist_ang=0.
        self._no_hand=0;self._session_acc=[]
        self._audio_fb_t=0.;self._poly_fb_t=0.;self._strum_fb_t=0.
        self.state={}
        self.fb.push(self.fb.t("welcome"),OR["orange"],4.)
        self.ai.speak_tip(self.fb.t("welcome"), cd=1.0, emotion="explaining")

    def elapsed_min(self): return (time.time()-self._start)/60.
    def t(self,key,**kw): return self.fb.t(key,**kw)

    def switch_lang(self):
        keys=list(LANG.keys())
        self.lang=keys[(keys.index(self.lang)+1)%len(keys)]
        self.fb.lang=self.lang;self.ai.lang=self.lang;self.tut.lang=self.lang
        self.fb.push(LANG[self.lang]["welcome"],OR["orange"],3.)

    def switch_instr(self,instr):
        self.profile.instrument=instr
        self.trainer=ChordTrainer(instr,self.profile.lesson_idx)
        if self.song.active: self.song.stop()
        self.fb.push("Gitarre" if instr=="guitar" else "Klavier",OR["orange"])

    def start_song(self,song_key=None):
        instr=self.profile.instrument
        songs=SONGS.get(instr,{})
        if not songs: self.fb.push("Keine Songs",OR["danger"]); return
        if song_key is None:
            song_key=random.choice(list(songs.keys()))
        ok=self.song.start(song_key,songs,self.metro.bpm)
        if ok:
            self.fb.push(self.t("song_start",name=song_key),OR["song"],5.)
            self.trainer.set_chord(self.song.cur_chord)
        else:
            self.fb.push("Song nicht gefunden",OR["danger"])

    def process(self,frame):
        now=time.time();dt=max(0.001,now-self._last_t);self._last_t=now
        self._fc+=1
        if self._fc%25==0:
            self._fps=25/max(0.01,now-self._fps_t);self._fps_t=now
        H,W=frame.shape[:2]

        # ── Audio Update (mit Kontext) ─────────────────────────────────────
        cur_chord=self.trainer.cur_chord
        cur_pattern=self.song.cur_pattern if self.song.active else ""
        self.audio.update(target_chord=cur_chord,
                          target_pattern=cur_pattern,
                          metro_bpm=float(self.metro.bpm))

        # ── Song-Modus Update ─────────────────────────────────────────────
        if self.song.active:
            chord_changed,song_done=self.song.update(self.audio.poly.chord_score)
            if chord_changed:
                self.trainer.set_chord(self.song.cur_chord)
                self.fb.push(self.t("song_change",chord=self.song.cur_chord),OR["song"],2.)
            if song_done:
                xp=50+int(self.song.score*100)
                self.fb.push(self.t("song_done",xp=xp),OR["success"],6.,bold=True)
                self.profile.add_xp(xp)

        # ── Tutorial-System Update ────────────────────────────────────────
        self.tut.update()

        # ── Audio-Feedback ────────────────────────────────────────────────
        if now-self._audio_fb_t>2.5 and not self.audio.is_silent:
            self._audio_fb_t=now
            if self.audio.chord_match:
                cur=self.trainer.cur_key
                if self.audio.chord_match in(cur,cur.replace("-","").replace(" ","")):
                    if self.audio.chord_conf>0.6:
                        self.fb.push(self.t("pitch_match"),OR["success"],2.)
                elif self.audio.chord_conf>0.5:
                    self.fb.push(f"{self.t('pitch_close')}: {self.audio.chord_match}",OR["info"],2.)

        # ── Polyphoner Feedback (Saiten) ──────────────────────────────────
        if now-self._poly_fb_t>4.0 and not self.audio.is_silent:
            self._poly_fb_t=now
            poly=self.audio.poly
            if poly.chord_score>0.75:
                self.fb.push(self.t("poly_ok"),OR["success"],2.)
            elif poly.muted_strings:
                self.fb.push(f"{self.t('poly_bad')} Saite {poly.muted_strings[0]}",OR["info"],3.)
                self.tut.try_trigger("accuracy_low")

        # ── Sauberkeits-Feedback (Buzz + Cleanliness) ────────────────────
        if not self.audio.is_silent and now - getattr(self,'_clean_fb_t',0) > 4.0:
            if self.audio.buzz_detected:
                self._clean_fb_t = now
                msg = ("Saite schnarrt! Finger fester andrücken – näher zum Bund!" 
                       if self.lang=="de" else
                       "String buzzing! Press finger harder – closer to fret!")
                self.fb.push(msg, OR["danger"], 3.5)
                self.ai.speak_tip(msg, cd=6., emotion="correcting")
            elif self.audio.cleanliness < 0.38:
                self._clean_fb_t = now
                msg = ("Ton nicht ganz sauber – Fingerdruck erhöhen!" 
                       if self.lang=="de" else
                       "Tone not clean – increase finger pressure!")
                self.fb.push(msg, OR["info"], 3.)
                self.ai.speak_tip(msg, cd=8., emotion="correcting")

        # ── Attack-Timing Feedback ────────────────────────────────────────
        # Nur beim Anschlag analysieren (nicht ständig)
        if self.audio.attack_detected and not self.audio.is_silent:
            # Sofort bei Anschlag Poly prüfen
            if self.audio.poly.chord_score < 0.3 and self.audio.cleanliness > 0.5:
                # Klarer Anschlag aber schlechter Akkord = Griff-Problem
                if now - getattr(self,'_grip_fb_t',0) > 6.:
                    self._grip_fb_t = now
                    self.fb.push("Griffbild prüfen!", OR["info"], 2.)

        # ── Strumming-Feedback ─────────────────────────────────────────────
        if now-self._strum_fb_t>3.0 and self.audio.strum.onset_flash:
            self._strum_fb_t=now
            strum=self.audio.strum
            if cur_pattern and strum.pattern_match>0.75:
                self.fb.push(self.t("strum_ok"),OR["success"],2.)
            elif cur_pattern and strum.pattern_match<0.4 and len(strum.pattern_live)>=4:
                self.fb.push(self.t("strum_bad"),OR["info"],3.)
                self.tut.try_trigger("strum_wrong")

        # ── Hand Tracking ─────────────────────────────────────────────────
        ts=int(now*1000)
        frame,hands=self.tracker.process(frame,ts)
        fret_data=None;self._posture_t+=dt

        if hands:
            self._no_hand=0;h0=hands[0]
            if self.cal.done:
                fret_data=self.tracker.fret_map(h0,frame.shape,self.cal.pts)
            if self._posture_t>=2.:
                self._posture_t=0

                # ── Handgelenk (verbessert: 3 Rückgabewerte) ─────────────
                wrist_result = self.tracker.wrist_ang(h0,frame.shape)
                self._wrist_ok  = wrist_result[0]
                self._wrist_ang = wrist_result[1]
                align_ok        = wrist_result[2] if len(wrist_result)>2 else True
                self._align_ok  = align_ok

                if not self._wrist_ok:
                    self.fb.push(self.t("posture_warn"),OR["danger"],3.5)
                    self.ai.speak_tip(self.t("posture_warn"), cd=8.0, emotion="correcting")
                    self.tut.try_trigger("posture_bad")
                elif not align_ok:
                    # Daumen zu hoch = Handgelenk zu hoch
                    self.fb.push("Handgelenk tiefer – Daumen unter den Hals!",OR["info"],3.)
                    self.ai.speak_tip("Handgelenk tiefer halten!", cd=10., emotion="correcting")

                # ── Finger-Krümmungs-Analyse ──────────────────────────────
                finger_data = self.tracker.finger_analysis(h0,frame.shape)
                flat_fingers = [n for n,d in finger_data.items() if d.get("flat")]
                if flat_fingers and now - getattr(self,'_finger_fb_t',0) > 5.:
                    self._finger_fb_t = now
                    name = flat_fingers[0]
                    msg = f"{name} zu flach – Fingerkuppe einsetzen!"
                    self.fb.push(msg, OR["info"], 3.5)
                    self.ai.speak_tip(msg, cd=8., emotion="correcting")
                    self.tut.try_trigger("finger_flat")

                # ── Fingerkuppen-Winkel ────────────────────────────────────
                tip_data = self.tracker.fingertip_angle(h0,frame.shape)
                wrong_tips = [n for n,ok in tip_data.items() if not ok]
                if wrong_tips and not flat_fingers and now - getattr(self,'_tip_fb_t',0) > 7.:
                    self._tip_fb_t = now
                    self.fb.push(f"Fingerkuppe nutzen, nicht den Nagel!", OR["info"], 3.)
        else:
            self._no_hand+=1

        # ── Training ──────────────────────────────────────────────────────
        acc=self.trainer.eval(fret_data)
        self._session_acc.append(acc)
        leveled=self.trainer.update(acc,dt,self.fb,self.ai)
        if leveled:
            did=self.profile.add_xp(20+self.trainer.cur_chord.get("diff",1)*15)
            if did:
                self.fb.push(self.t("level_up",lvl=self.profile.level),OR["orangeD"],5.,bold=True)
                self.ai.speak_tip(self.t("level_up",lvl=self.profile.level), cd=2.0, emotion="proud")
                # Ab Level 3: Flair freischalten
                if self.profile.level >= 3:
                    flair_msg = self.t("flair_unlock") if hasattr(self,"t") else "🌹 Flamenco & Latin freigeschaltet!"
                    self.fb.push(flair_msg, OR["song"], 6., bold=False)

        # Barré-Tutorial bei schwierigen Akkorden
        if self.trainer.cur_chord.get("barre_fret") and acc<0.3 and self._fc%300==0:
            self.tut.try_trigger("barre_chord")

        # ── Metronome ─────────────────────────────────────────────────────
        self.metro.update()

        # ── Tipp-Rotation ─────────────────────────────────────────────────
        if now-self._tip_t>30.:
            tips=TIPS.get(self.profile.instrument,[])
            if tips:
                self.fb.push(f"Tipp: {tips[self._tip_i%len(tips)]}",OR["muted"],7.)
                self._tip_i+=1
            self._tip_t=now

        self.cal.draw(frame)

        # ══ RENDERING ════════════════════════════════════════════════════
        warm=np.zeros_like(frame);warm[:,:,2]=8;warm[:,:,1]=4
        frame=cv2.add(frame,warm)
        PW,RW=248,232
        _panel(frame,0,0,PW,H)
        _panel(frame,W-RW,0,W,H)

        # ──── LINKES PANEL ────────────────────────────────────────────────
        cv2.rectangle(frame,(8,8),(PW-8,36),OR["orange"],-1)
        cv2.rectangle(frame,(8,8),(PW-8,36),OR["orangeD"],1)
        instr_label="  Gitarre" if self.profile.instrument=="guitar" else "  Klavier"
        grp=self.trainer.cur_chord.get("group","")
        if grp: instr_label+=f"  [{grp}]"
        _txt(frame,instr_label,(14,28),0.52,(255,255,255))

        chord=self.trainer.cur_chord
        if self.profile.instrument=="guitar":
            _guitar_diagram(frame,chord,14,76,fret_data)
        else:
            _piano_diagram(frame,chord,8,76)

        d=chord.get("diff",1)
        _txt(frame,"★"*d+"☆"*(3-d),(12,222),0.58,OR["orange"])

        tip=chord.get("tip","")
        words=tip.split();lines=[];cl=[]
        for w in words:
            cl.append(w)
            if len(" ".join(cl))>30: lines.append(" ".join(cl[:-1]));cl=[w]
        if cl: lines.append(" ".join(cl))
        for li,ln in enumerate(lines[:3]):
            _txt(frame,ln,(10,238+li*17),0.34,OR["muted"],1)

        y0=296
        _txt(frame,"Genauigkeit",(10,y0),0.38,OR["muted"],1)
        _bar(frame,10,y0+8,PW-22,12,acc,OR["orange"])
        _txt(frame,"Halten",(10,y0+30),0.38,OR["muted"],1)
        _bar(frame,10,y0+38,PW-22,12,self.trainer.hold_prog,OR["success"])
        _txt(frame,"Lektion",(10,y0+60),0.38,OR["muted"],1)
        _bar(frame,10,y0+68,PW-22,12,self.trainer.total,(0,130,200))
        _txt(frame,f"Gemeistert: {len(self.trainer.mastered)}/{len(self.trainer.keys)}",
             (10,y0+96),0.44,OR["success"])
        ms="  ".join(self.trainer.mastered[:5])
        if ms: _txt(frame,ms,(10,y0+114),0.33,OR["success"],1)

        # Audio-Info links
        y1=y0+134
        cv2.line(frame,(10,y1),(PW-10,y1),OR["border"],1)
        _txt(frame,"Audio",(10,y1+14),0.38,OR["muted"],1)
        if not self.audio.is_silent:
            _txt(frame,f"{self.audio.note_name} {self.audio.pitch_hz:.0f}Hz",
                 (10,y1+30),0.42,OR["orange"])
            _pitch_meter(frame,10,y1+36,PW-22,20,self.audio.cents_off,self.audio.note_name)
        else:
            _txt(frame,"–",(10,y1+30),0.42,OR["muted"])
        _audio_bars(frame,10,y1+64,PW-22,24,self.audio.spectrum,self.audio.onset)

        # Polyphon-Saiten
        if self.profile.instrument=="guitar":
            _txt(frame,"Saiten",(10,y1+98),0.34,OR["muted"],1)
            _string_indicator(frame,10,y1+106,
                              self.audio.poly.string_presence,
                              self.audio.poly.string_notes,
                              self.audio.poly.muted_strings)

        # Strumming
        y2=y1+165
        cv2.line(frame,(10,y2),(PW-10,y2),OR["border"],1)
        strum=self.audio.strum
        _txt(frame,"Strumming",(10,y2+14),0.38,OR["muted"],1)
        _strum_arrow(frame,12,y2+18,strum.last_dir,strum.last_dynamic,strum.onset_flash)
        if strum.measured_bpm>0:
            _txt(frame,f"{strum.measured_bpm:.0f}BPM",(50,y2+32),0.38,OR["info"])
        if strum.pattern_live:
            _txt(frame,strum.pattern_live,(10,y2+64),0.36,OR["coal"],1)
        if self.song.active and self.song.cur_pattern:
            _txt(frame,f"→{self.song.cur_pattern}",(10,y2+78),0.33,OR["orangeD"],1)
            score_pct=int(strum.pattern_match*100)
            col=OR["success"] if score_pct>=70 else OR["info"] if score_pct>=40 else OR["danger"]
            _txt(frame,f"{score_pct}%",(PW-38,y2+78),0.38,col)

        # Level / XP
        y3=min(H-80,y2+100)
        cv2.line(frame,(10,y3),(PW-10,y3),OR["border"],1)
        _txt(frame,f"Lv.{self.profile.level}",(10,y3+14),0.5,OR["orangeD"])
        _txt(frame,f"{self.profile.name[:12]}",(10,y3+32),0.38,OR["coal"],1)
        xp_p=self.profile.xp/max(1,self.profile.xp_next())
        _bar(frame,10,y3+38,PW-22,8,xp_p,OR["orange"])
        _txt(frame,f"XP {self.profile.xp}/{self.profile.xp_next()}",(10,y3+56),0.32,OR["muted"],1)

        # ──── RECHTES PANEL ───────────────────────────────────────────────
        rx=W-RW+8
        cv2.rectangle(frame,(W-RW+4,8),(W-8,36),OR["orangeD"],-1)
        _txt(frame,f"v5.0  FPS:{self._fps:.0f}",(rx,28),0.46,(255,255,255))

        # Song-Modus Banner
        if self.song.active:
            cv2.rectangle(frame,(W-RW+4,40),(W-8,70),OR["song"],-1)
            _txt(frame,f"♪ {self.song.song_key[:20]}",(rx,58),0.44,(255,255,255))
            _song_timeline(frame,rx,72,RW-16,14,self.song.get_state())
            _txt(frame,f"→ {self.song.next_chord}",(rx,102),0.46,OR["info"])
            _txt(frame,f"Score: {self.song.score*100:.0f}%",(rx+80,102),0.4,OR["success"])
            ry_off=110
        else:
            ry_off=44

        # Metronom
        _beat_dot(frame,self.metro,W-RW+RW//2,ry_off+28,20)
        _txt(frame,"BPM",(rx,ry_off+18),0.34,OR["muted"],1)
        ry_off+=62

        # Akkord-Vollständigkeit (Poly)
        poly=self.audio.poly
        cv2.line(frame,(rx,ry_off),(W-12,ry_off),OR["border"],1)
        _txt(frame,"Akkord-Vollst.",(rx,ry_off+14),0.38,OR["muted"],1)
        _bar(frame,rx,ry_off+18,RW-16,10,poly.chord_score,OR["success"])
        if poly.poly_notes:
            _txt(frame," ".join(poly.poly_notes[:4]),(rx,ry_off+36),0.38,OR["coal"],1)
        if poly.muted_strings:
            _txt(frame,f"Gedämpft: Saite {poly.muted_strings}",(rx,ry_off+52),0.34,OR["danger"],1)
        ry_off+=60

        # Haltung
        cv2.line(frame,(rx,ry_off),(W-12,ry_off),OR["border"],1)
        _txt(frame,"Haltung",(rx,ry_off+14),0.38,OR["muted"],1)
        col=OR["success"] if self._wrist_ok else OR["danger"]
        _txt(frame,f"Handgelenk: {self._wrist_ang:.0f}°",(rx,ry_off+30),0.4,col)
        if not self._wrist_ok:
            _ring(frame,W-RW+RW//2,ry_off+60,18,min(1.,self._wrist_ang/60),OR["danger"],"!")
        ry_off+=88

        # Tutorial-Overlay (CV-Fenster)
        if self.tut.active_id:
            tut=self.tut.current_tut
            if tut:
                cv2.line(frame,(rx,ry_off),(W-12,ry_off),OR["border"],1)
                cv2.rectangle(frame,(rx,ry_off+2),(W-12,ry_off+80),OR["orangeD"],-1)
                icon=tut["icon"]; title=tut["title"].get(self.lang,"")
                _txt(frame,f"{icon} {title}",(rx+2,ry_off+18),0.42,(255,255,255))
                txt=self.tut.current_step_text
                words2=txt.split();lines2=[];cl2=[]
                for ww in words2:
                    cl2.append(ww)
                    if len(" ".join(cl2))>28: lines2.append(" ".join(cl2[:-1]));cl2=[ww]
                if cl2: lines2.append(" ".join(cl2))
                for li,ln in enumerate(lines2[:2]):
                    _txt(frame,ln,(rx+2,ry_off+34+li*16),0.32,(220,230,255),1)
                steps_total=len(tut["steps"])
                _txt(frame,f"Schritt {self.tut.step+1}/{steps_total}",(rx+2,ry_off+72),0.32,(180,200,255),1)
                ry_off+=90

        # Metronom-Takt-Visualisierung (Mitte unten)
        _txt(frame,self.t("hint"),(PW+10,H-10),0.28,OR["muted"],1)

        # Feedback-Messages (Mitte, überlagert)
        msgs=self.fb.active()
        fh_off=80
        for m in msgs:
            a=m.alpha()
            ov=frame.copy()
            bg=(200,220,240) if not m.bold else (20,20,200)
            tw=max(200,len(m.text)*9)
            fx=max(PW+10,W//2-tw//2)
            cv2.rectangle(ov,(fx-6,H//2+fh_off-20),(fx+tw+6,H//2+fh_off+6),bg,-1)
            cv2.addWeighted(ov,a*0.72,frame,1-a*0.72,0,frame)
            col=OR["coal"] if not m.bold else (255,255,255)
            sc=0.62 if m.bold else 0.5
            _txt(frame,m.text,(fx,H//2+fh_off),sc,col,2 if m.bold else 1)
            fh_off+=28

        # Overlays
        if self._show_help:  self._draw_help(frame,PW,W,RW)
        if self._show_stats: self._draw_stats(frame,PW,W,RW)
        if self._show_lib:   self._draw_lib(frame,PW,W,H)
        if self._show_song:  self._draw_song_select(frame,PW,W,H)

        self._update_state(acc)
        return frame

    def _draw_help(self,frame,PW,W,RW):
        ox,oy,bw=PW+20,60,W-PW-RW-40
        _panel(frame,ox-8,oy-8,ox+bw+8,oy+360,0.95)
        _txt(frame,"HILFE",(ox,oy+16),0.72,OR["orange"])
        lines=[
            "Q/ESC – Beenden",  "H – Hilfe ein/aus",
            "S – Statistiken",  "B – Akkord-Bibliothek",
            "G – Song auswählen","T – Akkord wählen",
            "L – Sprache wechseln","1 – Gitarre  2 – Klavier",
            "M – Metronom",     "+/- – BPM ändern",
            "C – Kalibrieren",  "A – KI fragen",
            "P – Nächster Akkord","R – Haltungs-Reset",
        ]
        for i,l in enumerate(lines):
            _txt(frame,l,(ox,oy+40+i*22),0.4,OR["coal"],1)

    def _draw_stats(self,frame,PW,W,RW):
        ox,oy,bw=PW+20,60,W-PW-RW-40
        _panel(frame,ox-8,oy-8,ox+bw+8,oy+400,0.95)
        p=self.profile
        _txt(frame,"STATISTIKEN",(ox,oy+16),0.72,OR["orange"])
        rows=[
            f"Name: {p.name}",          f"Instrument: {p.instrument}",
            f"Level: {p.level}",         f"XP: {p.xp}/{p.xp_next()}",
            f"Ø Genauigkeit: {p.avg_acc():.0f}%",
            f"Gesamt: {p.total_min:.0f} Min",
            f"Sessions: {len(p.sessions)}",
            f"Gemeistert: {len(p.mastered)}",
            f"Best Streak: {p.best_streak}",
            f"Pitch: {self.audio.pitch_hz:.1f}Hz ({self.audio.note_name})",
            f"Poly-Notes: {' '.join(self.audio.poly.poly_notes[:4])}",
            f"Strum-BPM: {self.audio.strum.measured_bpm:.0f}",
        ]
        for i,r in enumerate(rows):
            _txt(frame,r,(ox,oy+38+i*22),0.4,OR["coal"],1)

    def _draw_lib(self,frame,PW,W,H):
        ox,oy=PW+10,20;bw=W-PW-250
        _panel(frame,ox,oy,ox+bw,H-20,0.97)
        _txt(frame,"AKKORD-BIBLIOTHEK",(ox+10,oy+24),0.7,OR["orange"])
        instr=self.profile.instrument
        groups={}
        for k,v in CHORDS[instr].items():
            g=v.get("group","?"); groups.setdefault(g,[]).append((k,v))
        y=oy+44; x=ox+8
        cur=self.trainer.cur_key
        for grp,items in sorted(groups.items()):
            if y>H-30: break
            _txt(frame,grp,(x,y),0.42,OR["orangeD"])
            y+=18
            for k,v in items:
                if y>H-20: break
                col=OR["success"] if k in self.trainer.mastered else \
                    OR["orange"] if k==cur else OR["coal"]
                marker="●" if k==cur else " "
                _txt(frame,f"{marker}{v.get('name',k)}",(x,y),0.36,col,1)
                y+=16
            y+=4

    def _draw_song_select(self,frame,PW,W,H):
        ox,oy=PW+10,20; bw=W-PW-250
        _panel(frame,ox,oy,ox+bw,H-20,0.97)
        _txt(frame,"SONG AUSWÄHLEN  [1-9]",(ox+10,oy+24),0.65,OR["song"])
        instr=self.profile.instrument
        songs=SONGS.get(instr,{})
        y=oy+50; x=ox+10
        for i,(sk,sd) in enumerate(songs.items(),1):
            if y>H-30: break
            _txt(frame,f"[{i}] {sk}",(x,y),0.42,OR["coal"],1)
            _txt(frame,f"    {sd.get('tip','')[:40]}",(x,y+16),0.33,OR["muted"],1)
            y+=38

    def _update_state(self,acc):
        try:
            self.state={
                "profile": {"name":self.profile.name,"level":self.profile.level,
                            "xp":self.profile.xp,"next_xp":self.profile.xp_next(),
                            "avg_acc":round(self.profile.avg_acc(),1),
                            "total_min":round(self.profile.total_min,1)},
                "chord":  {"key":self.trainer.cur_key,
                           "data":self.trainer.cur_chord,
                           "acc":round(acc,3),
                           "hold":round(self.trainer.hold_prog,3),
                           "mastered":self.trainer.mastered,
                           "lesson_total":round(self.trainer.total,3)},
                "training":{"acc":round(acc,3),"streak":self.trainer.streak},
                "metro":  {"bpm":self.metro.bpm,"on":self.metro.on,"pulse":round(self.metro.pulse,3)},
                "posture":{"wrist_ok":self._wrist_ok,"angle":round(self._wrist_ang,1),"align_ok":getattr(self,"_align_ok",True)},
                "session":{"fps":round(self._fps,1),"elapsed_min":round(self.elapsed_min(),2)},
                "lang":   self.lang,
                "ai":     {"last":self.ai.last,"thinking":self.ai.thinking},
                "system_status": {
                    "audio_ok":      self.audio.active,
                    "audio_error":   self.audio.error,
                    "audio_errcode": self.audio.error_code,
                    "cam_ok":        True,   # Kamera läuft wenn wir hier sind
                    "tracker_ok":    self.tracker._mode == "mediapipe",
                    "tracker_error": self.tracker.tracker_error,
                    "tracker_mode":  self.tracker.tracker_mode_info,
                    "ai_ok":         self.ai.ok(),
                    "tts_eleven":    self.ai._eleven_ok,
                    "tts_openai":    self.ai._oai is not None,
                    "groq_ok":       self.ai._groq is not None,
                    "scipy_ok":      SCIPY_OK,
                    "mp_ok":         MP_OK,
                },
                "audio":  self.audio.get_state(),
                "feedback":[{"text":m.text,"bold":m.bold,"alpha":round(m.alpha(),2)}
                            for m in self.fb.active()],
                "library":{"guitar":list(CHORDS["guitar"].keys()),
                           "piano":list(CHORDS["piano"].keys()),
                           "guitar_data":CHORDS["guitar"],
                           "piano_data":CHORDS["piano"]},
                "song":   self.song.get_state(),
                "tutorial":self.tut.get_state(),
                "songs_available": [
                    k for k,v in SONGS.get(self.profile.instrument,{}).items()
                    if self.profile.level >= v.get("min_level", 1)
                ],
            }
        except Exception as e:
            print(f"[State] {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  WEB-UI HTML  (Orange Studio v5.0)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  PÄDAGOGISCHES SYSTEM v11  –  Technique → Exercise → Song → XP → Next Level
# ══════════════════════════════════════════════════════════════════════════════

LESSON_CURRICULUM: Dict[str, List[Dict]] = {

# ══════════════════════════════════════════════════════════════════════════════
#  GITARRE  –  20 Lektionen  ·  Level 1–5
#  Logische Progression:
#    Lv1 (1–5):  Gitarre kennenlernen, Haltung, Erste Töne, Em/Am, D/G
#    Lv2 (6–10): Strumming, C-Dur, 5 Grundakkorde, Fingerpicking, Songs
#    Lv3 (11–14): Barré F/Bm, Erweiterte Akkorde, Blues, Pentatonik
#    Lv4 (15–17): Fortg. Techniken, Jazz-Voicings, Sus/Add-Akkorde
#    Lv5 (18–20): Improvisation, Komposition, eigener Stil
# ══════════════════════════════════════════════════════════════════════════════
    "guitar": [

        # ── LEVEL 1  ───────────────────────────────────────────────────────────
        {
            "id": "g_intro", "title": "Deine Gitarre kennenlernen",
            "level": 1, "duration_min": 10, "xp_reward": 50,
            "topics": ["Teile der Gitarre", "Saiten E-A-D-G-B-e", "Haltung", "Erste Töne"],
            "technique": {
                "name": "Gitarren-Anatomie & Körperhaltung",
                "description": "Die Gitarre hat sechs Saiten: E (tiefste) – A – D – G – B – e (höchste). Der Hals hat Bünde, die bestimmen welche Note du spielst. Sitz aufrecht, Gitarre auf dem rechten Oberschenkel (klassisch) oder links (entspannt). Rücken gerade, Schultern locker.",
                "tips": ["Daumen HINTER den Hals, nicht drüber", "Handgelenk unten – nie nach oben abknicken", "Ellenbogen locker vom Körper, nicht angeklemmt"],
                "luna_speech": "Hallo! Ich bin NoteIQ, deine KI-Musiklehrerin! Heute fangen wir ganz am Anfang an. Nimm deine Gitarre und schau dir jeden Teil genau an – das Fundament für alles was folgt!",
            },
            "exercise": {
                "name": "Saiten kennenlernen",
                "description": "Zupfe jede Saite einzeln von oben nach unten: E – A – D – G – B – e. Sage den Namen laut. Dann umgekehrt. Wiederhole das 5 Mal bis die Namen sitzen.",
                "duration_sec": 120, "target_bpm": 0,
                "strumming": "Einzelne Saiten zupfen",
                "chords": [],
                "luna_speech": "Zupfe jetzt jede Saite und sage den Namen laut dazu. Ich höre ob der Klang klar ist!",
            },
            "song": {
                "name": "Ode an die Freude – Eine Saite",
                "description": "Melodie auf der hohen e-Saite. Bünde: 0-0-2-3 | 3-2-0 | offene Saite bedeutet Bund 0. Langsam, jede Note gleich lang. Tempo 55 BPM.",
                "chords": [], "tempo_bpm": 55,
                "strumming": "Einzelne Töne, Fingerspitze oder Plektrum",
                "luna_speech": "Du spielst schon echte Musik! Ode an die Freude auf einer Saite – Beethoven wäre stolz!",
            },
            "chords": [],
            "speech": {"de": "Hallo! Ich bin NoteIQ. Heute beginnen wir ganz von vorne.", "en": "Hello! I'm Luna. Today we start from scratch."},
        },

        {
            "id": "g_posture", "title": "Haltung & Fingertechnik",
            "level": 1, "duration_min": 15, "xp_reward": 80,
            "topics": ["Sitzposition", "Handgelenk-Technik", "Daumen", "1-2-3-4 Übung"],
            "technique": {
                "name": "Linke-Hand-Grundposition",
                "description": "Die linke Hand greift die Bünde. Finger immer HINTER den Bund – nie auf den Metalldraht. Nur die Fingerkuppen berühren die Saite. Das Handgelenk bleibt unten, leicht nach vorne gekippt – nie nach oben abgeknickt (das schmerzt und blockiert). Die 1-2-3-4 Übung trainiert jeden Finger unabhängig.",
                "tips": ["Finger senkrecht auf die Saite – nicht flach", "Fingernägel kurz halten – sonst klingt es nicht", "Wrist rund und tief – wie eine Kugel unterm Hals"],
                "luna_speech": "Deine Fingerhaltung bestimmt alles! Wenn der Ton schnarrt oder dumpf klingt, liegt es fast immer an der Finger-Position. Ich beobachte dein Handgelenk die ganze Zeit!",
            },
            "exercise": {
                "name": "1-2-3-4 Chromatische Übung",
                "description": "Jede Saite: Finger 1 auf Bund 1, Finger 2 auf Bund 2, Finger 3 auf Bund 3, Finger 4 auf Bund 4. Aufwärts E-A-D-G-B-e, dann rückwärts. Metronom 52 BPM – langsam und sauber ist wichtiger als schnell.",
                "duration_sec": 180, "target_bpm": 52,
                "strumming": "Einzelne Töne, chromatic",
                "chords": ["Em"],
                "luna_speech": "Die 1-2-3-4 Übung klingt langweilig – aber sie ist das Geheimnis schneller Finger! Jeder große Gitarrist macht sie täglich.",
            },
            "song": {
                "name": "Smoke on the Water – Riff (Deep Purple)",
                "description": "Tiefe E-Saite: Bünde 0-3-5, Pause, 0-3-6-5, Pause, 0-3-5-3. Eines der bekanntesten Riffs der Welt! Tempo 65 BPM.",
                "chords": [], "tempo_bpm": 65,
                "strumming": "Einzelne Töne, tiefe E-Saite",
                "luna_speech": "Smoke on the Water! Jeder Gitarrist kennt dieses Riff. Mit deiner 1-2-3-4 Technik spielst du es jetzt!",
            },
            "chords": ["Em"],
            "speech": {"de": "Deine Haltung ist das A und O! Ich beobachte dein Handgelenk.", "en": "Your posture is everything! I watch your wrist."},
        },

        {
            "id": "g_open_strings", "title": "Offene Akkorde: Em & Am",
            "level": 1, "duration_min": 20, "xp_reward": 100,
            "topics": ["E-Moll Griff", "A-Moll Griff", "Erster Akkordwechsel"],
            "technique": {
                "name": "Em und Am – Die einfachsten Akkorde",
                "description": "Em (E-Moll): Finger 2 auf A-Saite Bund 2, Finger 3 auf D-Saite Bund 2. Alle 6 Saiten klingen. Am (A-Moll): Finger 1 auf H-Saite Bund 1, Finger 2 auf D-Saite Bund 2, Finger 3 auf G-Saite Bund 2. Nur 5 Saiten (nicht die tiefe E). Finger bilden ein Dreieck.",
                "tips": ["Em: setze beide Finger gleichzeitig – nicht nacheinander", "Am: das Dreieck einprägen, nicht einzeln denken", "Alle Saiten zupfen und auf schnarrende Töne hören"],
                "luna_speech": "Em ist der perfekte Einstiegs-Akkord – nur zwei Finger! Mit Em und Am zusammen klingt es schon nach echter Musik. Das ist aufregend!",
            },
            "exercise": {
                "name": "Em – Am Wechsel",
                "description": "Em für 4 Schläge anschlagen, dann Am für 4 Schläge. Metronom 55 BPM. Der Wechsel muss FLIESSEND sein – keine Pause dazwischen. Übe nur den Wechsel: Em-Griff aufnehmen, Am-Griff legen, zurück.",
                "duration_sec": 240, "target_bpm": 55,
                "strumming": "4 Abschläge pro Akkord",
                "chords": ["Em", "Am"],
                "luna_speech": "Der Akkord-Wechsel ist das Schwierigste am Anfang! Erst langsam üben – ich zähle laut mit dir.",
            },
            "song": {
                "name": "House of the Rising Sun (The Animals)",
                "description": "Am – C – D – F – Am – C – E. Arpeggio-Muster: Bass-Note, dann Akkord-Töne nacheinander. Tempo 72 BPM. Ein echter Klassiker mit deinen ersten Akkorden!",
                "chords": ["Am", "C", "D", "F", "E"], "tempo_bpm": 72,
                "strumming": "Arpeggio: Bass – Mitte – Hoch",
                "luna_speech": "House of the Rising Sun! Ein Klassiker. Dein Am-Griff trägt den ganzen Song – du bist bereit dafür!",
            },
            "chords": ["Em", "Am", "E", "A"],
            "speech": {"de": "Em – nur zwei Finger! So einfach und klingt so gut.", "en": "Em – only two fingers! So simple and sounds so good."},
        },

        {
            "id": "g_dm_g", "title": "Dm & G – Akkorde erweitern",
            "level": 1, "duration_min": 20, "xp_reward": 110,
            "topics": ["D-Moll Griff", "G-Dur Griff", "3-Akkord-Wechsel"],
            "technique": {
                "name": "Dm und G – Neue Farben",
                "description": "Dm (D-Moll): Finger 1 auf e-Saite Bund 1, Finger 2 auf G-Saite Bund 2, Finger 3 auf H-Saite Bund 3. 4 Saiten (D-G-H-e). G-Dur: Finger 2 auf A-Saite Bund 2, Finger 3 auf E-Saite Bund 3, Finger 4 auf e-Saite Bund 3 – alle 6 Saiten klingen. G klingt offen und hell.",
                "tips": ["Dm: Finger 1 liegt schräg – fast ein Mini-Barré", "G: Der kleine Finger auf e streckt sich weit – üben!", "G als Alternative: nur Finger 2 und 3 (tiefer G-Sound)"],
                "luna_speech": "Mit Dm und G öffnen sich ganz neue Klang-Welten. Dm klingt melancholisch, G klingt offen und froh – das ist Musik!",
            },
            "exercise": {
                "name": "Am – Dm – G – Em Progression",
                "description": "4 Akkorde, je 4 Schläge, 58 BPM. Am → Dm → G → Em → zurück. Diese Progression klingt schon fast wie ein richtiger Song! Fokus auf fließende Wechsel.",
                "duration_sec": 300, "target_bpm": 58,
                "strumming": "4 Abschläge, gleichmäßig",
                "chords": ["Am", "Dm", "G", "Em"],
                "luna_speech": "Diese vier Akkorde zusammen klingen fantastisch! Konzentriere dich auf den Wechsel – ich höre ob er fließt.",
            },
            "song": {
                "name": "All of Me – John Legend (vereinfacht)",
                "description": "Am – F – C – G. Strumming: Abschlag auf 1 und 3. Tempo 63 BPM. Einer der bekanntesten modernen Songs!",
                "chords": ["Am", "Dm", "G", "C"], "tempo_bpm": 63,
                "strumming": "D – D – D – D",
                "luna_speech": "All of Me! Genau diese Akkorde. Du klingst jetzt schon wie ein echter Musiker!",
            },
            "chords": ["Dm", "G", "Am", "Em"],
            "speech": {"de": "Dm klingt melancholisch, G offen und hell – Musik hat Gefühle!", "en": "Dm sounds melancholic, G open and bright – music has feelings!"},
        },

        {
            "id": "g_d_chord", "title": "D-Dur & C-Dur – Die nächsten Schritte",
            "level": 1, "duration_min": 20, "xp_reward": 120,
            "topics": ["D-Dur Griff", "C-Dur Griff", "5-Akkord-Kombinationen"],
            "technique": {
                "name": "D-Dur und C-Dur",
                "description": "D-Dur: Finger 1 auf G-Saite Bund 2, Finger 2 auf e-Saite Bund 2, Finger 3 auf H-Saite Bund 3. Nur 4 Saiten (D-G-H-e). C-Dur: Finger 1 auf H-Saite Bund 1, Finger 2 auf D-Saite Bund 2, Finger 3 auf A-Saite Bund 3. 5 Saiten. C-Dur ist der kniffligste Anfänger-Akkord – Geduld!",
                "tips": ["D-Dur: Dreieck eng zusammen – alle drei Finger auf benachbarten Saiten", "C-Dur: der gestreckte Finger 3 auf A ist der Trick", "D→C Wechsel: Finger 2 und 3 wandern nach links – minimal bewegen"],
                "luna_speech": "D und C sind die letzten zwei der großen Fünf! C-Dur braucht etwas Geduld – aber wenn du ihn hast, kannst du hunderte Songs spielen.",
            },
            "exercise": {
                "name": "G – D – C – G Progression",
                "description": "Die Basis vieler Pop-Songs! G → D → C → G, je 4 Schläge, 60 BPM. Dann auch: G – D – Em – C (4 Chords der Popmusik).",
                "duration_sec": 300, "target_bpm": 60,
                "strumming": "D – DU – DU",
                "chords": ["G", "D", "C", "Em"],
                "luna_speech": "G-D-C – diese drei Akkorde sind die Basis von hunderten Hits! Wechsel sie so fließend wie möglich.",
            },
            "song": {
                "name": "Knockin' on Heaven's Door (Bob Dylan)",
                "description": "G – D – Am – G – D – C. Pattern: D-DU-UDU. Tempo 72 BPM. Einer der größten Songs aller Zeiten – mit genau diesen Akkorden!",
                "chords": ["G", "D", "Am", "C"], "tempo_bpm": 72,
                "strumming": "D-DU-UDU",
                "luna_speech": "Bob Dylan! Knockin' on Heaven's Door – du spielst jetzt echten Rock-Klassiker!",
            },
            "chords": ["D", "C", "G", "Em"],
            "speech": {"de": "D und C – die letzten der großen Fünf. Dann kannst du alles spielen!", "en": "D and C – the last of the big five. Then you can play everything!"},
        },

        # ── LEVEL 2  ───────────────────────────────────────────────────────────
        {
            "id": "g_strumming_basic", "title": "Strumming – Rhythmus und Gefühl",
            "level": 2, "duration_min": 20, "xp_reward": 130,
            "topics": ["Plektrum 45°", "D-D-U-U-D-U Muster", "Auf- und Abschlag", "Betonungen"],
            "technique": {
                "name": "Das Strumming-System",
                "description": "Das Plektrum hältst du in einem 45° Winkel zur Saite – das gibt weicheren Klang und weniger Widerstand. Grundmuster: D-D-U-U-D-U (Down-Down-Up-Up-Down-Up). Zähle: 1 – 2 – und – 3 – und – 4. Die Bewegung kommt aus dem HANDGELENK, nicht dem ganzen Arm. Schultern entspannt.",
                "tips": ["45° Winkel – nie senkrecht zum Plektrum", "Handgelenk locker wie eine schlappende Hand", "Abschlag betont – Aufschlag leichter"],
                "luna_speech": "Rhythmus ist die Seele der Musik! Das Strumming-Pattern macht einen guten Gitarristen aus. Mit diesem Muster klingst du sofort professionell.",
            },
            "exercise": {
                "name": "D-D-U-U-D-U auf Em",
                "description": "Nur Em – der einfachste Akkord. Konzentriere dich ausschließlich auf den Rhythmus. Metronom 65 BPM. Wenn das Muster sitzt, wechsle zu Am. Zähle laut: 1-2-und-3-und-4.",
                "duration_sec": 300, "target_bpm": 65,
                "strumming": "D-D-U-U-D-U",
                "chords": ["Em", "Am", "G"],
                "luna_speech": "Nur Em! Kein Akkordwechsel. NUR der Rhythmus. Ich höre ob das Muster gleichmäßig ist!",
            },
            "song": {
                "name": "Wonderwall (Oasis)",
                "description": "Em7 – G – Dsus4 – A7sus4. Strumming: D-DU-UDU. Tempo 87 BPM. Eines der meistgespielten Songs der Geschichte!",
                "chords": ["Em", "G", "D", "A"], "tempo_bpm": 87,
                "strumming": "D-DU-UDU",
                "luna_speech": "Wonderwall von Oasis! Dieses Strumming-Pattern ist der Sound des Jahrzehnts. Das wirst du lieben!",
            },
            "chords": ["Em", "Am", "G", "D"],
            "speech": {"de": "Rhythmus ist die Seele der Musik!", "en": "Rhythm is the soul of music!"},
        },

        {
            "id": "g_5_chords", "title": "Die 5 Grundakkorde meistern",
            "level": 2, "duration_min": 25, "xp_reward": 150,
            "topics": ["E-Dur", "A-Dur", "Wechsel-Training", "Akkord-Roulette"],
            "technique": {
                "name": "E-Dur und A-Dur vervollständigen",
                "description": "E-Dur: Finger 1 auf G-Saite Bund 1, Finger 2 auf A-Saite Bund 2, Finger 3 auf D-Saite Bund 2. Alle 6 Saiten. A-Dur: Finger 2-3-4 auf D-G-H-Saite alle Bund 2 (engste Formation). 5 Saiten. Jetzt hast du E-Em-A-Am-D-Dm-G-C – das Fundament von 90% aller Songs!",
                "tips": ["E-Dur vs Em: genau ein Finger Unterschied – trainiere den Wechsel", "A-Dur: alle drei Finger in einer Linie – eng zusammen", "Akkord-Roulette: jemanden einen Akkord rufen lassen und sofort greifen"],
                "luna_speech": "Mit E und A hast du jetzt das volle Fundament! E-Em-A-Am-D-Dm-G-C. Damit spielst du 90% aller Songs.",
            },
            "exercise": {
                "name": "Akkord-Roulette: alle 8",
                "description": "E – Em – A – Am – D – Dm – G – C, je 2 Schläge, 65 BPM. Dann random: zufällige Reihenfolge üben bis jeder Wechsel automatisch läuft. Ziel: unter 1 Sekunde Reaktionszeit.",
                "duration_sec": 360, "target_bpm": 65,
                "strumming": "2 Abschläge pro Akkord",
                "chords": ["E", "Em", "A", "Am", "D", "G", "C"],
                "luna_speech": "Akkord-Roulette! Ich nenne einen Akkord – du spielst ihn sofort. Das trainiert Muskelgedächtnis!",
            },
            "song": {
                "name": "Brown Eyed Girl (Van Morrison)",
                "description": "G – C – G – D – G – C – G – D – Em – C – G – D. Strumming: D-DU-UDU. Tempo 148 BPM (halb nehmen). Klassischer Rockklassiker!",
                "chords": ["G", "C", "D", "Em"], "tempo_bpm": 148,
                "strumming": "D-DU-UDU",
                "luna_speech": "Brown Eyed Girl! Ein zeitloser Klassiker. Alle Akkorde kennst du schon – jetzt verbinden wir sie!",
            },
            "chords": ["E", "A", "D", "G", "C"],
            "speech": {"de": "8 Akkorde – jetzt kannst du 90% aller Songs spielen!", "en": "8 chords – now you can play 90% of all songs!"},
        },

        {
            "id": "g_fingerpicking", "title": "Fingerpicking Grundlagen",
            "level": 2, "duration_min": 25, "xp_reward": 140,
            "topics": ["p-i-m-a Technik", "Travis Picking", "Arpeggio"],
            "technique": {
                "name": "Fingerpicking – p, i, m, a",
                "description": "Statt Plektrum: Daumen (p) spielt E-A-D, Zeigefinger (i) spielt G, Mittelfinger (m) spielt H, Ringfinger (a) spielt e. Das gibt einen vollen, warmen Klang. Arpeggio: die Töne eines Akkords nacheinander – wie eine Harfe. Travis-Picking: Daumen wechselt zwischen zwei Basssaiten, während andere Finger Melodie spielen.",
                "tips": ["Hand entspannt wie eine gebogene Kugel", "Finger nicht in die Luft nach dem Zupfen", "Daumen zieht nach unten, Finger ziehen nach oben"],
                "luna_speech": "Fingerpicking gibt deinem Spiel eine völlig andere Tiefe. Jede einzelne Note kann atmen. Das ist echter Ausdruck!",
            },
            "exercise": {
                "name": "p-i-m-a Arpeggio auf Am",
                "description": "Am halten, spielen: A-Saite(p) – G-Saite(i) – H-Saite(m) – e-Saite(a). Dann rückwärts. Dann Am – G – C – D mit diesem Muster. 50 BPM – sehr langsam beginnen!",
                "duration_sec": 300, "target_bpm": 50,
                "strumming": "p-i-m-a Arpeggio",
                "chords": ["Am", "G", "C", "D"],
                "luna_speech": "p-i-m-a: Daumen, Zeige, Mittel, Ring. Lass uns das Muster sehr langsam aufbauen – sauber ist wichtiger als schnell!",
            },
            "song": {
                "name": "Let Her Go (Passenger)",
                "description": "C – G – Am – F. Charakteristisches Arpeggio-Picking. Tempo 76 BPM. Eines der emotionalsten Songs mit dieser Technik.",
                "chords": ["C", "G", "Am", "D"], "tempo_bpm": 76,
                "strumming": "Arpeggio p-i-m-a",
                "luna_speech": "Let Her Go von Passenger! Dieses Arpeggio klingt so schön – dein Fingerpicking macht den Song!",
            },
            "chords": ["Am", "G", "C", "D"],
            "speech": {"de": "Fingerpicking – dein Spiel wird sofort tiefer und ausdrucksvoller.", "en": "Fingerpicking – your playing instantly becomes deeper and more expressive."},
        },

        {
            "id": "g_power_chords", "title": "Power Chords & Rockgitarre",
            "level": 2, "duration_min": 20, "xp_reward": 130,
            "topics": ["Power Chords Technik", "E5-A5 Form", "Palm Muting", "Rock-Rhythmus"],
            "technique": {
                "name": "Power Chords – der Rock-Klang",
                "description": "Power Chords (5er): Nur Grundton + Quinte, keine Terz. Das macht sie sowohl 'dur' als auch 'moll'-neutral – perfekt für Rock und Metal. E5: Finger 1 auf E-Saite Bund 1 (oder offen), Finger 3 auf A-Saite Bund 3. Nur 2-3 Saiten. Mit Verzerrung klingt das enorm fett. Palm Muting: Handflächenkante leicht auf die Saiten – gedämpfter, aggressiver Sound.",
                "tips": ["Power Chord Form bleibt immer gleich – nur Position ändert sich", "Palm Mute: Handfläche ganz leicht, nicht zu fest", "Mit Verzerrung (Distortion) klingen Power Chords am besten"],
                "luna_speech": "Power Chords sind das Fundament von Rock und Metal! Mit einer Form spielst du jeden Power Chord auf dem ganzen Hals.",
            },
            "exercise": {
                "name": "E5 – A5 – D5 – G5 Progression",
                "description": "E5 (offen) – A5 (offen) – D5 – G5. Strumming: 4× Down mit Palm Mute. Dann ohne Palm Mute. Metronom 80 BPM. Rock-Feeling!",
                "duration_sec": 240, "target_bpm": 80,
                "strumming": "4× Down, Palm Mute",
                "chords": ["E5", "A5", "D5", "G5"],
                "luna_speech": "Jetzt rocken wir! E5, A5, D5, G5 – mit Palm Mute klingt das sofort nach Rock!",
            },
            "song": {
                "name": "Smoke on the Water – Full Riff (Deep Purple)",
                "description": "G5 – Bb5 – C5 – G5 – Bb5 – Db5 – C5. Das ikonischste Riff der Rockgeschichte! Tempo 112 BPM.",
                "chords": ["G5", "C5", "E5", "A5"], "tempo_bpm": 112,
                "strumming": "Straight Down, hard",
                "luna_speech": "Das VOLLES Smoke on the Water Riff! Jetzt klingt es wie das Original. Rock!",
            },
            "chords": ["E5", "A5", "D5", "G5", "C5"],
            "speech": {"de": "Power Chords – eine Form, unendliche Songs!", "en": "Power chords – one shape, infinite songs!"},
        },

        {
            "id": "g_sus_add", "title": "Sus- und Add-Akkorde",
            "level": 2, "duration_min": 20, "xp_reward": 140,
            "topics": ["Asus2", "Dsus2", "Cadd9", "Gadd9", "Spannung & Auflösung"],
            "technique": {
                "name": "Schwebende Klänge – Sus und Add",
                "description": "Sus2: Die Terz wird durch die Sekunde ersetzt – klingt offen, unentschlossen. Asus2: nur A-D-E (Finger 2+3 weg lassen). Dsus2: D-A-e (Finger 1 weglassen). Add9: normaler Akkord + 9. Ton. Cadd9: C-Dur + D-Note. Gadd9: G-Dur + A-Note. Diese Akkorde klingen modern und cineastisch.",
                "tips": ["Sus-Akkorde wollen zu 'normalen' Akkorden aufgelöst werden", "Asus2 → Am: ganz einfach – nur Finger aufsetzen", "Cadd9: Finger 3 und 4 auf G+e-Saite Bund 3 – offen und voll"],
                "luna_speech": "Sus- und Add-Akkorde sind das Geheimnis moderner Gitarren-Sounds. Sie klingen frisch und interessant – hör den Unterschied!",
            },
            "exercise": {
                "name": "Sus-Auflösung & Add-Klänge",
                "description": "Asus2 → Am → Asus4 → Am (Auflösungsmuster). Dann Dsus2 → D. Dann Cadd9 → C → Gadd9 → G. Jeder Übergang zeigt die Spannung und Auflösung. 65 BPM.",
                "duration_sec": 300, "target_bpm": 65,
                "strumming": "D-DU langsam, Klang hören",
                "chords": ["Asus2", "Asus4", "Dsus2", "Cadd9", "Gadd9"],
                "luna_speech": "Hör wie die Sus-Akkorde Spannung aufbauen und sich dann in die normalen auflösen – das ist Musikgefühl!",
            },
            "song": {
                "name": "More Than Words (Extreme)",
                "description": "G – Cadd9 – Am – C – D – Em. Fingerpicking mit wunderschönen Cadd9 Klängen. Tempo 66 BPM.",
                "chords": ["Cadd9", "G", "Am", "D", "Em"], "tempo_bpm": 66,
                "strumming": "Fingerpicking p-i-m",
                "luna_speech": "More Than Words! Der Cadd9 macht diesen Song so besonders. Das klingt wunderschön.",
            },
            "chords": ["Asus2", "Asus4", "Dsus2", "Cadd9", "Gadd9"],
            "speech": {"de": "Sus und Add – der Sound moderner Gitarrenmusik!", "en": "Sus and Add – the sound of modern guitar music!"},
        },

        # ── LEVEL 3  ───────────────────────────────────────────────────────────
        {
            "id": "g_barre_f", "title": "Barré-Akkorde: F-Dur meistern",
            "level": 3, "duration_min": 35, "xp_reward": 200,
            "topics": ["F-Dur Barré", "Zeigefinger-Technik", "Barre-Aufbau 3 Stufen"],
            "technique": {
                "name": "F-Dur – Die Barré-Grundform",
                "description": "Barré: Der Zeigefinger drückt alle 6 Saiten gleichzeitig. Bei F-Dur auf Bund 1: Zeigefinger als Barré + A-Form (Finger 2-3-4 auf A-D-G). Der Schlüssel: Zeigefinger direkt HINTER dem Bund, leicht zur Seite gedreht (knöchlige Außenkante nutzen). Daumen senkrecht hinter dem Hals für maximalen Druck.",
                "tips": ["Erst nur die Barré üben – alle 6 Saiten klar", "Knöchlige Außenkante des Zeigefingers nutzen", "Nicht die ganzen 6 Saiten brauchen Druck – oft reicht die Außenkante"],
                "luna_speech": "Barré-Akkorde sind die größte Hürde für Anfänger – aber auch der größte Durchbruch! Wenn du F-Dur kannst, kannst du JEDEN Akkord auf dem Hals spielen.",
            },
            "exercise": {
                "name": "Barré-Aufbau: 3 Stufen",
                "description": "Stufe 1: Nur Zeigefinger-Barré auf Bund 1, alle 6 Saiten nacheinander zupfen. Stufe 2: F-Dur vollständig greifen, jede Saite testen. Stufe 3: Am – F – C – G Wechsel, 8 Schläge pro Akkord, 50 BPM. Täglich 5 Minuten – nach 2 Wochen sitzt es!",
                "duration_sec": 420, "target_bpm": 50,
                "strumming": "Langsam, 8 Schläge pro Akkord",
                "chords": ["F", "Bm", "Bb"],
                "luna_speech": "Stufe 1: Nur die Barré. Hörst du alle Saiten klar? Dann Stufe 2. Geh nicht weiter bis jede Stufe sitzt!",
            },
            "song": {
                "name": "Let It Be (Beatles)",
                "description": "C – G – Am – F. Das F ist der einzige Barré – aber er macht den Song komplett! Strumming: D-DU-UDU. Tempo 66 BPM.",
                "chords": ["F", "C", "G", "Am"], "tempo_bpm": 66,
                "strumming": "D-DU-UDU",
                "luna_speech": "Let It Be! Das F-Dur vervollständigt diesen wunderschönen Beatles-Song. Du hast die Barré-Hürde überwunden!",
            },
            "chords": ["F", "Bb", "C", "G"],
            "speech": {"de": "Barré – die größte Hürde, aber auch der größte Durchbruch!", "en": "Barré – the biggest hurdle, but also the biggest breakthrough!"},
        },

        {
            "id": "g_barre_moveable", "title": "Barré-Formen auf dem ganzen Hals",
            "level": 3, "duration_min": 30, "xp_reward": 200,
            "topics": ["E-Form Barré", "A-Form Barré", "Bm & F#m", "Halsnavigation"],
            "technique": {
                "name": "Bewegliche Barré-Formen",
                "description": "Die E-Form (wie offenes E, plus Barré) und A-Form (wie offenes A, plus Barré) funktionieren auf JEDEM Bund. E-Form auf Bund 2 = F#m. E-Form auf Bund 3 = Gm. A-Form auf Bund 2 = H-Moll (Bm). A-Form auf Bund 3 = Cm. Der Bund bestimmt den Grundton – der Ton der tiefsten gegriffenen Saite.",
                "tips": ["Bm (A-Form Bund 2): Finger 3+4 als Mini-Barré auf D-G-H", "F#m (E-Form Bund 2): wie F-Dur, einen Bund höher", "Hals-Navigation: E-A-D-G-A-H nach Bünden lernen"],
                "luna_speech": "Jetzt wird die Gitarre riesig! Mit E-Form und A-Form kannst du JEDEN Akkord auf dem ganzen Hals spielen – das ist revolutionär!",
            },
            "exercise": {
                "name": "Bm – F#m – A – E Progression",
                "description": "Bm (A-Form Bund 2) – F#m (E-Form Bund 2) – A – E. Je 4 Schläge, 55 BPM. Dann die ganze Progression einen Bund höher verschieben: Cm – Gm – Bb – F.",
                "duration_sec": 360, "target_bpm": 55,
                "strumming": "D-DU-UDU",
                "chords": ["Bm", "F#m", "A", "E"],
                "luna_speech": "Bm und F#m sind die neuen Waffen in deinem Arsenal! Ich höre ob die Barré sauber klingt.",
            },
            "song": {
                "name": "Hotel California – Intro (Eagles)",
                "description": "Bm – F# – A – E – G – D – Em – F#. Das legendäre Arpeggio-Intro. Tempo 74 BPM. Episch!",
                "chords": ["Bm", "F#m", "A", "E", "G", "D"], "tempo_bpm": 74,
                "strumming": "Arpeggio Intro-Muster",
                "luna_speech": "Hotel California! Das Arpeggio-Intro ist eines der bekanntesten der Welt. Mit deinen Barré-Akkorden klingt es nun episch!",
            },
            "chords": ["Bm", "F#m", "Bb", "Gm"],
            "speech": {"de": "E-Form + A-Form = der ganze Hals ist dein!", "en": "E-shape + A-shape = the whole neck is yours!"},
        },

        {
            "id": "g_seventh_chords", "title": "7-Akkorde & Blues-Feeling",
            "level": 3, "duration_min": 30, "xp_reward": 180,
            "topics": ["E7 A7 D7 G7", "Blues-Progression", "12-Takt-Blues"],
            "technique": {
                "name": "Dominantseptakkorde – der Blues-Klang",
                "description": "E7: wie E-Dur, aber Finger 4 auf G-Saite Bund 2 kommt dazu (oder Finger 3 auf G-Saite weglassen). A7: wie A-Dur ohne Finger 3 auf G (leer). D7: Finger 1 auf H-Bund 1, Finger 2 auf G-Bund 2, Finger 3 auf e-Bund 2. G7: wie G aber Finger 1 auf e-Saite Bund 1. Alle 7-Akkorde haben eine 'Spannung', die sich nach vorne zieht.",
                "tips": ["E7 vs E: ein Finger weniger – aber ein wärmerer Klang", "A7: G-Saite leer schwingen lassen!", "D7 klingt süßlich-jazztig – wunderschön"],
                "luna_speech": "7-Akkorde klingen so voll und warm! Sie sind der Grundbaustein von Blues, Jazz und Soul. Wenn du E7 und A7 hast, spielst du sofort Blues.",
            },
            "exercise": {
                "name": "12-Takt-Blues in E",
                "description": "E7 (4 Takte) – A7 (2 Takte) – E7 (2 Takte) – B7 (1 Takt) – A7 (1 Takt) – E7 (2 Takte). Der klassische 12-Takt-Blues! Shuffle-Rhythmus, 80 BPM.",
                "duration_sec": 360, "target_bpm": 80,
                "strumming": "Shuffle: D-U D-U",
                "chords": ["E7", "A7", "B7", "D7"],
                "luna_speech": "Der 12-Takt-Blues ist die Mutter aller modernen Musik! Rock, Jazz, Soul – alles kommt davon. Los!",
            },
            "song": {
                "name": "Sweet Home Chicago (Blues Standard)",
                "description": "E7 – A7 – E7 – B7 – A7 – E7. Shuffle-Rhythmus. Tempo 116 BPM. Klassischer Blues!",
                "chords": ["E7", "A7", "D7", "G7"], "tempo_bpm": 116,
                "strumming": "Blues Shuffle",
                "luna_speech": "Sweet Home Chicago! Ein echter Blues-Klassiker. Deine 7-Akkorde machen den Blues-Sound!",
            },
            "chords": ["E7", "A7", "D7", "G7", "B7"],
            "speech": {"de": "7-Akkorde – der Sound von Blues, Jazz und Soul!", "en": "7th chords – the sound of blues, jazz and soul!"},
        },

        {
            "id": "g_pentatonic", "title": "Pentatonik & erste Improvisation",
            "level": 3, "duration_min": 35, "xp_reward": 220,
            "topics": ["A-Moll Pentatonik Box 1", "Licks", "Improvisation über Blues"],
            "technique": {
                "name": "A-Moll Pentatonik – 5 Töne für alles",
                "description": "Die Pentatonik-Tonleiter hat nur 5 Töne und klingt über fast jeden Akkord gut. Am Pentatonik Box 1 (5. Bund): E-Saite Bünde 5-8, A-Saite 5-7, D-Saite 5-7, G-Saite 5-7, H-Saite 5-8, e-Saite 5-8. Diese Box ist die Basis der gesamten Rockgitarre. Licks: kurze melodische Muster aus dieser Box.",
                "tips": ["Box 1 auswendig lernen – dann ist Improvisation einfach", "Immer nur 1-2 Töne auf einer Saite in dieser Box", "Bend (Saite ziehen) auf H-Saite Bund 7 = Gitarren-Schrei"],
                "luna_speech": "Die Pentatonik ist das Geheimnis aller Gitarren-Soli! Jimi Hendrix, Eric Clapton, Carlos Santana – sie alle nutzen diese Box. Jetzt lernst du wie sie denken!",
            },
            "exercise": {
                "name": "Box 1 aufwärts und abwärts",
                "description": "Am Pentatonik Box 1 komplett auf/ab: e5-e8, A5-A7, D5-D7, G5-G7, H5-H8, e5-e8. Dann rückwärts. Metronom 65 BPM. Dann: einfache Licks üben (Bund 5-7-5 auf H-Saite).",
                "duration_sec": 360, "target_bpm": 65,
                "strumming": "Einzelne Töne, legato",
                "chords": ["Am", "Em"],
                "luna_speech": "Box 1 – aufwärts und abwärts bis sie sich wie zuhause anfühlt. Dann beginnt die Improvisation!",
            },
            "song": {
                "name": "Improvisation über Am Blues",
                "description": "Am – G – F – E. Endlos wiederholen. Du improvisierst mit Box 1 darüber! Tempo 75 BPM. Kein 'falscher' Ton in der Pentatonik.",
                "chords": ["Am", "G", "F", "E"], "tempo_bpm": 75,
                "strumming": "Improvisation – du entscheidest",
                "luna_speech": "Jetzt improvisierst du! Kein falscher Ton in der Pentatonik – spiel was du fühlst. Das ist echter Ausdruck!",
            },
            "chords": ["Am", "Em", "Dm"],
            "speech": {"de": "Pentatonik – 5 Töne für unendliche Musik!", "en": "Pentatonic – 5 notes for infinite music!"},
        },

        # ── LEVEL 4  ───────────────────────────────────────────────────────────
        {
            "id": "g_jazz_voicings", "title": "Jazz-Akkorde & Voicings",
            "level": 4, "duration_min": 35, "xp_reward": 250,
            "topics": ["Am7 Em7 Dm7", "Cmaj7 Gmaj7", "ii-V-I Kadenz", "Jazz-Rhythmus"],
            "technique": {
                "name": "Jazz-Voicings – Erweiterte Akkordklänge",
                "description": "Moll-Sept: Am7 (A-Dur ohne Finger 3, G-Saite offen). Em7 (Em ohne Finger 3). Dm7 (Dm mit kleinem Finger auf e-Saite Bund 1). Major7: Cmaj7 (C ohne G-Saite, B-Saite offen). Gmaj7 (G mit Finger 1 auf e-Bund 2). Die ii-V-I Kadenz ist das Fundament des Jazz: Dm7 → G7 → Cmaj7.",
                "tips": ["m7-Akkorde: oft ein Finger weniger als m-Grundform", "maj7 klingt schwebend und modern", "ii-V-I: probiere viele Inversionen und Positionen"],
                "luna_speech": "Jazz-Voicings klingen auf einmal voll und sophisticated! Diese Akkorde hörst du in Jazz, Bossa Nova und modernem Pop.",
            },
            "exercise": {
                "name": "ii-V-I in C, G und D",
                "description": "Dm7 – G7 – Cmaj7 (in C). Em7 – A7 – Dmaj7 (in D). Am7 – D7 – Gmaj7 (in G). Je 4 Schläge, 70 BPM. Jazz-Komping-Rhythmus: Betonungen auf 2 und 4.",
                "duration_sec": 360, "target_bpm": 70,
                "strumming": "Jazz-Komp: Betonung 2+4",
                "chords": ["Am7", "Em7", "Dm7", "Cmaj7", "Gmaj7"],
                "luna_speech": "ii-V-I – das Fundament des Jazz! Diese Kadenz erkennst du jetzt in jedem Jazz-Standard.",
            },
            "song": {
                "name": "Autumn Leaves (Jazz Standard)",
                "description": "Am7 – D7 – Gmaj7 – Cmaj7 – F#m7b5 – B7 – Em. Der bekannteste Jazz-Standard. Tempo 100 BPM Swing.",
                "chords": ["Am7", "Dm7", "Cmaj7", "Gmaj7"], "tempo_bpm": 100,
                "strumming": "Jazz-Swing Komp",
                "luna_speech": "Autumn Leaves! Du spielst jetzt echten Jazz. Das ist ein riesiger Meilenstein!",
            },
            "chords": ["Am7", "Em7", "Dm7", "Cmaj7", "Gmaj7", "Fmaj7"],
            "speech": {"de": "Jazz-Voicings – die Musik wird sophistiziert!", "en": "Jazz voicings – the music becomes sophisticated!"},
        },

        {
            "id": "g_extended_scales", "title": "Dur-Tonleiter & Modi",
            "level": 4, "duration_min": 35, "xp_reward": 240,
            "topics": ["C-Dur Tonleiter auf Gitarre", "Dorisch", "Mixolydisch", "3-Noten-pro-Saite"],
            "technique": {
                "name": "Tonleitern – die Landkarte der Gitarre",
                "description": "Die C-Dur Tonleiter (C-D-E-F-G-A-H) lässt sich in 5 Positionen über den ganzen Hals spielen. 3-Noten-pro-Saite: schnelle Technik für Läufe. Modi entstehen wenn die Tonleiter ab einem anderen Ton gespielt wird: Dorisch (D-Modus) klingt jazzig-dunkel, Mixolydisch (G-Modus) klingt rockig-bluesig.",
                "tips": ["3-Noten-pro-Saite = gleicher Fingersatz auf jeder Saite", "Mixolydisch = Dur-Tonleiter mit kleiner 7 – Rock-Klang!", "Dorisch: kleines Moll mit großer 6 – Santana-Sound"],
                "luna_speech": "Tonleitern sind die Landkarte der Gitarre! Mit Modi verändert sich die Stimmung komplett – von rockig bis mystisch.",
            },
            "exercise": {
                "name": "C-Dur alle 5 Positionen",
                "description": "Position 1 (C auf A-Bund 3) bis Position 5 (C auf E-Bund 8). Jede Position aufwärts und abwärts, 70 BPM. Dann mixolydische Tonleiter ab G.",
                "duration_sec": 420, "target_bpm": 70,
                "strumming": "Einzelne Töne, gleichmäßig",
                "chords": ["C", "G", "Am", "F"],
                "luna_speech": "Alle 5 Positionen – dann bist du überall auf dem Hals zu Hause!",
            },
            "song": {
                "name": "Oye Como Va (Santana)",
                "description": "Am7 – D9 im Dorisch-Modus. Das charakteristische Santana-Feeling. Improvisation im Am-Dorisch. Tempo 138 BPM.",
                "chords": ["Am7", "Dm7", "Em7"], "tempo_bpm": 138,
                "strumming": "Latein-Rhythmus + Solo",
                "luna_speech": "Santana! Oye Como Va lebt vom Am-Dorisch Klang. Das ist der mystisch-lateinische Sound!",
            },
            "chords": ["Am7", "Dm7", "Em7"],
            "speech": {"de": "Modi – eine Tonleiter, viele Klang-Welten!", "en": "Modes – one scale, many sonic worlds!"},
        },

        {
            "id": "g_advanced_rhythm", "title": "Fortgeschrittene Rhythmik",
            "level": 4, "duration_min": 30, "xp_reward": 230,
            "topics": ["Synkopen", "16tel-Rhythmus", "Funk-Strum", "Ghost Notes"],
            "technique": {
                "name": "Funk-Rhythmus & Synkopen",
                "description": "Funk-Strumming: 16tel-Noten (4 pro Schlag), viele gedämpfte Ghost Notes dazwischen. Das Handgelenk bewegt sich ständig (down-up-down-up), aber nicht jede Bewegung trifft die Saiten. Ghost Notes: gedämpfte Saiten die den Rhythmus füllen ohne Ton. Synkopen: Betonungen ZWISCHEN den Hauptschlägen.",
                "tips": ["Handgelenk läuft immer – nur manchmal Saiten treffen", "Ghost Notes mit Zeige-/Mittelfinger leicht dämpfen", "Funk ist Groove – nicht laut, sondern präzise"],
                "luna_speech": "Funk-Rhythmus klingt unglaublich! Das Geheimnis: das Handgelenk hört NIE auf zu schwingen. Ghost Notes geben den Groove.",
            },
            "exercise": {
                "name": "16tel Funk-Pattern auf Em",
                "description": "Em halten, 16tel-Noten: D-d-U-u-D-d-U-u (Großbuchstabe = klingt, klein = gedämpft). Metronom 75 BPM. Dann mit Akkordwechseln Em7 – A7.",
                "duration_sec": 360, "target_bpm": 75,
                "strumming": "16tel Funk D-d-U-u",
                "chords": ["Em7", "A7", "D7"],
                "luna_speech": "Funk! Das Handgelenk läuft durch, aber nur die betonten Schläge klingen. Ich höre den Groove!",
            },
            "song": {
                "name": "Superstition (Stevie Wonder) – Gitarre",
                "description": "Ebm7 Funk-Riff. Alternativ in Em7. 16tel-Funk-Rhythmus. Tempo 100 BPM. Einer der berühmtesten Funk-Grooves!",
                "chords": ["Em7", "Am7", "Dm7"], "tempo_bpm": 100,
                "strumming": "Funk 16tel Groove",
                "luna_speech": "Superstition! Dieser Funk-Groove ist legendär. Mit deinen Ghost Notes klingt es jetzt wirklich nach Stevie Wonder!",
            },
            "chords": ["Em7", "Am7", "Dm7"],
            "speech": {"de": "Funk – Groove ist alles!", "en": "Funk – groove is everything!"},
        },

        # ── LEVEL 5  ───────────────────────────────────────────────────────────
        {
            "id": "g_advanced_solo", "title": "Lead-Gitarre & Solotechnik",
            "level": 5, "duration_min": 40, "xp_reward": 300,
            "topics": ["String Bending", "Vibrato", "Hammer-On/Pull-Off", "Legato-Läufe"],
            "technique": {
                "name": "Ausdruckstechniken für Soli",
                "description": "String Bending: Saite seitlich ziehen – H-Saite Bund 7 um einen Ganzton benden = klingt wie Bund 9. Vibrato: schnelles kleines Benden hin und her – gibt Tönen Leben. Hammer-On: Finger 1 spielt Ton, Finger 3 'hämmert' auf Bund 3 ohne Anschlag. Pull-Off: umgekehrt – Finger 3 zieht weg und Finger 1 klingt. Legato: viele Töne mit wenigen Anschlägen.",
                "tips": ["Bend: ganze Hand dreht sich – nicht nur Finger", "Vibrato: gleichmäßig und kontrolliert – nicht zu wild", "Hammer-On: fest und schnell aufsetzen"],
                "luna_speech": "Jetzt lernst du die Ausdruckssprache der Gitarre! Bending, Vibrato, Hammer-On – das sind die Emotionen des Gitarren-Soli.",
            },
            "exercise": {
                "name": "Bending & Vibrato Technik",
                "description": "H-Saite Bund 7: Bend auf Bund 9-Pitch. Dann mit Vibrato halten. Dann Hammer-On+Pull-Off Bund 5-7-5 auf H-Saite. Metronom 60 BPM – Qualität vor Geschwindigkeit!",
                "duration_sec": 420, "target_bpm": 60,
                "strumming": "Einzelne Töne, ausdrucksvoll",
                "chords": ["Am", "Em"],
                "luna_speech": "Bending ist das Herzstück des Rock-Soli! Der Ton soll klingen wie eine menschliche Stimme. Gefühl ist alles.",
            },
            "song": {
                "name": "Nothing Else Matters – Intro (Metallica)",
                "description": "Em – Am7 – C – D – Em. Fingerpicking + melodische Licks. Tempo 69 BPM. Emotionaler Klassiker der Gitarrengeschichte.",
                "chords": ["Em", "Am7", "C", "D"], "tempo_bpm": 69,
                "strumming": "Fingerpicking + Licks",
                "luna_speech": "Nothing Else Matters! Dieses Stück vereint Fingerpicking und Melodie perfekt. Das ist pure Emotion auf der Gitarre.",
            },
            "chords": ["Am", "Em", "Dm"],
            "speech": {"de": "Bending & Vibrato – die Stimme der Gitarre!", "en": "Bending & vibrato – the voice of the guitar!"},
        },

        {
            "id": "g_composition", "title": "Eigene Songs schreiben",
            "level": 5, "duration_min": 40, "xp_reward": 300,
            "topics": ["Akkord-Progressionen", "Melodie entwickeln", "Songstruktur", "Verse-Chorus"],
            "technique": {
                "name": "Komposition – dein eigener Sound",
                "description": "Grundlegende Songstruktur: Intro – Verse – Pre-Chorus – Chorus – Bridge – Outro. Akkord-Progressionen: I-V-vi-IV (die 'vier Akkorde der Popmusik' – C-G-Am-F) klingt in hunderten Hits. Melodik: beginne auf der 3. oder 5. Stufe, erzähl eine Geschichte. Kontrast: Verse ruhiger, Chorus energetischer.",
                "tips": ["Einfach anfangen: 4 Akkorde, simples Muster", "Melodie summen, dann auf Gitarre übertragen", "Notizbuch führen – Ideen aufschreiben sofort"],
                "luna_speech": "Komposition ist der höchste Ausdruck eines Musikers! Dein eigener Song ist mehr wert als hundert fremde. Ich begleite dich dabei.",
            },
            "exercise": {
                "name": "Eigene Progression entwickeln",
                "description": "Wähle 4 Akkorde aus deinen Favoriten. Spiele sie in verschiedenen Reihenfolgen bis du etwas findest das dir gefällt. Füge dann eine einfache Melodie auf der hohen e-Saite hinzu. Kein Metronom – freies Experimentieren!",
                "duration_sec": 600, "target_bpm": 0,
                "strumming": "Frei – was sich gut anfühlt",
                "chords": ["Am", "F", "C", "G"],
                "luna_speech": "Jetzt bist du der Komponist! Keine richtigen oder falschen Akkorde – nur was dir gefällt. Ich höre zu.",
            },
            "song": {
                "name": "Dein eigener Song",
                "description": "Spiele die Progression die du entwickelt hast. Füge eine einfache Melodie aus der Pentatonik dazu. Das ist deine Musik!",
                "chords": ["Am", "F", "C", "G"], "tempo_bpm": 75,
                "strumming": "Dein eigenes Pattern",
                "luna_speech": "Das ist DEIN Song! Jede Note, jeder Akkord – du hast das erschaffen. Ich bin so stolz auf dich!",
            },
            "chords": ["Am", "F", "C", "G"],
            "speech": {"de": "Komposition – jetzt bist du ein echter Musiker!", "en": "Composition – now you are a real musician!"},
        },

        {
            "id": "g_master", "title": "Meisterstück: Komplettsong",
            "level": 5, "duration_min": 45, "xp_reward": 350,
            "topics": ["Alle Techniken", "Vollständiger Song", "Interpretation", "Ausdruck"],
            "technique": {
                "name": "Alles zusammenbringen",
                "description": "Ein professioneller Song kombiniert: Intro (Arpeggio), Verse (Fingerpicking oder ruhiges Strumming), Pre-Chorus (aufbauende Energie), Chorus (volles Strumming, alle Saiten), Bridge (harmonische Überraschung), Solo (Pentatonik + Licks), Outro (ruhig ausklingen). Dynamik: Lautstärke und Intensität variieren.",
                "tips": ["Dynamik ist das wichtigste – nicht immer volle Kraft", "Verse und Chorus klar differenzieren", "Das Gefühl des Songs ist wichtiger als perfekte Technik"],
                "luna_speech": "Das ist dein Meisterstück! Alles was du gelernt hast – Haltung, Akkorde, Strumming, Fingerpicking, Barré, Soli – fließt jetzt zusammen.",
            },
            "exercise": {
                "name": "Blackbird (Beatles) komplett",
                "description": "G – Am7 – G/B – C – A/C# – D – D7sus4 – G. Fingerpicking-Melodie mit Basslauf. Tempo 94 BPM. McCartney's Meisterwerk kombiniert alles!",
                "duration_sec": 600, "target_bpm": 94,
                "strumming": "Fingerpicking + Melodie",
                "chords": ["G", "Am7", "C", "D7"],
                "luna_speech": "Blackbird ist das perfekte Meisterstück – Melodie, Bass, Rhythmus alles in einer Gitarre! Du bist bereit dafür.",
            },
            "song": {
                "name": "Stairway to Heaven – Intro (Led Zeppelin)",
                "description": "Am – G/H – C – D – Fmaj7 – G – Am. Das ikonischste Gitarren-Intro. Fingerpicking + Arpeggio. Tempo 76 BPM.",
                "chords": ["Am", "G", "Fmaj7", "C", "D"], "tempo_bpm": 76,
                "strumming": "Klassisches Arpeggio-Picking",
                "luna_speech": "Stairway to Heaven! Dieses Intro hat Millionen Menschen dazu gebracht, Gitarre zu lernen. Du spielst es jetzt – das ist dein Meisterstück!",
            },
            "chords": ["Am", "G", "Fmaj7", "C", "D"],
            "speech": {"de": "Das Meisterstück – alles zusammenbringen!", "en": "The masterpiece – bringing it all together!"},
        },

        # ── SPANISH / LATIN FLAIR – Optional ab Level 3 ──────────────────────
        {
            "id": "g_flamenco_intro",
            "title": "🎸 Flamenco – Die Seele Spaniens",
            "level": 3, "duration_min": 30, "xp_reward": 220,
            "category": "Spanish",  # ← Neue Kategorie
            "unlock_level": 3,      # ← Freigeschaltet ab Level 3
            "topics": ["Phrygische Kadenz", "Rasgueado-Technik", "Picado", "Flamenco-Rhythmus (Compás)"],
            "technique": {
                "name": "Phrygische Kadenz & Rasgueado",
                "description": (
                    "Die phrygische Kadenz Am-G-F-E ist das Herzstück des Flamenco. "
                    "E-Phrygisch klingt sofort spanisch, weil der Halbtonschritt F→E eine "
                    "uralte arabisch-andalusische Spannung erzeugt. "
                    "Rasgueado: Die vier Finger schnell nacheinander fächern (kleiner→Ring→Mittel→Zeige), "
                    "dann Daumen zurück – ergibt einen rollenden, kraftvollen Sound. "
                    "Picado: Wechsel zwischen Zeige- und Mittelfinger beim Einzelton-Spiel, "
                    "ähnlich dem Alternate-Picking, aber mit den Fingern statt dem Plektrum."
                ),
                "tips": [
                    "Phrygisch: Am – G – F – E und auf E den Rhythmus betonen",
                    "Rasgueado: kleiner Finger zuerst, sehr schnell, klingt wie ein Strumming",
                    "Flamenco-Compás: 12er-Rhythmus, Betonungen auf 3-6-8-10-12",
                    "Haltung: Daumen nicht über den Hals – klassische Gitarrenhaltung",
                ],
                "luna_speech": "Flamenco ist Leidenschaft und Präzision in einem! Die phrygische Kadenz ist wie Magie – hör wie sofort dieser spanische Klang entsteht!",
            },
            "exercise": {
                "name": "Phrygische Kadenz + Rasgueado",
                "description": (
                    "Schritt 1: Am – G – F – E langsam im Fingerpicking (60 BPM). "
                    "Schritt 2: Am – G – F – E mit einfachem D-Anschlag (80 BPM). "
                    "Schritt 3: Rasgueado auf E üben – 4-Finger-Fächer. "
                    "Schritt 4: Komplette Kadenz mit Rasgueado auf E (Höhepunkt)."
                ),
                "duration_sec": 480, "target_bpm": 80,
                "strumming": "D-U-DU + Rasgueado",
                "chords": ["Am_Flam", "G_Flam", "F_Maj", "E_Phryg"],
                "luna_speech": "Rasgueado klingt schwer, ist es aber nicht! Fächere die Finger einfach schnell – erst langsam, dann immer schneller. Du klingst schon wie ein echter Flamenco-Gitarrist!",
            },
            "song": {
                "name": "Malagueña (Flamenco-Klassiker)",
                "description": (
                    "Eines der bekanntesten Flamenco-Stücke. "
                    "Am – G – F – E Phrygisch im Wechsel. "
                    "Erst Arpeggio (Fingerpicking), dann Rasgueado. Tempo 80 BPM."
                ),
                "chords": ["Am_Flam", "G_Flam", "F_Maj", "E_Phryg"],
                "tempo_bpm": 80,
                "strumming": "Arpeggio + Rasgueado",
                "luna_speech": "Malagueña! Das klingt jetzt wie echtes Andalusien! Der E-Phrygisch-Akkord am Ende – das ist der Seufzer des Flamenco.",
            },
            "chords": ["Am_Flam", "G_Flam", "F_Maj", "E_Phryg"],
            "speech": {
                "de": "Flamenco – Leidenschaft, Rhythmus und die Seele Spaniens!",
                "en": "Flamenco – passion, rhythm and the soul of Spain!",
                "es": "¡Flamenco – pasión, ritmo y el alma de España!",
            },
        },

        {
            "id": "g_latin_bossa",
            "title": "🌴 Bossa Nova – Das Herz Brasiliens",
            "level": 3, "duration_min": 30, "xp_reward": 210,
            "category": "Latin",
            "unlock_level": 3,
            "topics": ["Bossa-Nova Rhythmus", "Samba-Strum", "Am7-D9 Vamp", "Joao Gilberto Stil"],
            "technique": {
                "name": "Bossa-Nova: Synkopierter Bass + Akkord",
                "description": (
                    "Bossa Nova (portugiesisch: 'neue Welle') entstand in den 1950ern in Brasilien. "
                    "João Gilberto's Gitarrentechnik: Daumen spielt den Bass (E/A-Saite) "
                    "unabhängig von den Fingern die den Akkord strummen. "
                    "Das typische Muster: Bass – (Pause) – Akkord – (Pause) – Bass – Akkord-Akkord. "
                    "Harmonik: reich an 9er, 7er und maj7-Akkorden die einen sanften, "
                    "romantischen Klang erzeugen. Tempo: entspannt, ca. 60-80 BPM."
                ),
                "tips": [
                    "Daumen (p) spielt Bass unabhängig – übe erst ohne Akkorde",
                    "Am7 – D9 Vamp ist die Basis von hunderten Bossa-Songs",
                    "Nie zu laut spielen – Bossa ist Finesse, nicht Kraft",
                    "Fingernägel leicht stehen lassen für warmen Ton",
                ],
                "luna_speech": "Bossa Nova ist wie ein warmer Sommerabend in Rio! Der Trick ist, dass Daumen und Finger völlig unabhängig arbeiten – ich zeige dir wie!",
            },
            "exercise": {
                "name": "João Gilberto Bass+Chord Pattern",
                "description": (
                    "Schritt 1: Nur Daumen auf E-Saite: ♩ – ♩ – (60 BPM). "
                    "Schritt 2: Finger auf Am7, Daumen auf A-Bass: Bass-Chord-Bass-Chord. "
                    "Schritt 3: Am7 – D9 Wechsel mit Bossa-Pattern. "
                    "Schritt 4: Volle Bossa-Begleitung Am7 – D9 – G7 – Cmaj9."
                ),
                "duration_sec": 420, "target_bpm": 65,
                "strumming": "Bossa p-i-m Daumen+Finger",
                "chords": ["Am7_Lat", "D9", "G7_Lat", "Cmaj9"],
                "luna_speech": "Jetzt klingt es nach Rio! Daumen und Finger arbeiten wie zwei unabhängige Musiker – das ist das Geheimnis der Bossa Nova.",
            },
            "song": {
                "name": "The Girl from Ipanema (Tom Jobim)",
                "description": (
                    "Eines der meistgespielten Stücke der Welt. "
                    "Fmaj9 – G7 – Gm7 – Gb7 – Fmaj9. Bossa-Nova Begleitung. "
                    "Tempo 68 BPM – entspannt und fließend."
                ),
                "chords": ["Fmaj9", "G7_Lat", "Am7_Lat", "Cmaj9"],
                "tempo_bpm": 68,
                "strumming": "Bossa Nova Begleitung",
                "luna_speech": "The Girl from Ipanema! Dieses Stück hat die Welt verliebt in Bossa Nova gemacht. Deine Begleitung klingt wunderschön!",
            },
            "chords": ["Am7_Lat", "D9", "G7_Lat", "Cmaj9"],
            "speech": {
                "de": "Bossa Nova – Sanft, Sophisticated, Brasilianisch!",
                "en": "Bossa Nova – Soft, sophisticated, Brazilian!",
                "es": "Bossa Nova – ¡Suave, sofisticada, brasileña!",
            },
        },

        {
            "id": "g_latin_rhythm",
            "title": "🥁 Latin Rhythmus – Rumba & Son",
            "level": 4, "duration_min": 35, "xp_reward": 250,
            "category": "Latin",
            "unlock_level": 4,
            "topics": ["Clave-Rhythmus (3-2 und 2-3)", "Rumba-Begleitung", "Son Cubano", "Guajira"],
            "technique": {
                "name": "Clave – der Herzschlag der Latin-Musik",
                "description": (
                    "Die Clave (spanisch: Schlüssel) ist das rhythmische Fundament aller "
                    "kubanischen Musik. 3-2 Clave: ♩.♩.♩ | ♩♩ (drei Schläge dann zwei). "
                    "Auf der Gitarre wird die Clave durch asymmetrisches Strumming gespielt: "
                    "nicht jeder Schlag klingt, aber jeder Schlag IST im Handgelenk. "
                    "Rumba-Strum: Bass(Daumen) – Strum – Bass – Strum-Strum in synkopiertem Muster. "
                    "Son Cubano: einfacheres Grundmuster, Basis von Salsa."
                ),
                "tips": [
                    "Clave zuerst klatschen – dann auf Gitarre übertragen",
                    "Handgelenk hört nie auf zu schwingen – nur manchmal trifft es die Saiten",
                    "Latin braucht einen lockeren Anschlag – nie verkrampft",
                    "G – C – D in Son Cubano klingt sofort nach Kuba",
                ],
                "luna_speech": "Die Clave ist wie ein magischer Rhythmus-Code! Wenn du sie erstmal hörst und fühlst, wirst du sie überall erkennen – in Salsa, Rumba, Mambo!",
            },
            "exercise": {
                "name": "Clave-Strum auf Am – D – G",
                "description": (
                    "Schritt 1: 3-2 Clave klatschen: 1–(2)–3–(4)–5 | 1–(2)–3–(4)–(5). "
                    "Schritt 2: Am mit Rumba-Strum: Bass-D-d-U (Daumen zuerst). "
                    "Schritt 3: Am – Dm – E Wechsel mit Clave-Rhythmus. "
                    "Schritt 4: Guantanamera Begleitung G – C – D."
                ),
                "duration_sec": 480, "target_bpm": 90,
                "strumming": "Latin Clave Bass-D-d-U",
                "chords": ["Am_Flam", "Dm_Flam", "E7_Span", "G7_Lat"],
                "luna_speech": "Jetzt klingt es nach echtem Kuba! Die Clave gibt deiner Begleitung diesen unwiderstehlichen Groove – das ist die Magie des Son Cubano.",
            },
            "song": {
                "name": "Guantanamera (Traditional) + Chan Chan",
                "description": (
                    "Guantanamera: G – C – D – G Clave-Rhythmus 100 BPM. "
                    "Eines der berühmtesten Lieder der Welt. "
                    "Danach: Chan Chan (Buena Vista) – Am – Dm – E Rumba-Strum."
                ),
                "chords": ["G", "Am7_Lat", "G7_Lat", "Dm_Flam"],
                "tempo_bpm": 100,
                "strumming": "Son Cubano Clave",
                "luna_speech": "Guantanamera! Diese Melodie kennt die ganze Welt. Mit deinem Clave-Rhythmus klingt es jetzt wie ein Strand in Havanna!",
            },
            "chords": ["Am_Flam", "Dm_Flam", "E7_Span", "G7_Lat"],
            "speech": {
                "de": "Latin Rhythmus – die Clave macht alles lebendig!",
                "en": "Latin rhythm – the clave brings everything to life!",
                "es": "¡Ritmo latino – la clave da vida a todo!",
            },
        },

        {
            "id": "g_spanish_advanced",
            "title": "💃 Flamenco Avanzado – Picado & Alzapúa",
            "level": 5, "duration_min": 40, "xp_reward": 300,
            "category": "Spanish",
            "unlock_level": 5,
            "topics": ["Picado (Einzelton-Technik)", "Alzapúa (Daumen-Technik)", "Falseta", "Compás de Soleares"],
            "technique": {
                "name": "Picado, Alzapúa und die Falseta",
                "description": (
                    "Picado: Wechsel i-m (Zeige-Mittel) für schnelle Melodiläufe – "
                    "der 'Legato-Lauf' des Flamenco. Klang: präzise, klar, schnell. "
                    "Alzapúa: Daumen spielt Einzelton UND Akkord in einer Bewegung – "
                    "zuerst nach unten (Einzelton), dann 'Aufschlag' über mehrere Saiten. "
                    "Falseta: eine kurze melodische Phrase die man über dem Compás spielt, "
                    "quasi das Solo im Flamenco. "
                    "Soleares-Compás: 12er-Takt mit Betonungen 3-6-8-10-12 – "
                    "das älteste und bedeutendste Flamenco-Rhythmus."
                ),
                "tips": [
                    "Picado: i-m wechseln so schnell wie Alternate-Picking",
                    "Alzapúa: Daumen kräftig und kontrolliert – nicht verkrampfen",
                    "Falseta zuerst langsam – jede Note muss sitzen",
                    "Soleares zählen: 1-2-3-4-5-6-7-8-9-10-11-12, Betonung auf 12",
                ],
                "luna_speech": "Jetzt betreten wir die Welt des echten Flamenco! Picado und Alzapúa sind die Ausdrucksmittel der großen Meister – Paco de Lucía, Vicente Amigo. Das hier ist fortgeschrittene Kunst!",
            },
            "exercise": {
                "name": "Falseta in Am-Phrygisch",
                "description": (
                    "Schritt 1: Picado-Übung auf e-Saite: i-m Wechsel chromatisch (60 BPM). "
                    "Schritt 2: Einfache Falseta: E→F→E→D→C→H→A auf e-Saite (70 BPM). "
                    "Schritt 3: Alzapúa auf E-Phrygisch-Akkord: Daumen einzeln dann Aufschlag. "
                    "Schritt 4: Komplette Phrase: Falseta + Rasgueado-Abschluss."
                ),
                "duration_sec": 600, "target_bpm": 70,
                "strumming": "Picado i-m + Rasgueado + Alzapúa",
                "chords": ["E_Phryg", "Am_Flam", "F_Maj", "G_Flam"],
                "luna_speech": "Das ist echte Flamenco-Technik! Jede Note in der Falseta erzählt eine Geschichte. Dein Picado wird immer sauberer – ich höre es!",
            },
            "song": {
                "name": "Soleares (Klassische Flamenco-Form)",
                "description": (
                    "Die Mutter aller Flamenco-Formen. "
                    "Am – G – F – E Phrygisch im 12er-Compás. "
                    "Eigene Falseta + Rasgueado + Alzapúa kombinieren. "
                    "Tempo 72 BPM – Würde und Kraft."
                ),
                "chords": ["Am_Flam", "G_Flam", "F_Maj", "E_Phryg"],
                "tempo_bpm": 72,
                "strumming": "Soleares Compás 12er",
                "luna_speech": "Soleares – die tiefste und älteste Form des Flamenco. Das ist nicht nur Gitarre spielen – das ist eine Kunstform. Du machst es wunderschön!",
            },
            "chords": ["E_Phryg", "Am_Flam", "F_Maj", "G_Flam"],
            "speech": {
                "de": "Flamenco Avanzado – du spielst jetzt auf dem Niveau der Meister!",
                "en": "Flamenco Advanced – you're playing at master level now!",
                "es": "¡Flamenco avanzado – ya tocas al nivel de los maestros!",
            },
        },

    ],  # Ende guitar

# ══════════════════════════════════════════════════════════════════════════════
#  KLAVIER  –  15 Lektionen  ·  Level 1–5
#  Logische Progression:
#    Lv1 (1–4):  Tastatur, C-Dur Tonleiter, Haltung, Erste Dreiklänge
#    Lv2 (5–8):  Alberti-Bass, Koordination, Songs, Volksmusik
#    Lv3 (9–11): Dur/Moll-Tonleitern, Harmonielehre, Septakkorde
#    Lv4 (12–13): Jazz, Improvisation, erweiterte Harmonik
#    Lv5 (14–15): Klassisches Stück, Komposition
# ══════════════════════════════════════════════════════════════════════════════
    "piano": [

        # ── LEVEL 1  ───────────────────────────────────────────────────────────
        {
            "id": "p_intro", "title": "Das Klavier kennenlernen",
            "level": 1, "duration_min": 10, "xp_reward": 50,
            "topics": ["Tastatur-Layout", "Noten C-D-E-F-G-A-H", "Finger 1-5", "Körperhaltung"],
            "technique": {
                "name": "Tastatur, Finger & Haltung",
                "description": "Die Tastatur: weiße und schwarze Tasten in einem Muster. C ist immer LINKS von zwei schwarzen Tasten. Dieses Muster wiederholt sich über alle Oktaven. Fingernummern: Daumen=1, Zeigefinger=2, Mittelfinger=3, Ringfinger=4, kleiner Finger=5. Haltung: aufrecht sitzen, Arme locker, Handgelenk auf Tastaturniveau.",
                "tips": ["Finger leicht gebogen – wie eine Kugel halten", "Handgelenk locker und leicht erhöht", "Auf Fingerkuppen spielen – nicht flach"],
                "luna_speech": "Willkommen am Klavier! Das Tastatur-Muster wiederholt sich über alle 88 Tasten. Wenn du es einmal verstehst, kannst du dich überall orientieren.",
            },
            "exercise": {
                "name": "C-Position Fünftonraum",
                "description": "Daumen (1) auf C4 (mittleres C). Spiele C-D-E-F-G mit Finger 1-2-3-4-5, dann rückwärts G-F-E-D-C. Beide Hände nacheinander. 10 Wiederholungen langsam, 60 BPM.",
                "duration_sec": 120, "target_bpm": 60,
                "strumming": "Legato, gleichmäßig",
                "chords": [],
                "luna_speech": "C-D-E-F-G mit 1-2-3-4-5. Jede Note gleich lang – das trainiert gleichmäßigen Anschlag!",
            },
            "song": {
                "name": "Ode an die Freude (Beethoven) – Rechte Hand",
                "description": "E-E-F-G-G-F-E-D-C-C-D-E-E-D-D. Rechte Hand, C-Position, Tempo 65 BPM. Dein erster echter Klassiker!",
                "chords": [], "tempo_bpm": 65,
                "strumming": "Rechte Hand, Legato",
                "luna_speech": "Beethoven! Ode an die Freude – dein allererster Klassiker. Das ist ein echter Meilenstein!",
            },
            "chords": [],
            "speech": {"de": "C ist immer links von zwei schwarzen Tasten – das ist deine Orientierung!", "en": "C is always left of two black keys – that's your orientation!"},
        },

        {
            "id": "p_cscale", "title": "C-Dur Tonleiter & Daumenuntersatz",
            "level": 1, "duration_min": 15, "xp_reward": 80,
            "topics": ["C-D-E-F-G-A-H-C", "Fingersatz 1-2-3-1-2-3-4-5", "Daumenuntersatz"],
            "technique": {
                "name": "Daumenuntersatz – die wichtigste Technik",
                "description": "Die C-Dur Tonleiter: C-D-E-F-G-A-H-C. Fingersatz rechts: 1-2-3-DANN-1-2-3-4-5. Der Daumen schwingt UNTER Finger 3 durch (nicht drüber). Das Handgelenk bleibt auf gleicher Höhe – kein Hochziehen. Links: 5-4-3-2-1-DANN-3-2-1.",
                "tips": ["Daumen schwingt UNTER – nicht drüber kratzen", "Handgelenk gleichbleibend hoch während des Untersatzes", "Erst rechts allein üben, dann links, dann beide"],
                "luna_speech": "Der Daumenuntersatz ist die Grundlage aller Tonleitern und Läufe! Wenn du ihn beherrschst, öffnet sich die ganze Klaviatur.",
            },
            "exercise": {
                "name": "C-Dur Tonleiter Hoch und Runter",
                "description": "Rechts: C-D-E-F-G-A-H-C und zurück, 10 Mal. Dann links: C-D-E-F-G-A-H-C. Metronom: 60→75→90 BPM steigern wenn sauber.",
                "duration_sec": 300, "target_bpm": 70,
                "strumming": "Legato, gleichmäßig",
                "chords": ["C-Dur"],
                "luna_speech": "Erst rechts bis es automatisch läuft, dann links, dann beide zusammen. Ich höre ob jede Note gleich klingt!",
            },
            "song": {
                "name": "Für Elise – Intro (Beethoven)",
                "description": "E-D#-E-D#-E-H-D-C-A. Rechte Hand, langsam. Tempo 55 BPM. Das berühmteste Klavierintro der Welt!",
                "chords": ["A-Moll"], "tempo_bpm": 55,
                "strumming": "Legato, ausdrucksvoll",
                "luna_speech": "Für Elise! Du spielst gerade das bekannteste Klavierstück der Welt. Das klingt wunderschön!",
            },
            "chords": ["C-Dur", "A-Moll"],
            "speech": {"de": "Der Daumenuntersatz – die Schlüsseltechnik für alle Tonleitern!", "en": "Thumb under – the key technique for all scales!"},
        },

        {
            "id": "p_first_chords", "title": "Erste Dreiklänge",
            "level": 1, "duration_min": 20, "xp_reward": 100,
            "topics": ["C-Dur", "A-Moll", "G-Dur", "F-Dur", "Akkord-Aufbau"],
            "technique": {
                "name": "Dreiklänge – Grundton, Terz, Quinte",
                "description": "Dreiklang = 3 Töne gleichzeitig: Grundton + Terz + Quinte. C-Dur: C-E-G mit Finger 1-3-5. Am: A-C-E mit 1-3-5. G-Dur: G-H-D. F-Dur: F-A-C. Dur klingt hell und froh, Moll klingt dunkel und traurig – das liegt an der Terz! Alle Töne GLEICHZEITIG anschlagen.",
                "tips": ["Alle drei Finger gleichzeitig aufsetzen, dann drücken", "Finger vorher positionieren – erst dann anschlagen", "Hände beobachten: wölbt sich der Handrücken schön?"],
                "luna_speech": "Dreiklänge sind die Bausteine aller Musik! Mit drei Tönen klingt es plötzlich wie ein richtiges Stück. Hör den Unterschied zwischen Dur und Moll!",
            },
            "exercise": {
                "name": "C – Am – F – G Progression",
                "description": "Rechts: C-Dur (1-3-5), Am (1-3-5), F-Dur (1-3-5), G-Dur (1-3-5). Je 4 Schläge, 60 BPM. Diese Progression ist in hunderten Pop-Songs!",
                "duration_sec": 240, "target_bpm": 60,
                "strumming": "Alle 3 Töne gleichzeitig",
                "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"],
                "luna_speech": "C-Am-F-G! Diese Progression erkennst du überall. Ich höre ob alle drei Töne gleichzeitig klingen!",
            },
            "song": {
                "name": "Let It Be (Beatles) – Einfach",
                "description": "C – G – Am – F. Rechts: Dreiklänge. Links: Grundton als Bassnote. Tempo 63 BPM. Ein Welthit mit deinen vier Akkorden!",
                "chords": ["C-Dur", "G-Dur", "A-Moll", "F-Dur"], "tempo_bpm": 63,
                "strumming": "Rechts: Dreiklang, Links: Bassnote",
                "luna_speech": "Let It Be von den Beatles! Diese vier Akkorde sind in hunderten Songs. Du hast das Fundament der modernen Musik!",
            },
            "chords": ["C-Dur", "A-Moll", "G-Dur", "F-Dur"],
            "speech": {"de": "Drei Töne – Grundton, Terz, Quinte – das ist alles was du für Dreiklänge brauchst!", "en": "Three notes – root, third, fifth – that's all you need for triads!"},
        },

        {
            "id": "p_legato_staccato", "title": "Legato & Staccato – Ausdrucksmittel",
            "level": 1, "duration_min": 20, "xp_reward": 110,
            "topics": ["Legato binden", "Staccato kürzen", "Dynamik p/f", "Phrasierung"],
            "technique": {
                "name": "Artikulation – wie Töne klingen",
                "description": "Legato: Töne nahtlos verbinden – Finger bleibt bis zum nächsten Ton. Staccato: Ton sofort nach Anschlag loslassen – kurz und knapp. Dynamik: p (piano) = leise, f (forte) = laut, mp = mittel-leise, mf = mittel-laut, cresc. = wird lauter, decresc. = wird leiser. Diese Mittel erzählen Geschichten!",
                "tips": ["Legato: jeder neue Finger landet BEVOR der alte abhebt", "Staccato: Handgelenk federt leicht nach oben", "Dynamik kommt aus dem Arm-Gewicht, nicht dem Finger-Druck"],
                "luna_speech": "Legato und Staccato sind die Sprache der Musik! Ohne Artikulation klingt alles gleich – mit ihr erzählst du Geschichten.",
            },
            "exercise": {
                "name": "C-Dur Tonleiter mit wechselnder Artikulation",
                "description": "Aufwärts Legato, abwärts Staccato. Dann: erste Hälfte p (leise), zweite Hälfte f (laut). Metronom 65 BPM. Höre den Unterschied aktiv!",
                "duration_sec": 240, "target_bpm": 65,
                "strumming": "Legato dann Staccato",
                "chords": ["C-Dur", "A-Moll"],
                "luna_speech": "Hör den Unterschied! Legato fließt wie Wasser, Staccato perlt wie Regen. Beide sind wunderschön.",
            },
            "song": {
                "name": "Minuet in G (Bach) – vereinfacht",
                "description": "G-Dur, 3/4 Takt. Legato-Melodie in der rechten Hand. Tempo 120 BPM. Bachs Minuet ist das perfekte Stück für Artikulation.",
                "chords": ["G-Dur", "D-Dur", "C-Dur", "A-Moll"], "tempo_bpm": 120,
                "strumming": "Rechts: Melodie Legato, Links: Akkorde",
                "luna_speech": "Bach! Das Minuet in G ist das schönste Stück für Artikulation. Dein Legato macht es atmen.",
            },
            "chords": ["G-Dur", "D-Dur", "C-Dur", "A-Moll"],
            "speech": {"de": "Legato und Staccato – die Sprache der Musik!", "en": "Legato and staccato – the language of music!"},
        },

        # ── LEVEL 2  ───────────────────────────────────────────────────────────
        {
            "id": "p_alberti", "title": "Alberti-Bass & beide Hände",
            "level": 2, "duration_min": 25, "xp_reward": 150,
            "topics": ["Alberti-Bass Muster", "Beide Hände koordinieren", "Linke Hand automatisch"],
            "technique": {
                "name": "Alberti-Bass – das Geheimnis der Klassik",
                "description": "Alberti-Bass: Gebrochener Dreiklang links: Grundton – Quinte – Terz – Quinte (1-5-3-5). C-Dur: C-G-E-G. Gibt einen fließenden Rhythmus ohne die rechte Hand zu stören. Das Ziel: die linke Hand läuft 'automatisch', die rechte spielt die Melodie. Erst beide Hände SEPARAT üben bis sie laufen, DANN zusammen!",
                "tips": ["Linke Hand allein üben bis sie ohne Nachdenken läuft", "Tempo langsam starten – 50 BPM, dann steigern", "Blick auf rechte Hand – linke fühlt sich die Wege"],
                "luna_speech": "Der Alberti-Bass ist der Klang der Klassik! Mozart und Beethoven liebten ihn. Das Geheimnis: linke Hand automatisch machen.",
            },
            "exercise": {
                "name": "Alberti-Bass Aufbau",
                "description": "Nur links: C-G-E-G, Am-E-C-E, F-C-A-C, G-D-H-D. Wiederholen bis automatisch, 55 BPM. Dann rechts C-Dur Arpeggio dazu. Dann zusammen.",
                "duration_sec": 360, "target_bpm": 60,
                "strumming": "Alberti-Bass Muster",
                "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"],
                "luna_speech": "Erst die linke Hand allein – wirklich automatisch! Dann erst die rechte Melodie dazu. Das ist der Schlüssel!",
            },
            "song": {
                "name": "Für Elise – Vollständig (Beethoven)",
                "description": "Rechts: e-d#-e-d#-e-h-d-c-a. Links: Am – E – Am – C – G – Am. Alberti-Begleitung. Tempo 58 BPM.",
                "chords": ["A-Moll", "E-Dur", "C-Dur", "G-Dur"], "tempo_bpm": 58,
                "strumming": "Rechts: Melodie, Links: Alberti",
                "luna_speech": "Für Elise vollständig – mit beiden Händen! Das ist das Beethoven-Gefühl!",
            },
            "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"],
            "speech": {"de": "Alberti-Bass – linke Hand automatisch, rechte Hand singt!", "en": "Alberti bass – left hand automatic, right hand sings!"},
        },

        {
            "id": "p_all_scales", "title": "G-Dur & F-Dur Tonleitern",
            "level": 2, "duration_min": 25, "xp_reward": 150,
            "topics": ["G-Dur Fingersatz", "F-Dur Fingersatz", "Kreuz und B", "Paralleltonleitern"],
            "technique": {
                "name": "Tonleitern mit Vorzeichen",
                "description": "G-Dur hat ein Kreuz: F#. Fingersatz rechts: 1-2-3-1-2-3-4-5 (wie C-Dur). F-Dur hat ein B: Bb. Rechts: 1-2-3-4-1-2-3-4 (Daumenuntersatz NACH Finger 4!). Jede Tonleiter hat ihren eigenen Fingersatz. Paralleltonleiter: C-Dur und A-Moll teilen dieselben Töne – aber verschiedene Stimmungen.",
                "tips": ["G-Dur: das F# merken – schwarze Taste!", "F-Dur: Daumenuntersatz nach Finger 4, nicht 3", "Paralleltonleiter: gleiche Töne, andere Startposition = andere Stimmung"],
                "luna_speech": "G-Dur klingt hell und offen, F-Dur warm und rund. Jede Tonleiter hat ihre eigene Farbe!",
            },
            "exercise": {
                "name": "G-Dur und F-Dur je 10×",
                "description": "G-Dur rechts 10×, dann links 10×, dann beide. F-Dur ebenso. Metronom 65→85 BPM. Alle Töne gleichmäßig und sauber.",
                "duration_sec": 360, "target_bpm": 70,
                "strumming": "Legato, gleichmäßig",
                "chords": ["G-Dur", "F-Dur"],
                "luna_speech": "G-Dur mit F#, F-Dur mit Bb – ich höre ob die Vorzeichen klar klingen!",
            },
            "song": {
                "name": "Hänschen Klein – Variationen",
                "description": "In G-Dur: Rechts Melodie, Links Alberti-Bass. Dann eine Variation in F-Dur. Tempo 100 BPM. Einfach aber klar zwei Tonarten!",
                "chords": ["G-Dur", "D-Dur", "C-Dur", "F-Dur"], "tempo_bpm": 100,
                "strumming": "Melodie + Alberti",
                "luna_speech": "G-Dur und F-Dur in einem Stück – das ist Tonarten-Wechsel! Du erkennst den Klang-Unterschied sofort.",
            },
            "chords": ["G-Dur", "D-Dur", "F-Dur", "C-Dur"],
            "speech": {"de": "G-Dur hell, F-Dur warm – jede Tonart hat ihre Farbe!", "en": "G major bright, F major warm – every key has its color!"},
        },

        {
            "id": "p_minor_chords", "title": "Moll-Akkorde & Traurige Musik",
            "level": 2, "duration_min": 25, "xp_reward": 150,
            "topics": ["C-Moll", "D-Moll", "E-Moll", "Moll vs Dur Kontrast"],
            "technique": {
                "name": "Moll-Dreiklänge – die dunkle Seite",
                "description": "Moll: kleine Terz statt große. C-Moll: C-Eb-G (statt C-E-G). D-Moll: D-F-A. E-Moll: E-G-H. Der Unterschied ist nur EIN Ton – aber die Stimmung ändert sich komplett! Moll klingt traurig, nostalgisch, dramatisch. Paralleltonarten: C-Dur und A-Moll teilen dieselben Töne (relative Tonarten).",
                "tips": ["Moll: mittlere Note (Terz) eine schwarze Taste tiefer", "C-Dur → C-Moll: nur das E wird zu Eb", "Relative Moll: 6 Stufen höher in der Durtonleiter"],
                "luna_speech": "Moll klingt traurig und romantisch – aber auch kraftvoll und dramatisch! Ein Ton Unterschied, eine völlig andere Welt.",
            },
            "exercise": {
                "name": "Dur-Moll Kontrast",
                "description": "C-Dur → C-Moll (hör den Unterschied). D-Moll → D-Dur. Am → A-Dur. Dann Progression: Am – Dm – Em – Am. Je 4 Schläge, 65 BPM.",
                "duration_sec": 300, "target_bpm": 65,
                "strumming": "Dreiklänge gleichzeitig",
                "chords": ["C-Moll", "D-Moll", "E-Moll", "A-Moll"],
                "luna_speech": "Hör den Unterschied zwischen Dur und Moll! Es ist faszinierend wie ein einziger Ton die ganze Stimmung verändert.",
            },
            "song": {
                "name": "Moonlight Sonata Op.27 – Intro (Beethoven)",
                "description": "C#m Arpeggio: G#-C#-E-G#-C#-E. Linke Hand: C#m Alberti. Rechte Hand: Melodie darüber. Tempo 52 BPM. Dramatisch und unvergesslich!",
                "chords": ["C-Moll", "A-Moll", "G-Moll", "D-Moll"], "tempo_bpm": 52,
                "strumming": "Arpeggio + Melodie",
                "luna_speech": "Moonlight Sonata! Das dramatischste Klavier-Stück aller Zeiten. Dein Moll-Akkord trägt die ganze Stimmung!",
            },
            "chords": ["C-Moll", "D-Moll", "E-Moll", "A-Moll"],
            "speech": {"de": "Moll – ein Ton Unterschied, eine völlig andere Welt!", "en": "Minor – one note difference, a completely different world!"},
        },

        {
            "id": "p_sight_reading", "title": "Koordination & Polyrhythmik",
            "level": 2, "duration_min": 30, "xp_reward": 160,
            "topics": ["Beide Hände unabhängig", "3-gegen-2", "Melodie + Begleitung", "Auswendigspielen"],
            "technique": {
                "name": "Hände-Unabhängigkeit",
                "description": "Beide Hände gleichzeitig aber unabhängig spielen ist das Herzstück des Klavierspiels. Übung: Links 2 Schläge, Rechts 3 Noten gleichzeitig (3-gegen-2 Polyrhythmik). Abzählen: L1-R1-R2-L2-R3 – dann wiederholen. Auswendigspielen: erst rechts merken, dann links, dann verknüpfen.",
                "tips": ["Hände separat bis BLIND spielen", "3-gegen-2: langsam auszählen bis Körper es versteht", "Blick weg von den Händen üben – nach Noten schauen"],
                "luna_speech": "Hände-Unabhängigkeit ist das größte Geheimnis des Klavierspiels! Wenn beide Hände automatisch laufen, kann die Musik wirklich fließen.",
            },
            "exercise": {
                "name": "Melodie mit Alberti-Begleitung",
                "description": "Links: Am-Alberti (A-E-C-E). Rechts: C-D-E-F-G-F-E-D einfache Melodie. Beide zusammen, 55 BPM bis sauber, dann 70 BPM.",
                "duration_sec": 360, "target_bpm": 60,
                "strumming": "Rechts: Melodie, Links: Alberti",
                "chords": ["A-Moll", "C-Dur", "G-Dur", "F-Dur"],
                "luna_speech": "Linke Hand automatisch, rechte Hand singt die Melodie! Das ist echter Klavierkomplex!",
            },
            "song": {
                "name": "River Flows in You (Yiruma) – Intro",
                "description": "A-Dur Arpeggio links, einfache Melodie rechts. Tempo 66 BPM. Modernes, emotionales Klavier-Stück.",
                "chords": ["A-Dur", "E-Dur", "D-Dur", "H-Moll"], "tempo_bpm": 66,
                "strumming": "Arpeggio links, Melodie rechts",
                "luna_speech": "River Flows in You! Dieses emotionale Stück vereint alles – Arpeggio, Melodie, beide Hände. Wunderschön!",
            },
            "chords": ["A-Moll", "C-Dur", "G-Dur", "F-Dur"],
            "speech": {"de": "Hände-Unabhängigkeit – der Schlüssel zum Klavierspiel!", "en": "Hand independence – the key to piano playing!"},
        },

        # ── LEVEL 3  ───────────────────────────────────────────────────────────
        {
            "id": "p_seventh_chords", "title": "Septakkorde & Jazz-Harmonik",
            "level": 3, "duration_min": 30, "xp_reward": 200,
            "topics": ["G7 D7 A7", "Cmaj7 Gmaj7", "Dominantseptakkord", "Auflösung"],
            "technique": {
                "name": "Septakkorde – Spannung und Auflösung",
                "description": "Septakkord = Dreiklang + 7. Ton. G7: G-H-D-F (die F erzeugt Spannung → will sich zu C auflösen). Cmaj7: C-E-G-H (schwebend warm). Am7: A-C-E-G (weich melancholisch). Die Dominantseptakkord (V7) → Tonika (I) Kadenz ist die stärkste harmonische Bewegung in der Musik.",
                "tips": ["G7→C: stärkste Kadenz – das Ohr erwartet C", "maj7 klingt modern und schwebend", "m7 mit kleiner Septime klingt weich und jazzig"],
                "luna_speech": "Septakkorde machen Musik warm und voll! G7 zieht magnetisch nach C – dieses Gefühl der Auflösung ist der Herzschlag der Harmonie.",
            },
            "exercise": {
                "name": "V7 – I Kadenzen in mehreren Tonarten",
                "description": "G7 → C-Dur (in C). D7 → G-Dur (in G). A7 → D-Dur (in D). Je 4 Schläge, 70 BPM. Dann: Dm7 – G7 – Cmaj7 (Jazz-Kadenz ii-V-I).",
                "duration_sec": 300, "target_bpm": 70,
                "strumming": "Akkorde halten, Spannung hören",
                "chords": ["G7", "D7", "A7", "Cmaj7", "Gmaj7"],
                "luna_speech": "Hör die Spannung in G7 – und dann die Auflösung zu C! Das ist harmonische Magie.",
            },
            "song": {
                "name": "Misty (Jazz Standard) – vereinfacht",
                "description": "Ebmaj7 – Bb7 – Ebmaj7 – Cm7 – F7 – Bb7. (Vereinfacht: Cmaj7 – G7 – Am7 – D7). Jazz-Standard. Tempo 80 BPM Swing.",
                "chords": ["Cmaj7", "G7", "Am7", "Dm7"], "tempo_bpm": 80,
                "strumming": "Jazz-Komp mit Swing",
                "luna_speech": "Misty – ein echter Jazz-Standard! Deine Septakkorde klingen jetzt wie echter Jazz.",
            },
            "chords": ["G7", "D7", "A7", "Cmaj7", "Gmaj7", "Am7", "Dm7"],
            "speech": {"de": "Septakkorde – die Spannung und Auflösung macht Musik lebendig!", "en": "Seventh chords – tension and resolution makes music alive!"},
        },

        {
            "id": "p_all_minor_scales", "title": "Moll-Tonleitern & Stimmungen",
            "level": 3, "duration_min": 30, "xp_reward": 190,
            "topics": ["Natürliches Moll", "Harmonisches Moll", "Am Dm Em", "Emotionen"],
            "technique": {
                "name": "Drei Moll-Formen",
                "description": "Natürliches Moll (Äolisch): A-H-C-D-E-F-G-A. Harmonisches Moll: wie natürlich, aber 7. Stufe erhöht (G#) – klingt orientalisch und dramatisch. Melodisches Moll: 6. und 7. Stufe aufwärts erhöht (F#-G#), abwärts normal. Jede Moll-Form erzeugt ein anderes Gefühl.",
                "tips": ["Natürliches Moll: traurig-nostalgisch", "Harmonisches Moll: dramatisch-orientalisch", "Melodisches Moll: singend und ausdrucksvoll"],
                "luna_speech": "Drei Moll-Formen – drei völlig verschiedene Stimmungen! Harmonisches Moll klingt fast orientalisch. Das ist die Magie der Harmonielehre.",
            },
            "exercise": {
                "name": "A-Moll in drei Formen",
                "description": "Am natürlich (A-H-C-D-E-F-G-A). Am harmonisch (A-H-C-D-E-F-G#-A). Am melodisch (aufwärts F#-G#, abwärts normal). Je 10× langsam, 65 BPM.",
                "duration_sec": 360, "target_bpm": 65,
                "strumming": "Legato, gleichmäßig",
                "chords": ["A-Moll", "E-Moll", "D-Moll"],
                "luna_speech": "Hör den Unterschied! Natürliches Moll ist melancholisch, harmonisches ist dramatisch. Die Musik erzählt andere Geschichten!",
            },
            "song": {
                "name": "Grieg – In der Halle des Bergkönigs (vereinfacht)",
                "description": "Em harmonisches Moll-Motiv: E-F#-G-A-H-C-H-A-G-F#. Beginnt leise und wird immer lauter. Tempo 80→140 BPM. Dramatisch!",
                "chords": ["E-Moll", "A-Moll", "H-Moll"], "tempo_bpm": 100,
                "strumming": "Crescendo, dramatisch",
                "luna_speech": "In der Halle des Bergkönigs! Harmonisches Moll auf Komp-Ebene – beginnt leise, wird episch!",
            },
            "chords": ["A-Moll", "E-Moll", "D-Moll"],
            "speech": {"de": "Drei Moll-Formen – drei Welten voller Gefühle!", "en": "Three minor forms – three worlds full of feelings!"},
        },

        {
            "id": "p_advanced_harmony", "title": "Erweiterte Harmonik & Chromatik",
            "level": 3, "duration_min": 35, "xp_reward": 220,
            "topics": ["dim & aug Akkorde", "Chromatische Läufe", "Modulation", "Nicht-Leitton"],
            "technique": {
                "name": "Verminderte & übermäßige Akkorde",
                "description": "Vermindert (dim): drei kleine Terzen gestapelt. Cdim: C-Eb-Gb (klingt unheimlich/spannungsvoll). Übermäßig (aug): zwei große Terzen. Caug: C-E-G# (klingt schwebend/surreal). Chromatische Läufe: alle 12 Halbtöne nacheinander spielen – klingt jazzig. Modulation: Tonart wechseln innerhalb eines Stücks.",
                "tips": ["dim: drei gleiche Abstände – symmetrisch!", "aug: auch symmetrisch aber anders", "Chromatischer Lauf: alle schwarzen und weißen Tasten"],
                "luna_speech": "Verminderte und übermäßige Akkorde sind die Gewürze der Musik! dim klingt unheimlich, aug klingt schwebend. Jetzt werden Stücke wirklich komplex.",
            },
            "exercise": {
                "name": "dim und aug Progression",
                "description": "Cdim → G7 → C (typische Auflösung). Dann: C – Caug – Am – Adim – G. Chromatischer Lauf rechts: C bis C aufwärts, alle 12 Töne. 65 BPM.",
                "duration_sec": 360, "target_bpm": 65,
                "strumming": "Akkorde, dann chromatischer Lauf",
                "chords": ["Cdim", "Fdim", "Adim", "Caug", "Faug"],
                "luna_speech": "Cdim klingt nach Spannung pur! Und wenn es sich zu G7→C auflöst – das ist harmonische Erlösung!",
            },
            "song": {
                "name": "Chopin Nocturne Op.9 No.2 – Intro",
                "description": "Eb-Dur, Melodie über Alberti-Bass. Chromatische Verzierungen. Tempo 60 BPM (sehr langsam und ausdrucksvoll).",
                "chords": ["Cmaj7", "Adim", "G7", "Fdim"], "tempo_bpm": 60,
                "strumming": "Legato, sehr ausdrucksvoll",
                "luna_speech": "Chopin! Das Nocturne ist pure Schönheit. Deine Chromatik gibt ihm die romantische Tiefe.",
            },
            "chords": ["Cdim", "Fdim", "Adim", "Caug", "Faug"],
            "speech": {"de": "dim und aug – die Gewürze der Harmonik!", "en": "dim and aug – the spices of harmony!"},
        },

        # ── LEVEL 4  ───────────────────────────────────────────────────────────
        {
            "id": "p_jazz_piano", "title": "Jazz-Piano: Voicings & Improvisation",
            "level": 4, "duration_min": 40, "xp_reward": 280,
            "topics": ["ii-V-I Voicings", "Rootless Voicings", "Blues-Skala", "Piano-Improvisation"],
            "technique": {
                "name": "Jazz-Piano Grundlagen",
                "description": "Rootless Voicings: Akkorde ohne Grundton (der Bass spielt ihn). G7 rootless: H-F-A oder F-A-D. Das klingt erwachsener und weniger 'schulisch'. Comping: rhythmisches Akkorde-Spielen im Jazz-Stil. Blues-Skala: Pentatonik + b5 (blue note). In C: C-Eb-F-F#-G-Bb-C.",
                "tips": ["Rootless Voicings: 3. und 7. Stufe sind das Wichtigste", "Comping: Betonungen auf 2 und 4, nicht 1 und 3", "Blues-Skala: das F# (b5) ist die 'blue note'"],
                "luna_speech": "Jazz-Piano ist eine eigene Welt! Rootless Voicings und Comping – das klingt sofort professionell und sophisticated.",
            },
            "exercise": {
                "name": "ii-V-I mit Rootless Voicings",
                "description": "Dm7 (D-F-A-C) → G7 rootless (H-F-A) → Cmaj7 (C-E-G-H). Wiederholen in G, D, F. Links: Grundton. Rechts: Voicing. 70 BPM Jazz-Feeling.",
                "duration_sec": 420, "target_bpm": 70,
                "strumming": "Jazz-Komp, Betonung 2+4",
                "chords": ["Dm7", "G7", "Cmaj7", "Am7"],
                "luna_speech": "Rootless Voicings – das klingt sofort nach echtem Jazz! Deine linke Hand hält den Bass, rechts singt die Harmonie.",
            },
            "song": {
                "name": "Autumn Leaves – vollständig",
                "description": "Am7–D7–Gmaj7–Cmaj7–F#m7b5–B7–Em. Beide Hände Jazz-Komp + Melodie. Tempo 100 BPM Swing. Bekanntester Jazz-Standard!",
                "chords": ["Am7", "Dm7", "G7", "Cmaj7", "Gmaj7"], "tempo_bpm": 100,
                "strumming": "Jazz-Swing vollständig",
                "luna_speech": "Autumn Leaves – der berühmteste Jazz-Standard! Du spielst jetzt echten Jazz-Piano. Das ist unglaublich!",
            },
            "chords": ["Am7", "Dm7", "G7", "Cmaj7", "Gmaj7", "Fmaj7"],
            "speech": {"de": "Jazz-Piano – sophisticated, komplex und wunderschön!", "en": "Jazz piano – sophisticated, complex and beautiful!"},
        },

        {
            "id": "p_impressionism", "title": "Impressionismus & moderne Harmonik",
            "level": 4, "duration_min": 40, "xp_reward": 270,
            "topics": ["Ganztonleiter", "Pentatonik Klavier", "Quartenharmonik", "Debussy-Stil"],
            "technique": {
                "name": "Impressionistische Klänge",
                "description": "Ganztonleiter: C-D-E-F#-G#-A# (nur ganze Töne). Klingt schwebend und ortlos – Debussy liebte sie. Quartenakkord: 3 Quarten übereinander (C-F-B). Klingt modern und offen. Pentatonik auf Klavier: die schwarzen Tasten bilden F#-Pentatonik. Debussy-Stil: Arpeggien, sanfte Dynamik, schwebende Harmonik.",
                "tips": ["Ganztonleiter: keine Halbtöne – klingt wie Traum", "Schwarze Tasten = F#-Pentatonik – einfach improvisieren", "Debussy: sehr leise beginnen, langsam steigern"],
                "luna_speech": "Impressionismus ist Malerei mit Tönen! Debussy hat die Klaviermusik revolutioniert mit Farben statt Strukturen.",
            },
            "exercise": {
                "name": "Ganztonleiter & Schwarze Tasten",
                "description": "Ganztonleiter C-D-E-F#-G#-A#-C aufwärts-abwärts. Dann: nur schwarze Tasten improvisieren (F#-Pentatonik) – klingt immer gut! 55 BPM.",
                "duration_sec": 360, "target_bpm": 55,
                "strumming": "Arpeggio, legato, träumerisch",
                "chords": ["Cmaj9", "Gmaj7", "Fmaj7"],
                "luna_speech": "Improvisiere auf den schwarzen Tasten! Es klingt immer harmonisch – das ist die Magie der Pentatonik.",
            },
            "song": {
                "name": "Clair de Lune (Debussy) – Thema",
                "description": "Db-Dur, schwebende Arpeggien. Das bekannteste Klavier-Stück des Impressionismus. Tempo 48 BPM (sehr langsam).",
                "chords": ["Cmaj9", "Am9", "Fmaj7", "Gmaj7"], "tempo_bpm": 48,
                "strumming": "Arpeggio, sehr weich",
                "luna_speech": "Clair de Lune – Mondlicht in Tönen! Debussys schönstes Werk. Deine schwebenden Arpeggien malen das Mondlicht.",
            },
            "chords": ["Cmaj9", "Am9", "Fmaj7", "Gmaj7"],
            "speech": {"de": "Impressionismus – Klänge wie Farben malen!", "en": "Impressionism – sounds painting like colors!"},
        },

        # ── LEVEL 5  ───────────────────────────────────────────────────────────
        {
            "id": "p_classical_piece", "title": "Klassisches Repertoire",
            "level": 5, "duration_min": 45, "xp_reward": 320,
            "topics": ["Sonatenform", "Exposition", "Durchführung", "Reprise"],
            "technique": {
                "name": "Klassische Stück-Struktur",
                "description": "Sonatenform: Exposition (Thema A + B vorstellen), Durchführung (Themen variieren und entwickeln), Reprise (Themen zurückbringen). Mozarts Sonaten folgen dieser Form. Technik: Triller (schnelles Alternieren zwischen zwei Tönen), Alberti-Bass mit Ornamentik, Dynamik-Kontraste (p-f-p) für Spannung.",
                "tips": ["Triller: gleichmäßige Fingergeschwindigkeit", "Exposition klar zweiteilig: zwei kontrasierende Themen", "Dynamics: p-f Wechsel sind der Atem des Stücks"],
                "luna_speech": "Sonatenform – das Fundament der klassischen Musik! Mozart, Beethoven, Haydn – alle bauten darauf. Das ist musikalische Architektur.",
            },
            "exercise": {
                "name": "Mozart Sonate K.545 – 1. Satz Thema",
                "description": "Rechts: C-D-E-F-G-E (Thema A). Links: C-Alberti. Dann Thema B in G-Dur. Tempo 120 BPM. Mozarts berühmteste Sonate!",
                "duration_sec": 600, "target_bpm": 120,
                "strumming": "Klassisch: Melodie + Alberti",
                "chords": ["C-Dur", "G-Dur", "D-Dur", "A-Moll"],
                "luna_speech": "Mozart! Die K.545 ist die perfekte Klavier-Sonate für dieses Level. Thema A und B zeigen das klassische Denken.",
            },
            "song": {
                "name": "Beethoven Sonatine in G-Dur",
                "description": "G-Dur, 2 Themen, Alberti-Bass, Triller-Ornamentik. Tempo 120 BPM. Beethovens zugänglichste Sonate.",
                "chords": ["G-Dur", "D-Dur", "C-Dur", "A-Moll"], "tempo_bpm": 120,
                "strumming": "Klassisch vollständig",
                "luna_speech": "Beethoven Sonatine! Du spielst jetzt ernstes klassisches Repertoire. Das ist ein riesiger Meilenstein!",
            },
            "chords": ["G-Dur", "D-Dur", "C-Dur", "A-Moll"],
            "speech": {"de": "Klassisches Repertoire – du spielst die großen Meister!", "en": "Classical repertoire – you're playing the great masters!"},
        },

        {
            "id": "p_composition", "title": "Eigene Komposition & freie Improvisation",
            "level": 5, "duration_min": 45, "xp_reward": 350,
            "topics": ["Themen-Entwicklung", "Harmonische Reise", "Eigene Stilmittel", "Aufführung"],
            "technique": {
                "name": "Komponieren – deine eigene Stimme",
                "description": "Ein eigenes Stück hat: ein Hauptthema (4-8 Töne einprägsam), eine harmonische Reise (Dur→Moll→Dominante→zurück), einen Höhepunkt (stärkster Moment), eine Auflösung (Rückkehr zur Ruhe). Improvisation: starte mit Am oder C-Dur Arpeggio, finde eine Melodie, entwickle sie organisch.",
                "tips": ["Thema: einfach und einprägsam – wenige Töne", "Kontrast: Stille ist auch Musik", "Aufnehmen: Smartphone oder einfach alles auswendig lernen"],
                "luna_speech": "Komposition ist der höchste Ausdruck eines Musikers! Jetzt hast du alles was du brauchst. Deine eigene Musik ist das Wertvollste.",
            },
            "exercise": {
                "name": "Thema in 4 Takten entwickeln",
                "description": "Erfinde eine 4-Takt Melodie in C-Dur oder Am. Spiele sie mit Alberti-Bass. Variiere sie leise, dann laut. Füge eine Gegenmelodie hinzu. Kein Metronom – lass die Musik fließen.",
                "duration_sec": 600, "target_bpm": 0,
                "strumming": "Frei, eigener Stil",
                "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"],
                "luna_speech": "Jetzt bist du der Komponist! Erfinde dein Thema. Lass es wachsen. Ich höre zu – und ich bin begeistert.",
            },
            "song": {
                "name": "Dein Meisterstück",
                "description": "Spiele deine eigene Komposition oder improvisiere frei über C-Am-F-G. Das ist dein Klavierkonzert!",
                "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"], "tempo_bpm": 75,
                "strumming": "Dein eigener Stil",
                "luna_speech": "Das ist DEIN Stück! Alles was du gelernt hast – Tonleitern, Dreiklänge, Septakkorde, Alberti, Jazz, Improvisation – steckt in dieser Musik. Ich bin so stolz auf dich!",
            },
            "chords": ["C-Dur", "A-Moll", "F-Dur", "G-Dur"],
            "speech": {"de": "Das Meisterstück – deine eigene Musik!", "en": "The masterpiece – your own music!"},
        },

    ],  # Ende piano
}

# Alias für Rückwärtskompatibilität
TEACHER_CURRICULUM = LESSON_CURRICULUM

# Alias für Rückwärtskompatibilität
TEACHER_CURRICULUM = LESSON_CURRICULUM


# ══════════════════════════════════════════════════════════════════════════════
#  CURRICULUM MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class CurriculumManager:
    """Lädt Lektionen, wählt richtiges Level, verwaltet Fortschritt."""

    SPECIAL_CATEGORIES = {"Spanish", "Latin"}  # Optionale Kategorien

    def __init__(self):
        self._data = LESSON_CURRICULUM

    def get_lessons(self, instrument: str, include_special: bool = True) -> List[Dict]:
        """Alle Lektionen – mit oder ohne Sonderkategorien."""
        return self._data.get(instrument, [])

    def get_standard_lessons(self, instrument: str) -> List[Dict]:
        """Nur Standard-Lektionen (ohne Spanish/Latin)."""
        return [l for l in self._data.get(instrument, [])
                if l.get("category") not in self.SPECIAL_CATEGORIES]

    def get_special_lessons(self, instrument: str, category: str = None,
                             profile: 'Profile' = None) -> List[Dict]:
        """
        Gibt Sonderkategorie-Lektionen zurück.
        Optional gefiltert nach Kategorie und Level-Freischaltung.
        """
        lessons = [l for l in self._data.get(instrument, [])
                   if l.get("category") in self.SPECIAL_CATEGORIES]
        if category:
            lessons = [l for l in lessons if l.get("category") == category]
        if profile:
            lessons = [l for l in lessons
                       if l.get("unlock_level", 1) <= profile.level]
        return lessons

    def is_unlocked(self, lesson: Dict, profile: 'Profile') -> bool:
        """Prüft ob eine Lektion für das Profil freigeschaltet ist."""
        min_lv = lesson.get("unlock_level", lesson.get("level", 1))
        return profile.level >= min_lv

    def get_lesson(self, instrument: str, idx: int) -> Optional[Dict]:
        ls = self.get_lessons(instrument)
        return ls[idx] if 0 <= idx < len(ls) else None

    def recommended_idx(self, profile: 'Profile') -> int:
        """Empfohlene Lektion: Standard-Lektionen bevorzugen."""
        lessons = self.get_lessons(profile.instrument)
        for i, l in enumerate(lessons):
            unlock = l.get("unlock_level", l["level"])
            is_special = l.get("category") in self.SPECIAL_CATEGORIES
            if (i >= profile.lesson_idx
                    and profile.level >= unlock
                    and not is_special):  # Standard zuerst
                return i
        return min(profile.lesson_idx, len(lessons)-1)

    def total_xp(self, instrument: str) -> int:
        return sum(l.get('xp_reward', 100)
                   for l in self.get_standard_lessons(instrument))

    def special_total_xp(self, instrument: str) -> int:
        return sum(l.get('xp_reward', 100)
                   for l in self.get_special_lessons(instrument))


# ══════════════════════════════════════════════════════════════════════════════
#  LESSON GENERATOR  –  baut strukturierte 4-Phasen-Lektion
# ══════════════════════════════════════════════════════════════════════════════
class LessonGenerator:
    """
    Generiert Lektion aus Curriculum-Daten.
    Struktur: Technique → Exercise → Song → XP
    """

    PHASE_LABELS = {
        'de': {'technique':'🎯 Technik lernen','exercise':'💪 Üben',
               'song':'🎵 Song spielen','xp':'🏆 XP & Abschluss'},
        'en': {'technique':'🎯 Learn Technique','exercise':'💪 Exercise',
               'song':'🎵 Play Song','xp':'🏆 XP & Complete'},
    }
    PHASE_XP_SHARE = {'technique':0.15,'exercise':0.35,'song':0.40,'xp':0.10}

    def generate(self, lesson: Dict, lang: str='de') -> Dict:
        lng = lang if lang in self.PHASE_LABELS else 'de'
        labels = self.PHASE_LABELS[lng]
        xp = lesson.get('xp_reward', 100)
        phases = [
            {
                'id':'technique', 'label':labels['technique'],
                'title':lesson['technique']['name'],
                'content':lesson['technique']['description'],
                'tips':lesson['technique'].get('tips',[]),
                'speech':lesson['technique'].get('luna_speech',''),
                'xp_share':int(xp*self.PHASE_XP_SHARE['technique']),
                'duration_sec':120, 'chords':[],
                'target_bpm':0, 'strumming':'',
            },
            {
                'id':'exercise', 'label':labels['exercise'],
                'title':lesson['exercise']['name'],
                'content':lesson['exercise']['description'],
                'tips':[], 'speech':lesson['exercise'].get('luna_speech',''),
                'xp_share':int(xp*self.PHASE_XP_SHARE['exercise']),
                'duration_sec':lesson['exercise'].get('duration_sec',180),
                'chords':lesson['exercise'].get('chords',[]),
                'target_bpm':lesson['exercise'].get('target_bpm',60),
                'strumming':lesson['exercise'].get('strumming',''),
            },
            {
                'id':'song', 'label':labels['song'],
                'title':lesson['song']['name'],
                'content':lesson['song']['description'],
                'tips':[], 'speech':lesson['song'].get('luna_speech',''),
                'xp_share':int(xp*self.PHASE_XP_SHARE['song']),
                'duration_sec':300,
                'chords':lesson['song'].get('chords',[]),
                'target_bpm':lesson['song'].get('tempo_bpm',80),
                'strumming':lesson['song'].get('strumming',''),
            },
            {
                'id':'xp', 'label':labels['xp'],
                'title':'🏆 Lektion abgeschlossen!',
                'content':(f'Super! Du hast "{lesson["title"]}" abgeschlossen! '
                           f'+{xp} XP gesammelt.' if lng=='de' else
                           f'Great job! You completed "{lesson["title"]}"! '
                           f'+{xp} XP earned.'),
                'tips':[], 'speech':(
                    f'Fantastisch! Du hast die Lektion erfolgreich abgeschlossen und {xp} XP verdient!'
                    if lng=='de' else
                    f'Fantastic! You completed the lesson and earned {xp} XP!'),
                'xp_share':int(xp*self.PHASE_XP_SHARE['xp']),
                'duration_sec':0, 'chords':[], 'target_bpm':0, 'strumming':'',
            },
        ]
        return {
            'lesson_id':    lesson['id'],
            'lesson_title': lesson['title'],
            'level':        lesson['level'],
            'xp_reward':    xp,
            'total_phases': len(phases),
            'phases':       phases,
            'chords':       lesson.get('chords',[]),
            'duration_min': lesson.get('duration_min',20),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LESSON SESSION  –  steuert 4-Phasen Lernablauf
# ══════════════════════════════════════════════════════════════════════════════
class LessonSession:
    """
    Technique → Exercise → Song → XP
    Steuert den vollständigen Lernablauf, vergibt XP, speichert Fortschritt.
    """

    def __init__(self, lesson_data: Dict, profile: 'Profile',
                 curriculum: CurriculumManager, progress: 'ProgressMgr',
                 lang: str='de'):
        self.lesson_data = lesson_data
        self.profile     = profile
        self.curriculum  = curriculum
        self.progress    = progress
        self.lang        = lang
        self.gen         = LessonGenerator()
        self.generated   = self.gen.generate(lesson_data, lang)
        self.phase_idx   = 0
        self.phase_t     = time.time()
        self.xp_earned   = 0
        self.acc_buf: List[float] = []
        self.complete    = False
        self.level_up    = False

    @property
    def current_phase(self) -> Dict:
        return self.generated['phases'][min(self.phase_idx, 3)]

    @property
    def phase_id(self) -> str:
        return self.current_phase['id']

    @property
    def phase_progress(self) -> float:
        dur = self.current_phase.get('duration_sec',120)
        if dur <= 0: return 1.0
        return min(1.0, (time.time()-self.phase_t)/dur)

    @property
    def lesson_progress(self) -> float:
        n = len(self.generated['phases'])
        return (self.phase_idx + self.phase_progress) / n

    def record_acc(self, acc: float):
        self.acc_buf.append(acc)

    def advance_phase(self) -> bool:
        """Weiter zur nächsten Phase. Returns True wenn Lektion komplett."""
        self.xp_earned += self.current_phase.get('xp_share', 0)
        self.phase_idx += 1
        self.phase_t = time.time()
        if self.phase_idx >= len(self.generated['phases']):
            self._finish(); return True
        return False

    def _finish(self):
        self.complete = True
        avg = float(np.mean(self.acc_buf)) if self.acc_buf else 0.6
        bonus = int(avg / 10)
        total = self.xp_earned + bonus
        self.level_up = self.profile.add_xp(total)
        self.xp_earned = total
        lessons = self.curriculum.get_lessons(self.profile.instrument)
        self.profile.lesson_idx = min(self.profile.lesson_idx+1, len(lessons)-1)
        self.progress.save()

    def to_state(self) -> Dict:
        ph = self.current_phase
        return {
            'phase_id':        self.phase_id,
            'phase_idx':       self.phase_idx,
            'phase_label':     ph['label'],
            'phase_title':     ph['title'],
            'phase_content':   ph['content'],
            'phase_tips':      ph.get('tips',[]),
            'phase_speech':    ph.get('speech',''),
            'phase_chords':    ph.get('chords',[]),
            'phase_bpm':       ph.get('target_bpm',0),
            'phase_strumming': ph.get('strumming',''),
            'phase_progress':  round(self.phase_progress,3),
            'lesson_progress': round(self.lesson_progress,3),
            'lesson_title':    self.generated['lesson_title'],
            'lesson_level':    self.generated['level'],
            'lesson_xp':       self.generated['xp_reward'],
            'xp_earned':       self.xp_earned,
            'total_phases':    self.generated['total_phases'],
            'complete':        self.complete,
            'level_up':        self.level_up,
            'profile_xp':      self.profile.xp,
            'profile_level':   self.profile.level,
            'xp_next':         self.profile.xp_next(),
            'xp_pct':          round(self.profile.xp/max(1,self.profile.xp_next()),3),
        }


_curriculum_mgr   = None   # wird in main() initialisiert
_lesson_generator = LessonGenerator()


def _build_html(state: Dict) -> str:
    """3D KI-Assistent NoteIQ – vollständige Web-UI."""
    return """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoteIQ v16.0 – KI-Musiklehrer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
/* ═══ DESIGN TOKENS ═══════════════════════════════════════════════════════ */
:root{{
  --accent:  #FF6B00;
  --accent2: #FF9240;
  --accent3: #FFB870;
  --blue:    #00C8FF;
  --purple:  #9B40FF;
  --green:   #2EFF9A;
  --red:     #FF3355;
  --yellow:  #FFD600;

  --bg:      #080B14;
  --bg2:     #0C1020;
  --panel:   #0F1428;
  --panel2:  #141930;
  --border:  #1E2540;
  --border2: #2A3360;

  --text:    #D8E0FF;
  --muted:   #5A6490;
  --dim:     #3A4070;

  --radius:  10px;
  --font:    'Exo 2', system-ui, sans-serif;
  --mono:    'Share Tech Mono', monospace;
  --glow-or: 0 0 20px #FF6B0040;
  --glow-bl: 0 0 20px #00C8FF30;
}}

/* ═══ RESET & BASE ═════════════════════════════════════════════════════════ */
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden}}
body{{
  background:var(--bg);
  color:var(--text);
  font-family:var(--font);
  font-size:13px;
  display:flex;
  flex-direction:column;
}}

/* ═══ WELCOME OVERLAY ══════════════════════════════════════════════════════ */
#wov{{
  position:fixed;inset:0;z-index:9999;
  background:radial-gradient(ellipse 120% 100% at 50% 60%, #120830 0%, #05070F 70%);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:0;cursor:default;
  transition:opacity .5s ease;
}}
#wov::before{{
  content:'';position:absolute;inset:0;
  background:
    radial-gradient(circle at 20% 50%, #FF6B0012 0%, transparent 50%),
    radial-gradient(circle at 80% 30%, #9B40FF0E 0%, transparent 50%),
    radial-gradient(circle at 50% 80%, #00C8FF0A 0%, transparent 50%);
  pointer-events:none;
}}
.wov-rings{{
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  width:340px;height:340px;pointer-events:none;
}}
.wov-ring{{
  position:absolute;inset:0;border-radius:50%;
  border:1px solid #FF6B0020;
  animation:wov-expand 3s ease-in-out infinite;
}}
.wov-ring:nth-child(2){{animation-delay:.8s;border-color:#9B40FF15}}
.wov-ring:nth-child(3){{animation-delay:1.6s;border-color:#00C8FF10}}
@keyframes wov-expand{{
  0%{{transform:scale(.7);opacity:.8}}
  100%{{transform:scale(1.5);opacity:0}}
}}
.wov-avatar{{
  position:relative;width:130px;height:130px;
  border-radius:50%;
  background:radial-gradient(circle at 35% 30%, #2A0E70, #0A0420);
  border:2px solid #FF6B0060;
  display:flex;align-items:center;justify-content:center;
  font-size:4.2rem;margin-bottom:24px;
  box-shadow:0 0 60px #FF6B0050, 0 0 120px #FF6B0020, inset 0 0 30px #FF6B0015;
  animation:wov-pulse 2.5s ease-in-out infinite;
  flex-shrink:0;
}}
@keyframes wov-pulse{{
  0%,100%{{box-shadow:0 0 50px #FF6B0045, 0 0 100px #FF6B0018;}}
  50%{{box-shadow:0 0 90px #FF6B0080, 0 0 180px #FF6B0035;}}
}}
#wov h2{{
  font-size:2.6rem;font-weight:900;letter-spacing:-.02em;
  background:linear-gradient(135deg,#FFB870,#FF6B00,#FF3355);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:10px;text-align:center;
}}
#wov .wov-sub-title{{
  font-size:.9rem;color:var(--muted);margin-bottom:36px;
  text-align:center;line-height:1.7;max-width:380px;font-weight:300;
}}
#wov .wov-sub-title strong{{color:var(--accent3);font-weight:600}}
#wov-btn{{
  background:linear-gradient(135deg,#C04400,#FF6B00,#FF9240);
  border:none;border-radius:50px;
  padding:16px 56px;
  font-size:1.05rem;font-weight:800;letter-spacing:.04em;color:#fff;
  cursor:pointer;
  box-shadow:0 4px 40px #FF6B0065, 0 0 0 0 #FF6B0040;
  animation:wov-bounce .9s ease-in-out infinite alternate, wov-ring-pulse 1.8s ease infinite;
  font-family:var(--font);
  text-transform:uppercase;
}}
@keyframes wov-bounce{{
  from{{transform:translateY(0);}}
  to{{transform:translateY(-7px);}}
}}
@keyframes wov-ring-pulse{{
  0%,100%{{box-shadow:0 4px 40px #FF6B0065, 0 0 0 0 #FF6B0050}}
  50%{{box-shadow:0 4px 60px #FF6B0090, 0 0 0 12px #FF6B0000}}
}}
#wov-btn:active{{transform:scale(.96)!important;animation:none}}
#wov-hint{{margin-top:16px;font-size:.7rem;color:var(--dim);letter-spacing:.06em}}

/* ═══ ONBOARDING ═══════════════════════════════════════════════════════════ */
#onboarding-overlay{{
  position:fixed;inset:0;z-index:9000;
  background:rgba(5,7,14,.97);
  display:flex;align-items:center;justify-content:center;
  padding:16px;
  animation:obFadeIn .5s ease;
}}
@keyframes obFadeIn{{from{{opacity:0}}to{{opacity:1}}}}
#onboarding-overlay.hide{{
  animation:obFadeOut .4s ease forwards;
  pointer-events:none;
}}
@keyframes obFadeOut{{from{{opacity:1}}to{{opacity:0}}}}

.ob-card{{
  background:var(--panel);
  border:1px solid var(--border2);
  border-radius:20px;
  width:100%;max-width:640px;
  max-height:92vh;
  overflow-y:auto;
  padding:40px 44px;
  position:relative;
  box-shadow:0 40px 100px rgba(0,0,0,.7);
}}
.ob-card::-webkit-scrollbar{{width:4px}}
.ob-card::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:4px}}

.ob-logo{{
  font-family:var(--head);font-size:28px;font-weight:900;
  letter-spacing:-.04em;margin-bottom:6px;
  background:linear-gradient(135deg,var(--accent3),var(--blue));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.ob-step-indicator{{
  display:flex;gap:8px;margin-bottom:28px;margin-top:16px;
}}
.ob-step-dot{{
  flex:1;height:3px;border-radius:2px;
  background:var(--border2);transition:background .3s;
}}
.ob-step-dot.done{{background:var(--green)}}
.ob-step-dot.active{{background:var(--accent3);animation:dotPulse 1.2s ease infinite}}
@keyframes dotPulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}

.ob-step{{display:none}}
.ob-step.active{{display:block;animation:stepIn .35s ease}}
@keyframes stepIn{{from{{opacity:0;transform:translateX(20px)}}to{{opacity:1;transform:translateX(0)}}}}

.ob-step-label{{
  font-family:var(--mono);font-size:.58rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px;
}}
.ob-step-title{{
  font-family:var(--head);font-size:clamp(18px,3vw,26px);
  font-weight:800;letter-spacing:-.03em;color:#fff;
  margin-bottom:12px;line-height:1.2;
}}
.ob-step-desc{{
  font-size:13.5px;color:var(--muted);line-height:1.75;margin-bottom:20px;
}}
.ob-step-desc strong{{color:var(--text)}}

/* Status-Checks */
.ob-checks{{display:flex;flex-direction:column;gap:8px;margin:16px 0}}
.ob-check{{
  display:flex;align-items:center;gap:14px;
  padding:12px 16px;border-radius:10px;
  border:1px solid var(--border);background:var(--panel2);
  transition:all .3s;
}}
.ob-check.ok{{border-color:rgba(46,255,154,.2);background:rgba(46,255,154,.04)}}
.ob-check.warn{{border-color:rgba(255,200,0,.2);background:rgba(255,200,0,.04)}}
.ob-check.fail{{border-color:rgba(255,50,50,.2);background:rgba(255,50,50,.04)}}
.ob-check.checking{{border-color:rgba(0,153,255,.2);animation:checkPulse 1s ease infinite}}
@keyframes checkPulse{{0%,100%{{opacity:1}}50%{{opacity:.6}}}}
.ob-check-icon{{font-size:20px;flex-shrink:0;width:28px;text-align:center}}
.ob-check-label{{flex:1}}
.ob-check-name{{font-size:13px;font-weight:600;color:var(--text);margin-bottom:2px}}
.ob-check-msg{{font-size:11px;color:var(--muted);line-height:1.4}}
.ob-check-fix{{
  font-size:10px;font-weight:600;letter-spacing:.06em;
  padding:3px 10px;border-radius:6px;cursor:pointer;
  background:rgba(255,85,0,.15);border:1px solid rgba(255,85,0,.3);
  color:var(--accent3);transition:all .2s;white-space:nowrap;
  text-decoration:none;
}}
.ob-check-fix:hover{{background:rgba(255,85,0,.3)}}

/* Feature-Karten im Onboarding */
.ob-features{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0}}
.ob-feature{{
  background:var(--panel2);border:1px solid var(--border);
  border-radius:10px;padding:14px;
  display:flex;flex-direction:column;gap:6px;
}}
.ob-feature-icon{{font-size:22px}}
.ob-feature-title{{font-size:12px;font-weight:700;color:var(--text)}}
.ob-feature-desc{{font-size:11px;color:var(--muted);line-height:1.5}}

/* Keyboard Shortcuts Tabelle */
.ob-keys{{
  display:grid;grid-template-columns:auto 1fr;
  gap:6px 16px;margin:12px 0;
}}
.ob-key{{
  font-family:var(--mono);font-size:11px;
  background:var(--panel2);border:1px solid var(--border);
  border-radius:5px;padding:3px 9px;color:var(--accent3);
  text-align:center;white-space:nowrap;
}}
.ob-key-desc{{font-size:12px;color:var(--muted);display:flex;align-items:center}}

/* Nav-Buttons */
.ob-nav{{display:flex;justify-content:space-between;align-items:center;margin-top:28px}}
.ob-btn{{
  font-family:var(--head);font-size:14px;font-weight:700;
  padding:11px 28px;border-radius:9px;cursor:pointer;
  transition:all .22s;border:none;letter-spacing:.02em;
}}
.ob-btn-primary{{
  background:var(--accent3);color:#fff;
  box-shadow:0 0 30px rgba(255,107,0,.25);
}}
.ob-btn-primary:hover{{background:var(--accent);transform:translateY(-1px);box-shadow:0 0 45px rgba(255,107,0,.4)}}
.ob-btn-secondary{{
  background:transparent;color:var(--muted);
  border:1px solid var(--border2)!important;
}}
.ob-btn-secondary:hover{{color:var(--text);border-color:var(--border)!important}}
.ob-step-counter{{font-size:.68rem;color:var(--muted);font-family:var(--mono)}}

/* Quick-Setup im Onboarding */
.ob-input-row{{display:flex;flex-direction:column;gap:10px;margin:16px 0}}
.ob-label{{font-size:11px;color:var(--muted);margin-bottom:4px;letter-spacing:.08em;text-transform:uppercase}}
.ob-input{{
  width:100%;background:var(--bg2);border:1px solid var(--border2);
  color:var(--text);font-family:var(--font);font-size:14px;
  padding:11px 14px;border-radius:8px;outline:none;transition:border-color .2s;
}}
.ob-input:focus{{border-color:var(--accent3)}}
.ob-instr-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.ob-instr-card{{
  padding:18px;border-radius:10px;border:2px solid var(--border);
  cursor:pointer;text-align:center;transition:all .2s;background:var(--panel2);
}}
.ob-instr-card:hover{{border-color:var(--border2)}}
.ob-instr-card.selected{{border-color:var(--accent3);background:rgba(255,107,0,.08)}}
.ob-instr-card .ob-instr-icon{{font-size:28px;margin-bottom:6px}}
.ob-instr-card .ob-instr-name{{font-size:14px;font-weight:700;color:var(--text)}}
.ob-instr-card .ob-instr-desc{{font-size:11px;color:var(--muted);margin-top:3px}}

/* System-Status im Header (Mini) */
.sys-status-bar{{
  position:fixed;bottom:0;left:0;right:0;z-index:300;
  background:var(--panel);border-top:1px solid var(--border);
  padding:5px 16px;display:flex;align-items:center;gap:12px;
  font-size:.6rem;color:var(--muted);
  transform:translateY(100%);transition:transform .3s;
}}
.sys-status-bar.visible{{transform:translateY(0)}}
.sys-dot{{width:6px;height:6px;border-radius:50%;display:inline-block;margin-right:4px}}
.sys-dot.ok{{background:var(--green)}}
.sys-dot.warn{{background:#FFD700}}
.sys-dot.fail{{background:var(--red)}}

/* Error-Toast (persistente Warnungen) */
.err-toast{{
  position:fixed;top:72px;right:16px;z-index:600;
  max-width:320px;
  background:rgba(30,10,10,.95);
  border:1px solid rgba(255,50,50,.4);
  border-radius:10px;padding:14px 18px;
  display:flex;gap:12px;align-items:flex-start;
  animation:slideInRight .35s ease;
  box-shadow:0 8px 32px rgba(0,0,0,.5);
}}
@keyframes slideInRight{{from{{opacity:0;transform:translateX(20px)}}to{{opacity:1;transform:translateX(0)}}}}
.err-toast-icon{{font-size:18px;flex-shrink:0;margin-top:1px}}
.err-toast-body{{flex:1}}
.err-toast-title{{font-size:12px;font-weight:700;color:#FF6060;margin-bottom:3px}}
.err-toast-msg{{font-size:11px;color:var(--muted);line-height:1.5}}
.err-toast-close{{
  font-size:16px;color:var(--muted);cursor:pointer;
  line-height:1;padding:2px;transition:color .2s;flex-shrink:0;
}}
.err-toast-close:hover{{color:var(--text)}}

/* Warn-Toast (gelb) */
.warn-toast{{
  border-color:rgba(255,200,0,.4)!important;
  background:rgba(30,25,0,.95)!important;
}}
.warn-toast .err-toast-title{{color:#FFD700!important}}

/* ═══ HEADER ═══════════════════════════════════════════════════════════════ */
header{{
  background:linear-gradient(90deg,#0A0D1E,#0F1428);
  border-bottom:1px solid var(--border);
  padding:0 16px;
  height:48px;
  display:flex;align-items:center;gap:10px;
  flex-shrink:0;
  position:relative;z-index:10;
}}
header::after{{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,#FF6B0040,#9B40FF30,transparent);
}}
.hdr-logo{{
  display:flex;align-items:center;gap:8px;margin-right:6px;
}}
.hdr-logo-icon{{font-size:1.3rem;filter:drop-shadow(0 0 6px #FF6B0080)}}
.hdr-logo-text{{
  font-size:.95rem;font-weight:900;letter-spacing:.02em;
  background:linear-gradient(90deg,var(--accent3),var(--accent));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.badge{{
  background:#ffffff10;border:1px solid #ffffff18;border-radius:20px;
  padding:3px 10px;font-size:.68rem;color:var(--muted);
  font-family:var(--mono);letter-spacing:.03em;
  white-space:nowrap;
}}
.badge.active{{background:#FF6B0020;border-color:#FF6B0060;color:var(--accent3)}}
.badge.blue{{background:#00C8FF15;border-color:#00C8FF50;color:var(--blue)}}
.badge.green{{background:#2EFF9A15;border-color:#2EFF9A50;color:var(--green)}}
.hdr-spacer{{flex:1}}

/* ═══ MAIN LAYOUT ══════════════════════════════════════════════════════════ */
.main{{
  display:grid;
  grid-template-columns:270px 1fr 240px 290px;
  flex:1;overflow:hidden;
  gap:0;
}}

/* ═══ PANELS ═══════════════════════════════════════════════════════════════ */
.panel{{
  background:var(--panel);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}}
.panel.right{{border-right:none;border-left:1px solid var(--border)}}
.panel-inner{{
  flex:1;overflow-y:auto;padding:12px 10px;
  scrollbar-width:thin;scrollbar-color:var(--border2) transparent;
}}
.panel-inner::-webkit-scrollbar{{width:3px}}
.panel-inner::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px}}

/* ═══ TABS ═════════════════════════════════════════════════════════════════ */
.tabs{{
  display:flex;background:var(--bg2);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}}
.tab{{
  flex:1;padding:9px 4px;text-align:center;
  font-size:.65rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
  color:var(--dim);cursor:pointer;transition:.15s;
  border-bottom:2px solid transparent;
  font-family:var(--font);
}}
.tab:hover{{color:var(--text)}}
.tab.active{{color:var(--accent3);border-bottom-color:var(--accent)}}

/* ═══ SECTION HEADERS ══════════════════════════════════════════════════════ */
.sect{{margin-bottom:14px}}
.sect-title{{
  font-size:.6rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:var(--dim);
  margin-bottom:8px;padding-bottom:4px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:5px;
}}

/* ═══ LESSON CARDS ═════════════════════════════════════════════════════════ */
.lesson-card{{
  background:var(--panel2);
  border:1px solid var(--border);border-radius:var(--radius);
  padding:9px 11px;margin-bottom:6px;
  cursor:pointer;transition:.18s;
  position:relative;overflow:hidden;
}}
.lesson-card::before{{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--border2);border-radius:2px 0 0 2px;
  transition:background .18s;
}}
.lesson-card:hover{{border-color:var(--border2);background:#181e38}}
.lesson-card:hover::before{{background:var(--accent)}}
.lesson-card.active{{border-color:#FF6B0060;background:#1A1230}}
.lesson-card.active::before{{background:var(--accent)}}
.lesson-card.done{{border-color:#2EFF9A40;opacity:.6}}
.lesson-card.done::before{{background:var(--green)}}
.lesson-card.spanish{{
  border-color:rgba(255,180,0,.25);
  background:linear-gradient(135deg,rgba(255,100,0,.06),rgba(220,50,0,.04));
}}
.lesson-card.spanish::before{{background:linear-gradient(180deg,#FF6B00,#FFB400)}}
.lesson-card.latin{{
  border-color:rgba(0,200,120,.22);
  background:linear-gradient(135deg,rgba(0,200,120,.05),rgba(0,160,200,.04));
}}
.lesson-card.latin::before{{background:linear-gradient(180deg,#00C878,#00A8C8)}}
.lesson-card.locked{{
  opacity:.45;cursor:not-allowed;filter:grayscale(.4);
}}
.lesson-card.locked::before{{background:var(--muted)}}
.lesson-flair-badge{{
  display:inline-flex;align-items:center;gap:3px;
  font-size:.56rem;font-weight:700;letter-spacing:.06em;
  padding:1px 7px;border-radius:8px;margin-left:6px;
  text-transform:uppercase;
}}
.flair-spanish{{background:rgba(255,100,0,.18);color:#FFB400;border:1px solid rgba(255,180,0,.3)}}
.flair-latin{{background:rgba(0,200,120,.15);color:#00D490;border:1px solid rgba(0,200,120,.25)}}
.flair-locked{{background:rgba(100,100,120,.2);color:var(--muted);border:1px solid var(--border2)}}
.lesson-title{{font-size:.76rem;font-weight:700;color:var(--text);margin-bottom:2px}}
.lesson-meta{{font-size:.62rem;color:var(--muted)}}
.lesson-level{{
  display:inline-block;background:#FF6B0025;border-radius:6px;
  padding:1px 7px;font-size:.58rem;color:var(--accent3);
  font-family:var(--mono);margin-bottom:3px;
}}
/* Spanish/Latin Tab-Highlight-Badge */
.tab-unlock-badge{{
  display:inline-block;background:linear-gradient(90deg,#FF6B00,#FFB400);
  color:#fff;font-size:.5rem;font-weight:800;
  padding:1px 5px;border-radius:6px;margin-left:4px;
  animation:tabpulse 2s ease infinite;vertical-align:middle;
}}
@keyframes tabpulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.7;transform:scale(.92)}}}}
/* Spanish Section Header */
.spanish-section-header{{
  display:flex;align-items:center;gap:8px;
  padding:8px 0 4px;margin-top:8px;
  border-top:1px solid rgba(255,180,0,.2);
}}
.spanish-section-title{{
  font-size:.65rem;font-weight:800;letter-spacing:.1em;
  text-transform:uppercase;
  background:linear-gradient(90deg,#FF6B00,#FFB400);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.spanish-section-sub{{font-size:.58rem;color:var(--muted)}}
.unlock-notice{{
  font-size:.62rem;color:var(--muted);
  background:rgba(255,180,0,.07);border:1px solid rgba(255,180,0,.15);
  border-radius:6px;padding:7px 10px;margin-bottom:6px;
  display:flex;align-items:center;gap:6px;
}}

/* ═══ PÄDAGOGISCHES SYSTEM v11 ══════════════════════════════════════════════ */
/* Phase-Stepper */
.phase-stepper{{
  display:flex;gap:0;margin:0 0 10px 0;
  border-radius:8px;overflow:hidden;
  border:1px solid var(--border);
}}
.phase-step{{
  flex:1;padding:6px 4px;text-align:center;
  font-size:.56rem;font-weight:700;color:var(--muted);
  background:var(--panel2);
  border-right:1px solid var(--border);
  transition:.2s;cursor:default;position:relative;
}}
.phase-step:last-child{{border-right:none}}
.phase-step .step-icon{{font-size:.8rem;display:block;margin-bottom:2px}}
.phase-step.done{{background:#2EFF9A12;color:var(--green)}}
.phase-step.done::after{{content:'✓';position:absolute;top:2px;right:3px;font-size:.5rem;color:var(--green)}}
.phase-step.active{{background:#FF6B0018;color:var(--accent3);border-color:#FF6B0060}}
/* Phase Card */
.phase-card{{
  background:var(--panel2);border:1px solid var(--border);
  border-radius:10px;padding:12px;margin-bottom:8px;
  border-left:3px solid var(--accent);
}}
.phase-card.exercise{{border-left-color:var(--blue)}}
.phase-card.song{{border-left-color:var(--green)}}
.phase-card.xp{{border-left-color:var(--yellow)}}
.phase-label{{
  font-size:.62rem;font-weight:900;color:var(--accent3);
  text-transform:uppercase;letter-spacing:.04em;margin-bottom:3px;
}}
.phase-title{{
  font-size:.84rem;font-weight:800;color:var(--text);margin-bottom:6px;
}}
.phase-content{{
  font-size:.7rem;color:var(--muted);line-height:1.65;margin-bottom:6px;
}}
.phase-tips{{margin:6px 0;padding:0;list-style:none}}
.phase-tips li{{
  font-size:.66rem;color:var(--text);padding:3px 0 3px 14px;
  position:relative;
}}
.phase-tips li::before{{content:'›';position:absolute;left:0;color:var(--accent3)}}
/* Phase Metadaten */
.phase-meta{{
  display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;
}}
.phase-tag{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:5px;padding:2px 7px;
  font-size:.6rem;color:var(--muted);font-family:var(--mono);
}}
.phase-tag.bpm{{border-color:#00C8FF40;color:var(--blue)}}
.phase-tag.strum{{border-color:#9B40FF40;color:var(--purple)}}
.phase-tag.chord{{border-color:#FF6B0040;color:var(--accent3)}}
/* Phase Fortschrittsbalken */
.phase-prog-row{{
  display:flex;align-items:center;gap:7px;margin:8px 0;
}}
.phase-prog-track{{
  flex:1;height:5px;background:#ffffff10;border-radius:3px;overflow:hidden;
}}
.phase-prog-fill{{
  height:100%;border-radius:3px;transition:width .3s ease;
  background:linear-gradient(90deg,var(--accent),var(--accent3));
}}
/* XP Bar */
.xp-row{{
  display:flex;align-items:center;gap:8px;
  background:var(--bg2);border-radius:8px;
  padding:8px 10px;border:1px solid var(--border);margin:8px 0;
}}
.xp-label{{font-size:.65rem;color:var(--muted);white-space:nowrap}}
.xp-track{{flex:1;height:8px;background:#ffffff10;border-radius:4px;overflow:hidden}}
.xp-fill{{
  height:100%;border-radius:4px;transition:width .5s ease;
  background:linear-gradient(90deg,#FF6B00,#FFD600,#2EFF9A);
}}
.xp-val{{font-size:.7rem;font-weight:900;color:var(--yellow);white-space:nowrap;font-family:var(--mono)}}
/* Level Badge */
.level-badge{{
  display:inline-flex;align-items:center;gap:5px;
  background:linear-gradient(135deg,#FF6B0025,#9B40FF20);
  border:1px solid #FF6B0040;border-radius:8px;
  padding:4px 10px;font-size:.7rem;font-weight:900;
}}
.level-badge .lv-num{{
  font-size:1.0rem;font-weight:900;
  background:linear-gradient(135deg,var(--accent3),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-family:var(--mono);
}}
/* Level-Up Overlay */
#levelup-overlay{{
  position:fixed;inset:0;z-index:9999;
  display:none;align-items:center;justify-content:center;
  background:#00000088;backdrop-filter:blur(6px);
}}
#levelup-overlay.show{{display:flex}}
.levelup-card{{
  background:var(--bg2);border:2px solid var(--yellow);
  border-radius:20px;padding:32px 40px;text-align:center;
  animation:levelup-pop .4s cubic-bezier(.2,1.4,.4,1);
  max-width:380px;
}}
@keyframes levelup-pop{{
  from{{transform:scale(.5);opacity:0}}
  to{{transform:scale(1);opacity:1}}
}}
.levelup-star{{font-size:3.5rem;animation:spin 1s ease infinite}}
@keyframes spin{{0%{{transform:rotate(0deg)}}100%{{transform:rotate(360deg)}}}}
.levelup-title{{font-size:1.6rem;font-weight:900;color:var(--yellow);margin:10px 0 5px}}
.levelup-sub{{font-size:.85rem;color:var(--muted);margin-bottom:18px}}
/* Phase Next Button */
.btn-phase-next{{
  width:100%;padding:9px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),#FF9240);
  color:#fff;font-weight:900;font-size:.78rem;
  border:none;cursor:pointer;font-family:var(--font);
  transition:.15s;letter-spacing:.02em;
  box-shadow:0 2px 12px #FF6B0030;
}}
.btn-phase-next:hover{{transform:translateY(-1px);box-shadow:0 4px 18px #FF6B0050}}
.btn-phase-next.song{{background:linear-gradient(135deg,var(--green),#00AA66)}}
.btn-phase-next.xp{{background:linear-gradient(135deg,var(--yellow),#FFA000)}}
/* Akkord-Chips in Phase */
.chord-chips{{display:flex;flex-wrap:wrap;gap:4px;margin:5px 0}}
.chord-chip{{
  background:#FF6B0015;border:1px solid #FF6B0035;
  border-radius:5px;padding:2px 8px;
  font-size:.65rem;font-weight:700;color:var(--accent3);
  cursor:pointer;transition:.12s;font-family:var(--mono);
}}
.chord-chip:hover{{background:#FF6B0030;border-color:#FF6B0070}}

/* ═══ BARS & METRICS ═══════════════════════════════════════════════════════ */
.bar-row{{display:flex;align-items:center;gap:6px;margin:4px 0}}
.bar-label{{width:70px;font-size:.65rem;color:var(--muted);flex-shrink:0}}
.bar-track{{
  flex:1;background:#0F1428;border-radius:4px;height:7px;
  overflow:hidden;border:1px solid var(--border);
}}
.bar-fill{{height:100%;border-radius:4px;transition:width .25s ease}}
.bar-fill.or{{background:linear-gradient(90deg,#C04400,var(--accent))}}
.bar-fill.ok{{background:linear-gradient(90deg,#18804A,var(--green))}}
.bar-fill.info{{background:linear-gradient(90deg,#006090,var(--blue))}}
.bar-fill.purple{{background:linear-gradient(90deg,#5A1890,var(--purple))}}
.bar-val{{width:32px;font-size:.65rem;color:var(--text);text-align:right;font-family:var(--mono)}}

/* ═══ ANALYSIS CARDS ═══════════════════════════════════════════════════════ */
.analysis-card{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:8px;padding:8px 10px;margin-bottom:5px;
  display:flex;align-items:center;justify-content:space-between;
}}
.analysis-title{{font-size:.64rem;color:var(--muted);font-weight:600}}
.analysis-score{{font-size:1rem;font-weight:800;font-family:var(--mono)}}
.analysis-score.good{{color:var(--green)}}
.analysis-score.ok{{color:var(--yellow)}}
.analysis-score.bad{{color:var(--red)}}
.buzz-warn{{
  font-size:10px;font-weight:700;color:var(--red);
  animation:buzzpulse 0.8s ease infinite;
}}
@keyframes buzzpulse{{
  0%,100%{{opacity:1}} 50%{{opacity:0.4}}
}}
/* Flair Filter Buttons */
.flair-filter-btn{{
  font-size:.6rem;padding:3px 12px;border-radius:8px;
  border:1px solid var(--border2);background:var(--panel2);
  color:var(--muted);cursor:pointer;transition:all .2s;
  font-family:var(--font);
}}
.flair-filter-btn.active{{
  border-color:var(--accent);color:var(--text);background:#1A1A35;
}}
.flair-filter-btn[data-filter="Spanish"].active{{
  border-color:rgba(255,180,0,.5);color:#FFB400;background:rgba(255,100,0,.1);
}}
.flair-filter-btn[data-filter="Latin"].active{{
  border-color:rgba(0,200,120,.4);color:#00D490;background:rgba(0,200,120,.08);
}}

/* ═══ CHORD DISPLAY ════════════════════════════════════════════════════════ */
.chord-big{{
  background:linear-gradient(135deg,#120A30,#1A1040);
  border:1px solid #4A2080;border-radius:var(--radius);
  padding:12px;text-align:center;margin-bottom:10px;
  box-shadow:0 4px 20px #9B40FF15;
}}
.chord-big-name{{font-size:1.6rem;font-weight:900;color:var(--accent3);font-family:var(--font)}}
.chord-big-diff{{font-size:.78rem;color:var(--accent);margin:3px 0}}
.chord-big-tip{{font-size:.68rem;color:var(--muted);line-height:1.5;margin-top:5px}}

/* ═══ BUTTONS ══════════════════════════════════════════════════════════════ */
.btn{{
  display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border-radius:8px;border:none;cursor:pointer;
  font-family:var(--font);font-weight:700;font-size:.74rem;
  letter-spacing:.03em;transition:.15s;padding:8px 14px;
}}
.btn-primary{{
  background:linear-gradient(135deg,#C04400,#FF6B00);
  color:#fff;box-shadow:0 2px 16px #FF6B0040;
}}
.btn-primary:hover{{background:linear-gradient(135deg,#D05000,#FF7A10);transform:translateY(-1px);box-shadow:0 4px 24px #FF6B0060}}
.btn-primary:active{{transform:translateY(0)}}
.btn-secondary{{
  background:var(--panel2);color:var(--muted);
  border:1px solid var(--border);
}}
.btn-secondary:hover{{border-color:var(--border2);color:var(--text)}}
.btn-danger{{
  background:#300815;color:var(--red);
  border:1px solid #FF335530;
}}
.btn-danger:hover{{background:#400A1C;border-color:var(--red)}}
.btn-blue{{
  background:linear-gradient(135deg,#004870,#006CA0);
  color:var(--blue);box-shadow:0 2px 16px #00C8FF25;
}}
.btn-blue:hover{{background:linear-gradient(135deg,#005880,#007AB0)}}
.btn-sm{{padding:5px 10px;font-size:.68rem;border-radius:6px}}
.btn-wide{{width:100%}}

/* ═══ METRONOME BPM BUTTONS ════════════════════════════════════════════════ */
.bpm-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px;margin:10px 0}}
.bpm-btn{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:7px;padding:6px 4px;
  font-size:.72rem;font-weight:700;color:var(--text);
  cursor:pointer;text-align:center;transition:.12s;
  font-family:var(--mono);
}}
.bpm-btn:hover{{border-color:var(--accent);color:var(--accent3);background:#1A1030}}

/* ═══ AUDIO / PITCH ════════════════════════════════════════════════════════ */
.pitch-box{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;padding:10px;text-align:center;margin-bottom:8px;
}}
.pitch-note{{font-size:1.8rem;font-weight:900;color:var(--accent3);font-family:var(--font);line-height:1}}
.pitch-hz{{font-size:.65rem;color:var(--muted);font-family:var(--mono);margin-top:2px}}
.pitch-meter{{
  height:18px;
  background:linear-gradient(90deg,#FF3355,#FF7A00,#2EFF9A,#FF7A00,#FF3355);
  border-radius:4px;position:relative;margin:6px 0;overflow:hidden;
}}
.pitch-needle{{
  position:absolute;top:1px;bottom:1px;width:3px;background:#fff;
  border-radius:2px;transform:translateX(-50%);transition:left .06s;
  box-shadow:0 0 8px #ffffff90;
}}
.spectrum{{display:flex;align-items:flex-end;height:28px;gap:1px;margin:4px 0}}
.spec-bar{{flex:1;min-width:2px;border-radius:1px 1px 0 0;transition:height .04s}}

/* ═══ STRINGS ══════════════════════════════════════════════════════════════ */
.strings-row{{display:flex;gap:2px;justify-content:space-between;margin:4px 0}}
.str-col{{display:flex;flex-direction:column;align-items:center;gap:1px}}
.str-bar{{
  width:22px;height:34px;background:#0F1428;border-radius:4px;
  overflow:hidden;position:relative;border:1px solid var(--border);
}}
.str-fill{{position:absolute;bottom:0;left:0;right:0;transition:height .1s;border-radius:3px}}
.str-note{{font-size:.54rem;color:var(--muted);font-family:var(--mono)}}
.str-label{{font-size:.58rem;color:var(--dim)}}

/* ═══ STRUM ════════════════════════════════════════════════════════════════ */
.strum-big{{font-size:1.8rem;text-align:center;color:var(--accent);transition:all .08s}}
.strum-big.flash{{color:var(--green);text-shadow:0 0 16px var(--green)}}

/* ═══ FEEDBACK ═════════════════════════════════════════════════════════════ */
#feedback-area{{
  position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);
  pointer-events:none;
  display:flex;flex-direction:column;align-items:center;gap:7px;z-index:10;
}}
.feedback-msg{{
  background:#000000AA;border-radius:24px;padding:7px 18px;
  font-size:.82rem;color:#fff;backdrop-filter:blur(6px);
  border:1px solid #ffffff20;animation:fadeup .28s ease;white-space:nowrap;
  font-weight:600;
}}
.feedback-msg.bold{{background:var(--accent);font-weight:800;font-size:.92rem}}
.feedback-msg.ok{{background:#143828;border-color:var(--green);color:var(--green)}}
.feedback-msg.warn{{background:#3A1010;border-color:var(--red);color:#FFB0B0}}
@keyframes fadeup{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}

/* ═══ STATS ════════════════════════════════════════════════════════════════ */
.stat-row{{
  display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid var(--border);font-size:.72rem;
}}
.stat-row:last-child{{border-bottom:none}}
.stat-k{{color:var(--muted)}}
.stat-v{{color:var(--text);font-weight:700;font-family:var(--mono)}}

/* ═══ CHAT ═════════════════════════════════════════════════════════════════ */
#chat-box{{
  background:var(--bg2);border-top:1px solid var(--border);
  padding:8px 10px;flex-shrink:0;
}}
.chat-history{{
  max-height:110px;overflow-y:auto;margin-bottom:6px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}}
.chat-msg{{font-size:.7rem;line-height:1.5;margin:2px 0;padding:4px 10px;border-radius:8px}}
.chat-msg.luna{{background:#1A0D30;color:#FFD0A0;border-left:3px solid var(--accent)}}
.chat-msg.user{{background:#0D1530;color:#B0D0FF;text-align:right;border-right:3px solid var(--blue)}}
.chat-input-row{{display:flex;gap:6px;align-items:center}}
#chat-input{{
  flex:1;background:var(--panel2);border:1px solid var(--border);
  border-radius:20px;padding:7px 14px;color:var(--text);font-size:.75rem;
  outline:none;font-family:var(--font);transition:.15s;
}}
#chat-input:focus{{border-color:#FF6B0060;box-shadow:0 0 0 2px #FF6B0015}}
#chat-send{{
  background:linear-gradient(135deg,#C04400,#FF6B00);
  border:none;border-radius:20px;padding:7px 14px;
  color:#fff;cursor:pointer;font-size:.78rem;font-weight:700;
  font-family:var(--font);flex-shrink:0;transition:.15s;
}}
#chat-send:hover{{background:linear-gradient(135deg,#D05000,#FF7A10)}}

/* ═══ CENTER 3D STAGE ══════════════════════════════════════════════════════ */
#center{{
  position:relative;
  background:radial-gradient(ellipse 80% 100% at 50% 80%, #0C0620, #070910);
  overflow:hidden;display:flex;flex-direction:column;
}}
#teacher-canvas{{width:100%;flex:1;display:block;min-height:0}}

/* ═══ SPEECH BUBBLE ════════════════════════════════════════════════════════ */
#speech-bubble{{
  position:absolute;
  bottom:170px;left:50%;transform:translateX(-50%);
  background:linear-gradient(135deg,#16093Aee,#1E0E5Aee);
  border:1px solid #FF6B0070;border-radius:18px;
  padding:11px 18px;max-width:440px;text-align:center;
  font-size:.82rem;line-height:1.6;color:#FFE8C0;
  box-shadow:0 8px 32px #00000060,0 0 24px #FF6B0025;
  display:none;backdrop-filter:blur(12px);
  animation:bubble-in .25s ease;z-index:30;
}}
@keyframes bubble-in{{from{{opacity:0;transform:translateX(-50%) translateY(8px)}}to{{opacity:1;transform:translateX(-50%) translateY(0)}}}}
#speech-bubble::after{{
  content:'';position:absolute;bottom:-10px;left:50%;
  transform:translateX(-50%);
  border:10px solid transparent;
  border-top-color:#FF6B0070;border-bottom:none;
}}

/* ═══ EMOTION TAG ══════════════════════════════════════════════════════════ */
#emotion-tag{{
  position:absolute;top:10px;right:12px;
  background:#00000070;backdrop-filter:blur(8px);
  border-radius:20px;padding:4px 12px;
  font-size:.68rem;color:var(--accent3);
  border:1px solid #ffffff15;z-index:20;
  font-family:var(--font);font-weight:600;
}}

/* ═══ MIC BUTTON ═══════════════════════════════════════════════════════════ */
#active-listen-btn:hover{{border-color:#00DD88!important;color:#00DD88!important}}
#mic-button{{
  position:absolute;bottom:170px;right:12px;
  width:46px;height:46px;border-radius:50%;
  border:2px solid var(--border2);
  background:linear-gradient(135deg,#0D1230,#141840);
  cursor:pointer;font-size:1.2rem;
  display:flex;align-items:center;justify-content:center;
  z-index:25;transition:.2s;
  box-shadow:0 4px 16px #00000050;
}}
#mic-button:hover{{border-color:var(--accent);box-shadow:0 4px 24px #FF6B0030}}
#mic-button.active{{
  border-color:var(--red);background:#280818;
  animation:mic-pulse .8s ease infinite;
  box-shadow:0 0 24px #FF335540;
}}
@keyframes mic-pulse{{
  0%,100%{{box-shadow:0 0 20px #FF335540}}
  50%{{box-shadow:0 0 35px #FF335570}}
}}
#mic-interim-text{{
  position:absolute;top:46%;left:50%;
  transform:translate(-50%,-50%);
  background:#000000A0;border:1px solid #FF6B0050;
  border-radius:14px;padding:9px 18px;
  font-size:.82rem;color:#FFD090;display:none;
  backdrop-filter:blur(12px);text-align:center;
  max-width:340px;z-index:26;
}}

/* ═══ SONG BAR ═════════════════════════════════════════════════════════════ */
#song-bar{{
  position:absolute;top:0;left:0;right:0;
  background:linear-gradient(135deg,#08101Eee,#0C1830ee);
  backdrop-filter:blur(8px);padding:8px 14px;
  border-bottom:1px solid var(--blue);display:none;z-index:20;
}}

/* ═══ CAMERA STRIP ═════════════════════════════════════════════════════════ */
#cam-strip{{
  position:absolute;bottom:0;left:0;right:0;height:125px;
  background:linear-gradient(transparent,#00000095);
  display:flex;align-items:flex-end;padding:8px 10px;gap:10px;
  z-index:15;
}}
#cam-img{{
  width:162px;height:112px;object-fit:cover;
  border-radius:8px;border:2px solid var(--border2);flex-shrink:0;
  box-shadow:0 4px 16px #00000060;
}}

/* ═══ SESSION FEEDBACK ═════════════════════════════════════════════════════ */
.session-fb{{
  background:linear-gradient(135deg,#140B38ee,#0D1530ee);
  border:1px solid var(--accent);border-radius:14px;padding:16px;
  display:none;position:absolute;inset:16px;z-index:40;
  backdrop-filter:blur(16px);overflow-y:auto;
  box-shadow:0 16px 64px #00000080;
}}
.session-fb h2{{color:var(--accent3);margin-bottom:12px;font-size:1.1rem}}
.fb-section{{margin-bottom:10px;padding:10px;background:#ffffff08;border-radius:10px;border:1px solid var(--border)}}
.fb-section h4{{color:var(--accent);font-size:.76rem;margin-bottom:5px}}
.fb-section p{{font-size:.73rem;color:var(--text);line-height:1.6}}
.fb-grade{{font-size:3rem;font-weight:900;text-align:center;margin:10px 0;filter:drop-shadow(0 0 16px #FF6B0050)}}

/* ═══ SCROLLBAR GLOBAL ═════════════════════════════════════════════════════ */
::-webkit-scrollbar{{width:3px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px}}

/* ═══ METRO DOT ════════════════════════════════════════════════════════════ */
#metro-dot{{
  width:70px;height:70px;border-radius:50%;
  background:var(--bg2);border:3px solid var(--border2);
  margin:0 auto;transition:all .06s;
  display:flex;align-items:center;justify-content:center;
  font-size:1.3rem;font-weight:900;color:var(--muted);
  font-family:var(--mono);
  box-shadow:0 4px 24px #00000050;
}}


/* ═══ AVATAR PANEL ═════════════════════════════════════════════════════════ */
#avatar-panel{{
  background:radial-gradient(ellipse at 50% 30%,#110830 0%,#080510 100%);
  border-left:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;position:relative;
}}
#avatar-canvas{{width:100%;flex:1;display:block;min-height:0}}
#av-infobar{{
  padding:8px 10px;background:var(--bg2);
  border-top:1px solid var(--border);flex-shrink:0;
}}
.av-name{{
  font-size:.84rem;font-weight:900;
  background:linear-gradient(90deg,var(--accent3),var(--accent));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:2px;
}}
.av-status{{font-size:.62rem;color:var(--muted);margin-bottom:7px}}
.av-btns{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}
.av-btn{{
  background:var(--panel2);border:1px solid var(--border);
  border-radius:6px;padding:5px 4px;
  font-size:.6rem;font-weight:700;color:var(--muted);
  cursor:pointer;text-align:center;transition:.12s;
  font-family:var(--font);
}}
.av-btn:hover{{border-color:var(--accent);color:var(--accent3);background:#160A28}}
.av-btn.active{{background:#FF6B0022;border-color:#FF6B0060;color:var(--accent3)}}

/* ═══ MISC UTILITY ═════════════════════════════════════════════════════════ */
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.lib-item{{
  display:flex;justify-content:space-between;align-items:center;
  padding:5px 8px;border-radius:6px;font-size:.7rem;cursor:pointer;
  transition:.12s;border:1px solid transparent;margin-bottom:2px;
}}
.lib-item:hover{{background:var(--panel2);border-color:var(--border);color:var(--text)}}
.lib-item-name{{color:var(--text);font-weight:600}}
.lib-item-meta{{color:var(--muted);font-size:.62rem}}
#lib-search{{
  width:100%;background:var(--bg2);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;color:var(--text);
  font-size:.73rem;outline:none;margin-bottom:8px;font-family:var(--font);
  transition:.15s;
}}
#lib-search:focus{{border-color:#FF6B0060}}
</style>
</head>
<body>

<!-- ═══ WILLKOMMENS-OVERLAY ═══════════════════════════════════════════════ -->
<!-- ═══ ONBOARDING (5-Schritte Wizard) ════════════════════════════════════ -->
<div id="onboarding-overlay">
<div class="ob-card">

  <!-- Logo -->
  <div class="ob-logo">Note<span style="color:var(--blue)">IQ</span></div>
  <div style="font-size:12px;color:var(--muted);margin-bottom:4px">v16.0 · KI-Musiklehrer</div>

  <!-- Fortschritts-Dots -->
  <div class="ob-step-indicator" id="ob-dots">
    <div class="ob-step-dot active" id="ob-dot-0"></div>
    <div class="ob-step-dot" id="ob-dot-1"></div>
    <div class="ob-step-dot" id="ob-dot-2"></div>
    <div class="ob-step-dot" id="ob-dot-3"></div>
    <div class="ob-step-dot" id="ob-dot-4"></div>
  </div>

  <!-- ── Schritt 0: Willkommen ── -->
  <div class="ob-step active" id="ob-step-0">
    <div class="ob-step-label">Schritt 1 / 5</div>
    <div class="ob-step-title">Willkommen bei NoteIQ 🎸</div>
    <div class="ob-step-desc">
      Dein persönlicher <strong>KI-Musiklehrer</strong> analysiert in Echtzeit dein Spiel —
      Akkorde, Haltung, Rhythmus und Ton-Sauberkeit.<br><br>
      Dieses kurze Setup dauert nur <strong>~1 Minute</strong> und stellt sicher,
      dass alles perfekt funktioniert.
    </div>
    <div class="ob-features">
      <div class="ob-feature">
        <div class="ob-feature-icon">🎙️</div>
        <div class="ob-feature-title">Echtzeit-Analyse</div>
        <div class="ob-feature-desc">Akkorderkennung, Ton-Sauberkeit & Schnarren-Detection</div>
      </div>
      <div class="ob-feature">
        <div class="ob-feature-icon">📹</div>
        <div class="ob-feature-title">Kamera-Tracking</div>
        <div class="ob-feature-desc">Handgelenk-Winkel, Finger-Krümmung & Haltungsanalyse</div>
      </div>
      <div class="ob-feature">
        <div class="ob-feature-icon">🤖</div>
        <div class="ob-feature-title">KI-Gespräch</div>
        <div class="ob-feature-desc">Frag NoteIQ per Sprache oder Chat — powered by Groq AI</div>
      </div>
      <div class="ob-feature">
        <div class="ob-feature-icon">🌹</div>
        <div class="ob-feature-title">Spanish / Latin Flair</div>
        <div class="ob-feature-desc">Flamenco & Bossa Nova — ab Level 3 freigeschaltet</div>
      </div>
    </div>
  </div>

  <!-- ── Schritt 1: System-Check ── -->
  <div class="ob-step" id="ob-step-1">
    <div class="ob-step-label">Schritt 2 / 5</div>
    <div class="ob-step-title">System-Check ⚙️</div>
    <div class="ob-step-desc">
      Ich prüfe automatisch ob alle Komponenten bereit sind.
      Grüne Häkchen bedeuten volle Funktionalität — gelb/rot zeigt Optionen an.
    </div>
    <div class="ob-checks" id="ob-checks-list">
      <!-- Wird von JS befüllt -->
      <div class="ob-check checking">
        <div class="ob-check-icon">⏳</div>
        <div class="ob-check-label">
          <div class="ob-check-name">Prüfe System…</div>
          <div class="ob-check-msg">Bitte einen Moment warten</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── Schritt 2: Mikrofon-Erlaubnis ── -->
  <div class="ob-step" id="ob-step-2">
    <div class="ob-step-label">Schritt 3 / 5</div>
    <div class="ob-step-title">Mikrofon & Audio 🎙️</div>
    <div class="ob-step-desc">
      NoteIQ benötigt Zugriff auf dein <strong>Mikrofon</strong> um dein Gitarrenspiel
      in Echtzeit zu analysieren. Klick unten auf „Mikrofon erlauben" und bestätige
      die Browser-Anfrage.<br><br>
      <strong>Datenschutz:</strong> Das Mikrofon-Signal wird
      <strong>nur lokal verarbeitet</strong> — nichts wird hochgeladen.
    </div>
    <div class="ob-checks" id="ob-mic-status">
      <div class="ob-check" id="ob-mic-check">
        <div class="ob-check-icon">🎙️</div>
        <div class="ob-check-label">
          <div class="ob-check-name">Mikrofon-Zugriff</div>
          <div class="ob-check-msg" id="ob-mic-msg">Noch nicht angefragt</div>
        </div>
        <button class="ob-check-fix" onclick="obRequestMic()">Erlauben</button>
      </div>
      <div class="ob-check" id="ob-audio-ctx-check" style="display:none">
        <div class="ob-check-icon">🔊</div>
        <div class="ob-check-label">
          <div class="ob-check-name">Audio-Ausgabe</div>
          <div class="ob-check-msg" id="ob-audio-ctx-msg">Wird beim Starten aktiviert</div>
        </div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px 14px;background:rgba(0,153,255,.06);
                border:1px solid rgba(0,153,255,.2);border-radius:8px;
                font-size:11px;color:var(--muted);line-height:1.6">
      💡 <strong style="color:var(--text)">Kein Mikrofon?</strong>
      NoteIQ funktioniert auch ohne — du verlierst nur die Ton-Analyse.
      Klick einfach auf „Überspringen".
    </div>
  </div>

  <!-- ── Schritt 3: Schnell-Setup ── -->
  <div class="ob-step" id="ob-step-3">
    <div class="ob-step-label">Schritt 4 / 5</div>
    <div class="ob-step-title">Dein Profil 👤</div>
    <div class="ob-step-desc">
      Das Profil wurde bereits im Terminal eingerichtet.
      Hier siehst du deine aktuellen Einstellungen.
    </div>
    <div id="ob-profile-display" class="ob-checks">
      <!-- Wird aus STATE befüllt -->
    </div>
    <div style="margin-top:16px">
      <div class="ob-label">Bevorzugte Sprache</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap" id="ob-lang-btns">
        <button class="ob-instr-card" data-lang="de" onclick="obSetLang('de')" style="padding:10px 16px;text-align:left">
          🇩🇪 Deutsch
        </button>
        <button class="ob-instr-card" data-lang="en" onclick="obSetLang('en')" style="padding:10px 16px;text-align:left">
          🇬🇧 English
        </button>
        <button class="ob-instr-card" data-lang="es" onclick="obSetLang('es')" style="padding:10px 16px;text-align:left">
          🇪🇸 Español
        </button>
        <button class="ob-instr-card" data-lang="fr" onclick="obSetLang('fr')" style="padding:10px 16px;text-align:left">
          🇫🇷 Français
        </button>
      </div>
    </div>
  </div>

  <!-- ── Schritt 4: Bedienungsanleitung ── -->
  <div class="ob-step" id="ob-step-4">
    <div class="ob-step-label">Schritt 5 / 5</div>
    <div class="ob-step-title">So nutzt du NoteIQ 🎓</div>
    <div class="ob-step-desc">
      Kurze Übersicht — du kannst diese Anleitung jederzeit über den
      <strong>H-Knopf</strong> in der App aufrufen.
    </div>

    <div style="display:flex;flex-direction:column;gap:14px;margin:12px 0">

      <div style="background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px 16px">
        <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:8px">📱 Browser-Fenster (Web-UI)</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.8">
          Das ist die Haupt-Oberfläche. Links siehst du die <strong style="color:var(--text)">Lektionen</strong> und die Echtzeit-Analyse.
          In der Mitte läuft der <strong style="color:var(--text)">3D-Avatar</strong>.
          Rechts die detaillierten <strong style="color:var(--text)">Audio/Poly/Stats-Tabs</strong>.
          Unten findest du den <strong style="color:var(--text)">Chat</strong> — schreib oder sprich mit NoteIQ.
        </div>
      </div>

      <div style="background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px 16px">
        <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:10px">⌨️ Tastatur-Shortcuts (im App-Fenster)</div>
        <div class="ob-keys">
          <div class="ob-key">Q / ESC</div><div class="ob-key-desc">Beenden</div>
          <div class="ob-key">H</div><div class="ob-key-desc">Hilfe anzeigen</div>
          <div class="ob-key">L</div><div class="ob-key-desc">Sprache wechseln (DE/EN)</div>
          <div class="ob-key">1 / 2</div><div class="ob-key-desc">Gitarre / Klavier</div>
          <div class="ob-key">M</div><div class="ob-key-desc">Metronom an/aus</div>
          <div class="ob-key">+ / −</div><div class="ob-key-desc">BPM erhöhen / senken</div>
          <div class="ob-key">C</div><div class="ob-key-desc">Griffbrett kalibrieren</div>
          <div class="ob-key">G</div><div class="ob-key-desc">Song-Modus öffnen</div>
          <div class="ob-key">A</div><div class="ob-key-desc">KI-Tipp anfordern</div>
          <div class="ob-key">P</div><div class="ob-key-desc">Nächster Akkord</div>
        </div>
      </div>

      <div style="background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px 16px">
        <div style="font-size:12px;font-weight:700;color:var(--text);margin-bottom:8px">🎙️ Mit NoteIQ sprechen</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.75">
          Drücke den <strong style="color:var(--text)">🎤 Mikrofon-Button</strong> für eine einzelne Frage.<br>
          Aktiviere <strong style="color:var(--text)">👂 Aktiv-Zuhören</strong> damit NoteIQ dauerhaft zuhört.<br>
          Sprich natürlich: <em>"Wie halte ich das Handgelenk?"</em>, <em>"Nächste Lektion"</em>, <em>"Stopp"</em>.
        </div>
      </div>

      <div style="background:rgba(255,107,0,.06);border:1px solid rgba(255,107,0,.2);border-radius:10px;padding:14px 16px">
        <div style="font-size:12px;font-weight:700;color:var(--accent3);margin-bottom:6px">🌹 Spanish/Latin Flair</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.75">
          Ab <strong style="color:var(--accent3)">Level 3</strong> erscheint der Flair-Tab automatisch.
          Dort findest du Flamenco, Bossa Nova und Latin-Rhythmus Lektionen —
          vollständig optional, können jederzeit mit Standard-Lektionen kombiniert werden.
        </div>
      </div>

    </div>
  </div>

  <!-- Navigation -->
  <div class="ob-nav">
    <button class="ob-btn ob-btn-secondary" id="ob-back-btn" onclick="obBack()" style="display:none">
      ← Zurück
    </button>
    <div style="display:flex;gap:12px;align-items:center;margin-left:auto">
      <span class="ob-step-counter" id="ob-step-counter">1 / 5</span>
      <button class="ob-btn ob-btn-primary" id="ob-next-btn" onclick="obNext()">
        Weiter →
      </button>
    </div>
  </div>

</div><!-- /ob-card -->
</div><!-- /onboarding-overlay -->

<!-- System-Status-Bar (unten, nach Onboarding) -->
<div class="sys-status-bar" id="sys-status-bar"></div>

<!-- Error-Toast Container -->
<div id="err-toast-container" style="position:fixed;top:72px;right:16px;z-index:600;display:flex;flex-direction:column;gap:8px;max-width:340px"></div>

<!-- Alter wov-Overlay (behalten für Audio-Unlock, aber versteckt) -->
<div id="wov" style="display:none" onclick="_doUnlockAndGreet();_closeOverlay()">
  <div class="wov-rings">
    <div class="wov-ring"></div><div class="wov-ring"></div><div class="wov-ring"></div>
  </div>
  <div class="wov-avatar">🎸</div>
  <h2>Hallo! Ich bin NoteIQ</h2>
  <p class="wov-sub-title">
    Deine persönliche <strong>KI-Musiklehrerin</strong><br>
    Klick zum Aktivieren des Tons
  </p>
  <button id="wov-btn" onclick="_doUnlockAndGreet();_closeOverlay()">
    ▶ &nbsp; Starten &amp; Ton einschalten
  </button>
  <div id="wov-hint">Chrome · Edge · Firefox · Safari</div>
</div>

<!-- ═══ HEADER ════════════════════════════════════════════════════════════ -->
<header>
  <div class="hdr-logo">
    <span class="hdr-logo-icon">🎸</span>
    <span class="hdr-logo-text">NoteIQ v16.0</span>
  </div>
  <span class="badge active" id="hdr-lesson">Lektion –</span>
  <span class="badge" id="hdr-chord">–</span>
  <span class="badge blue" id="hdr-level">Lv.–</span>
  <span class="badge" id="hdr-audio">🔇</span>
  <div class="hdr-spacer"></div>
  <span style="font-size:.68rem;color:var(--muted)" id="hdr-song-name"></span>
  <span class="badge" id="hdr-emotion">😊 Bereit</span>
  <span class="badge" id="hdr-mic-status" style="display:none;border-color:#FF335560;color:var(--red)">🎤 Hört…</span>
  <button onclick="showOnboarding()" title="Hilfe / Onboarding neu starten"
    style="background:transparent;border:1px solid var(--border2);color:var(--muted);
           font-size:12px;padding:4px 10px;border-radius:6px;cursor:pointer;
           transition:all .2s;margin-left:4px"
    onmouseover="this.style.borderColor='var(--accent3)';this.style.color='var(--text)'"
    onmouseout="this.style.borderColor='var(--border2)';this.style.color='var(--muted)'">
    ❓
  </button>
</header>

<!-- ═══ MAIN ══════════════════════════════════════════════════════════════ -->
<div class="main">

<!-- ═══ LEFT PANEL ════════════════════════════════════════════════════════ -->
<div class="panel">
  <div class="tabs">
    <div class="tab active" onclick="setLTab('lesson')">📚 Lektion</div>
    <div class="tab" onclick="setLTab('chords')">🎸 Akkorde</div>
    <div class="tab" onclick="setLTab('song')">♪ Songs</div>
    <div class="tab" id="tab-spanish" onclick="setLTab('spanish')" style="display:none">
      🌹 Flair
      <span class="tab-unlock-badge" id="spanish-tab-badge">NEU</span>
    </div>
  </div>
  <div class="panel-inner">

    <!-- LESSON TAB v11: Technique → Exercise → Song → XP -->
    <div id="ltab-lesson">

      <!-- XP + Level Anzeige -->
      <div style="padding:8px 0 4px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
          <div class="level-badge">
            <span>Level</span>
            <span class="lv-num" id="lv-num">1</span>
          </div>
          <div style="font-size:.65rem;color:var(--muted)">
            <span id="xp-earned-label" style="color:var(--yellow);font-weight:800"></span>
          </div>
        </div>
        <div class="xp-row">
          <span class="xp-label">XP</span>
          <div class="xp-track"><div class="xp-fill" id="xp-fill" style="width:0%"></div></div>
          <span class="xp-val" id="xp-val">0/120</span>
        </div>
      </div>

      <!-- 4-Phasen Stepper -->
      <div class="phase-stepper">
        <div class="phase-step" id="pstep-0">
          <span class="step-icon">🎯</span>Technik
        </div>
        <div class="phase-step" id="pstep-1">
          <span class="step-icon">💪</span>Üben
        </div>
        <div class="phase-step" id="pstep-2">
          <span class="step-icon">🎵</span>Song
        </div>
        <div class="phase-step" id="pstep-3">
          <span class="step-icon">🏆</span>XP
        </div>
      </div>

      <!-- Aktuelle Phase -->
      <div class="phase-card" id="phase-card">
        <div class="phase-label" id="phase-label">🎯 Technik lernen</div>
        <div class="phase-title" id="phase-title">–</div>
        <div class="phase-content" id="phase-content"></div>
        <ul class="phase-tips" id="phase-tips"></ul>
        <!-- Metadaten: BPM, Strumming, Akkorde -->
        <div class="phase-meta" id="phase-meta"></div>
        <!-- Akkord-Chips -->
        <div class="chord-chips" id="phase-chords"></div>
        <!-- Phasen-Fortschritt -->
        <div class="phase-prog-row">
          <span style="font-size:.6rem;color:var(--muted)">Phase</span>
          <div class="phase-prog-track">
            <div class="phase-prog-fill" id="phase-prog-fill" style="width:0%"></div>
          </div>
          <span style="font-size:.62rem;color:var(--muted);font-family:var(--mono)" id="phase-prog-val">0%</span>
        </div>
        <!-- Lektion gesamt -->
        <div class="bar-row" style="margin-top:4px">
          <span class="bar-label">Lektion</span>
          <div class="bar-track"><div class="bar-fill or" id="lesson-prog-bar" style="width:0%"></div></div>
          <span class="bar-val" id="lesson-prog-val">0%</span>
        </div>
        <!-- Weiter-Button -->
        <button class="btn-phase-next" id="btn-phase-next" onclick="advancePhase()" style="margin-top:9px">
          Weiter zur nächsten Phase →
        </button>
      </div>

      <!-- Curriculum Liste -->
      <div class="sect" style="margin-top:4px">
        <div class="sect-title">📚 Curriculum</div>
        <div id="lesson-list"></div>
      </div>

      <!-- Echtzeit-Analyse (kompakter) -->
      <div class="sect" id="cur-lesson-info">
        <div class="sect-title">📊 Echtzeit-Analyse</div>
        <div class="analysis-card">
          <div class="analysis-title">Akkord-Genauigkeit</div>
          <div class="analysis-score good" id="an-acc">–</div>
        </div>
        <div class="analysis-card">
          <div class="analysis-title">Haltung</div>
          <div class="analysis-score good" id="an-posture">–</div>
        </div>
        <div class="analysis-card">
          <div class="analysis-title">Rhythmus (BPM)</div>
          <div class="analysis-score ok" id="an-rhythm">–</div>
        </div>
        <div class="analysis-card">
          <div class="analysis-title">Sauberkeit</div>
          <div class="analysis-score good" id="an-clean">–</div>
          <div id="an-buzz" style="display:none;font-size:10px;font-weight:700;color:var(--red);margin-top:2px">⚡ Schnarren</div>
        </div>
        <div class="analysis-card">
          <div class="analysis-title">Obertöne</div>
          <div class="analysis-score ok" id="an-harm">–</div>
        </div>
        <div class="analysis-card">
          <div class="analysis-title">Finger-Status</div>
          <div class="analysis-score good" id="an-finger">OK</div>
        </div>
      </div>

    </div>

    <!-- Level-Up Overlay -->
    <div id="levelup-overlay">
      <div class="levelup-card">
        <div class="levelup-star">⭐</div>
        <div class="levelup-title">LEVEL UP!</div>
        <div class="levelup-sub" id="levelup-msg">Du hast Level 2 erreicht!</div>
        <button class="btn btn-primary" onclick="closeLevelUp()" style="margin-top:6px">
          Weiter →
        </button>
      </div>
    </div>

    <!-- CHORDS TAB -->
    <div id="ltab-chords" style="display:none">
      <div class="sect">
        <div class="chord-big">
          <div class="chord-big-name" id="lb-chord-name">–</div>
          <div class="chord-big-diff" id="lb-chord-diff">☆☆☆</div>
          <div class="chord-big-tip" id="lb-chord-tip"></div>
        </div>
        <div class="bar-row"><span class="bar-label">Genauigkeit</span>
          <div class="bar-track"><div class="bar-fill or" id="lb-acc-bar" style="width:0%"></div></div>
          <span class="bar-val" id="lb-acc-val">0%</span></div>
        <div class="bar-row"><span class="bar-label">Halten</span>
          <div class="bar-track"><div class="bar-fill ok" id="lb-hold-bar" style="width:0%"></div></div>
          <span class="bar-val" id="lb-hold-val">0%</span></div>
        <div style="margin:6px 0;font-size:.68rem;color:var(--muted)" id="lb-mastered"></div>
      </div>
      <div class="sect">
        <div class="sect-title">📚 Akkord-Bibliothek</div>
        <input id="lib-search" placeholder="Akkord suchen…">
        <div id="lib-content"></div>
      </div>
    </div>

    <!-- SONG TAB -->
    <div id="ltab-song" style="display:none">
      <div class="sect">
        <div class="sect-title">♪ Song auswählen</div>
        <div id="song-list-left"></div>
      </div>
      <div id="song-active-info" style="display:none">
        <div class="sect-title">▶ Aktueller Song</div>
        <div style="font-weight:800;color:var(--accent3);margin-bottom:8px;font-size:.85rem" id="sal-name"></div>
        <div class="bar-row"><span class="bar-label">Fortschritt</span>
          <div class="bar-track"><div class="bar-fill info" id="sal-prog" style="width:0%"></div></div>
          <span class="bar-val" id="sal-prog-val">0%</span></div>
        <div style="margin:6px 0;font-size:.72rem">
          Aktuell: <span style="color:var(--accent3);font-weight:700" id="sal-cur">–</span>
          → <span style="color:var(--muted)" id="sal-next">–</span>
        </div>
        <div class="bar-row"><span class="bar-label">Score</span>
          <div class="bar-track"><div class="bar-fill ok" id="sal-score" style="width:0%"></div></div>
          <span class="bar-val" id="sal-score-val">0%</span></div>
        <button class="btn btn-danger btn-wide" style="margin-top:8px" onclick="stopSong()">
          ■ Song beenden
        </button>
      </div>
    </div>

    <!-- SPANISH / LATIN FLAIR TAB -->
    <div id="ltab-spanish" style="display:none">
      <div style="padding:8px 0 6px">
        <div style="font-family:var(--mono);font-size:.6rem;letter-spacing:.14em;
                    text-transform:uppercase;color:var(--muted);margin-bottom:4px">
          🌹 Spanish / Latin Flair
        </div>
        <div style="font-size:.72rem;color:var(--text);line-height:1.5;margin-bottom:8px">
          Optionale Spezial-Lektionen – freigeschaltet ab
          <strong style="color:var(--yellow)">Level 3</strong>.
        </div>
        <div id="spanish-unlock-notice" style="display:none">
          <div class="unlock-notice">
            🔒 <span id="spanish-unlock-msg">Erreiche Level 3 um diese Kategorie freizuschalten!</span>
          </div>
        </div>
      </div>
      <div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">
        <button class="flair-filter-btn active" data-filter="all" onclick="filterFlair('all')">Alle</button>
        <button class="flair-filter-btn" data-filter="Spanish" onclick="filterFlair('Spanish')">💃 Flamenco</button>
        <button class="flair-filter-btn" data-filter="Latin" onclick="filterFlair('Latin')">🌴 Latin</button>
      </div>
      <div id="flair-lesson-list"></div>
      <div style="margin-top:10px;padding:9px;background:var(--panel2);
                  border:1px solid var(--border);border-radius:var(--radius);
                  font-size:.62rem;color:var(--muted);line-height:1.6">
        💡 <em>Optional</em> – erweitert deinen Stil,
        zählt nicht zum Haupt-Curriculum.
      </div>
    </div>

  </div><!-- /panel-inner -->

  <!-- CHAT -->
  <div id="chat-box">
    <div class="chat-history" id="chat-history"></div>
    <div class="chat-input-row">
      <input id="chat-input" placeholder="Frag NoteIQ…" onkeydown="if(event.key==='Enter')sendChat()">
      <button id="chat-send" onclick="sendChat()">→</button>
    </div>
  </div>
</div>

<!-- ═══ CENTER: 3D TEACHER ════════════════════════════════════════════════ -->
<div id="center">
  <canvas id="teacher-canvas"></canvas>
  <div id="emotion-tag">😊 Bereit</div>
  <div id="speech-bubble"></div>
  <button id="mic-button" title="NoteIQ fragen" onclick="toggleMic()">🎤</button>
  <button id="active-listen-btn" onclick="toggleActiveListen()" title="Aktives Zuhören – NoteIQ hört dauerhaft zu" style="background:transparent;border:1px solid rgba(255,255,255,0.15);color:#8890A0;font-size:11px;padding:6px 14px;border-radius:8px;cursor:pointer;transition:all .3s;margin-left:4px">👂 Aktiv</button>
  <div id="mic-interim-text"></div>
  <div id="feedback-area"></div>

  <!-- Song Bar -->
  <div id="song-bar">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <span style="font-size:.76rem;color:var(--blue);font-weight:800" id="sb-name">♪</span>
      <span id="sb-alarm" style="display:none;background:var(--red);color:#fff;border-radius:8px;padding:2px 10px;font-size:.68rem;animation:pulse .4s infinite">WECHSEL!</span>
      <span style="font-size:.7rem;color:var(--muted);font-family:var(--mono)" id="sb-score">Score: 0%</span>
    </div>
    <div style="background:#ffffff18;border-radius:4px;height:6px;position:relative">
      <div id="sb-prog" style="height:100%;background:var(--blue);border-radius:4px;transition:width .2s"></div>
      <div id="sb-barpos" style="position:absolute;top:0;height:100%;background:#ffffff50;border-radius:4px;transition:width .05s"></div>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:.7rem;font-family:var(--mono)">
      <span style="color:var(--text);font-weight:700" id="sb-cur">–</span>
      <span style="color:var(--muted)" id="sb-next">→ –</span>
    </div>
  </div>

  <!-- Camera Strip -->
  <div id="cam-strip">
    <img id="cam-img" src="/stream" alt="Kamera">
    <div style="flex:1;display:flex;flex-direction:column;gap:5px;padding-bottom:4px">
      <div style="display:flex;gap:8px;align-items:center">
        <span style="font-size:.65rem;color:var(--muted)">Saiten:</span>
        <div class="strings-row" id="cam-strings" style="gap:3px"></div>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <div class="strum-big" id="strum-indicator">–</div>
        <div style="flex:1">
          <div style="font-size:.62rem;color:var(--muted)">Gespielt / Ziel</div>
          <div style="font-size:.72rem;font-family:var(--mono)" id="strum-live">–</div>
          <div style="font-size:.62rem;font-family:var(--mono);color:var(--muted)" id="strum-tgt">–</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.62rem;color:var(--muted)">BPM</div>
          <div style="font-size:.95rem;font-weight:800;color:var(--blue);font-family:var(--mono)" id="bpm-live">–</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Session Feedback Overlay -->
  <div class="session-fb" id="session-fb">
    <h2>🎓 NoteIQ Bewertung</h2>
    <div class="fb-grade" id="sfb-grade">–</div>
    <div id="sfb-content"></div>
    <button class="btn btn-primary btn-wide" style="margin-top:10px" onclick="closeFeedback()">
      Weiter lernen →
    </button>
  </div>
</div>

<!-- ═══ AVATAR PANEL: vollständiger 3D CSS-Avatar Luna ═══════════════════ -->
<div id="avatar-panel">
  <canvas id="avatar-canvas"></canvas>
  <div id="av-infobar">
    <div class="av-name">NoteIQ · KI-Avatar</div>
    <div class="av-status" id="av-status">Bereit zum Unterrichten</div>
    <div class="av-btns">
      <div class="av-btn active" id="avb-neutral"    onclick="avSetMood('neutral')">😊 Neutral</div>
      <div class="av-btn"        id="avb-happy"      onclick="avSetMood('happy')">🌟 Froh</div>
      <div class="av-btn"        id="avb-explaining" onclick="avSetMood('explaining')">👩‍🏫 Erklärt</div>
      <div class="av-btn"        id="avb-correcting" onclick="avSetMood('correcting')">⚠️ Korrigiert</div>
    </div>
  </div>
</div>


<!-- ═══ RIGHT PANEL ═══════════════════════════════════════════════════════ -->
<div class="panel right">
  <div class="tabs">
    <div class="tab active" onclick="setRTab('audio')">🎵 Audio</div>
    <div class="tab" onclick="setRTab('poly')">🎸 Poly</div>
    <div class="tab" onclick="setRTab('stats')">📊 Stats</div>
    <div class="tab" onclick="setRTab('metro')">🥁 Metro</div>
  </div>
  <div class="panel-inner">

    <!-- AUDIO TAB -->
    <div id="rtab-audio">
      <div class="pitch-box">
        <div class="pitch-note" id="rp-note">–</div>
        <div class="pitch-hz" id="rp-hz">0 Hz</div>
        <div class="pitch-meter"><div class="pitch-needle" id="rp-needle" style="left:50%"></div></div>
      </div>
      <div class="spectrum" id="rp-spectrum"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin:6px 0">
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;height:22px;overflow:hidden;position:relative">
          <div id="rb-bass" style="height:100%;background:linear-gradient(90deg,#802000,#D45200);border-radius:5px;width:0%;transition:width .1s"></div>
          <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.58rem;color:#ffffff90;font-weight:700">BASS</div>
        </div>
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;height:22px;overflow:hidden;position:relative">
          <div id="rb-mid" style="height:100%;background:linear-gradient(90deg,#904000,#FF6B00);border-radius:5px;width:0%;transition:width .1s"></div>
          <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.58rem;color:#ffffff90;font-weight:700">MID</div>
        </div>
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;height:22px;overflow:hidden;position:relative">
          <div id="rb-high" style="height:100%;background:linear-gradient(90deg,#006070,#00C8FF);border-radius:5px;width:0%;transition:width .1s"></div>
          <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.58rem;color:#ffffff90;font-weight:700">HIGH</div>
        </div>
      </div>
      <div style="margin:5px 0;font-size:.7rem">
        <span style="color:var(--muted)">Noten: </span>
        <span id="rp-notes" style="color:var(--accent3);font-weight:700;font-family:var(--mono)"></span>
        <span id="rp-match" style="margin-left:5px;background:#FF6B0025;border-radius:6px;padding:1px 7px;font-size:.62rem;color:var(--accent)"></span>
      </div>
      <div id="rp-onset" style="display:none;background:#0E3020;border-radius:6px;padding:3px 10px;font-size:.68rem;color:var(--green);margin:3px 0;border:1px solid #2EFF9A30">
  <span id="rp-harm" style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:8px"></span>
  <span id="an-buzz" style="font-size:11px;font-weight:700;margin-left:8px;display:none"></span>● ANSCHLAG</div>
    </div>

    <!-- POLY TAB -->
    <div id="rtab-poly" style="display:none">
      <div class="sect-title">🎸 Saiten-Analyse (6-stimmig)</div>
      <div class="strings-row" id="poly-strings-r" style="justify-content:space-between"></div>
      <div class="bar-row" style="margin-top:10px"><span class="bar-label">Akkord-Score</span>
        <div class="bar-track"><div class="bar-fill ok" id="poly-score-bar-r" style="width:0%"></div></div>
        <span class="bar-val" id="poly-score-val-r">0%</span></div>
      <div style="margin:8px 0;font-size:.7rem">
        <div style="margin-bottom:3px"><span style="color:var(--muted)">Erkannte Noten: </span><span id="poly-notes-r" style="color:var(--text);font-weight:700;font-family:var(--mono)"></span></div>
        <div><span style="color:var(--muted)">Gedämpft: </span><span id="poly-muted-r" style="color:var(--red);font-family:var(--mono)"></span></div>
      </div>
      <div class="analysis-card" style="flex-direction:column;align-items:flex-start;gap:4px;margin-top:8px">
        <div class="analysis-title">Interpretation</div>
        <div id="poly-interp" style="font-size:.7rem;color:var(--text);line-height:1.6"></div>
      </div>
    </div>

    <!-- STATS TAB -->
    <div id="rtab-stats" style="display:none">
      <div class="sect-title">📊 Sitzungs-Statistiken</div>
      <div id="stats-rows"></div>
      <div style="margin-top:12px;display:flex;flex-direction:column;gap:6px">
        <button class="btn btn-primary btn-wide" onclick="requestFeedback()">
          🎓 NoteIQ Bewertung anfordern
        </button>
        <button class="btn btn-secondary btn-wide" onclick="startNewLesson()">
          📚 Lektion neu starten
        </button>
      </div>
    </div>

    <!-- METRO TAB -->
    <div id="rtab-metro" style="display:none">
      <div class="sect-title">🥁 Metronom</div>
      <div style="text-align:center;padding:16px 0">
        <div id="metro-dot" id="metro-bpm-display">80</div>
      </div>
      <div class="bpm-grid">
        <div class="bpm-btn" onclick="changeBPM(-10)">−10</div>
        <div class="bpm-btn" onclick="changeBPM(-5)">−5</div>
        <div class="bpm-btn" onclick="changeBPM(5)">+5</div>
        <div class="bpm-btn" onclick="changeBPM(10)">+10</div>
      </div>
      <button class="btn btn-primary btn-wide" id="metro-btn" onclick="toggleMetro()">
        ⏸ Pause
      </button>
    </div>

  </div><!-- /panel-inner -->
</div><!-- /right panel -->
</div><!-- /main -->


<script>
// ════════════════════════════════════════════════════════════════════════════
//  3D KI-ASSISTENT NoteIQ – Three.js prozedural
// ════════════════════════════════════════════════════════════════════════════
(function() {
const canvas = document.getElementById('teacher-canvas');
const container = document.getElementById('center');

// ── Renderer ──────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.setClearColor(0x060612, 1);

// ── Scene ─────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x060612, 8, 20);

// ── Camera ────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 1.5, 4.5);
camera.lookAt(0, 1.2, 0);

// ── Lights ────────────────────────────────────────────────────────────────
const ambientLight = new THREE.AmbientLight(0x202040, 0.8);
scene.add(ambientLight);
const keyLight = new THREE.PointLight(0xFF9240, 2.5, 12);
keyLight.position.set(-2, 4, 3);
keyLight.castShadow = true;
scene.add(keyLight);
const fillLight = new THREE.PointLight(0x4080FF, 1.2, 10);
fillLight.position.set(3, 2, 2);
scene.add(fillLight);
const rimLight = new THREE.PointLight(0xFF6B00, 1.8, 8);
rimLight.position.set(0, 3, -3);
scene.add(rimLight);
const topLight = new THREE.DirectionalLight(0xFFE0C0, 0.6);
topLight.position.set(0, 5, 2);
scene.add(topLight);

// ── Floor ─────────────────────────────────────────────────────────────────
const floorGeo = new THREE.PlaneGeometry(12, 12);
const floorMat = new THREE.MeshLambertMaterial({color:0x080816});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI/2;
floor.receiveShadow = true;
scene.add(floor);

// ── Glow particles ────────────────────────────────────────────────────────
const particleGeo = new THREE.BufferGeometry();
const pCount = 200;
const pPos = new Float32Array(pCount*3);
for(let i=0;i<pCount;i++){
  pPos[i*3]   = (Math.random()-0.5)*8;
  pPos[i*3+1] = Math.random()*6;
  pPos[i*3+2] = (Math.random()-0.5)*8-2;
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(pPos,3));
const particleMat = new THREE.PointsMaterial({color:0xFF6B00, size:0.04, transparent:true, opacity:0.4});
const particles = new THREE.Points(particleGeo, particleMat);
scene.add(particles);

// ════════════════════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════════════
//  NOTEIQ CHARAKTER-AUFBAU – natürlichere Geometrie + Mund-Morphing
// ════════════════════════════════════════════════════════════════════════════
const LUNA = new THREE.Group();
LUNA.position.set(0, 0, 0);
scene.add(LUNA);

// ── MATERIALIEN ──────────────────────────────────────────────────────────────
const skinMat   = new THREE.MeshPhongMaterial({color:0xF2C09A, shininess:22});
const skin2Mat  = new THREE.MeshPhongMaterial({color:0xE6AC88, shininess:16});
const hairMat   = new THREE.MeshPhongMaterial({color:0x250D04, shininess:55});
const eyeWMat   = new THREE.MeshPhongMaterial({color:0xF8F4F0});
const irisMat   = new THREE.MeshPhongMaterial({color:0x623012, emissive:0x1A0600});
const pupilMat  = new THREE.MeshPhongMaterial({color:0x040404});
const lipMat    = new THREE.MeshPhongMaterial({color:0xC03850, shininess:75});
const clothMat  = new THREE.MeshPhongMaterial({color:0x160735, shininess:6});
const clothOrMat= new THREE.MeshPhongMaterial({color:0xFF6B00, shininess:20});
const goldMat   = new THREE.MeshPhongMaterial({color:0xFFCC20, shininess:130});
const toothMat  = new THREE.MeshPhongMaterial({color:0xFFF9F2, shininess:95});
const tongueMat = new THREE.MeshPhongMaterial({color:0xD85868});
const mouthIMat = new THREE.MeshPhongMaterial({color:0x380A18});
const shoeMat   = new THREE.MeshPhongMaterial({color:0x060614, shininess:90});

// ── TORSO ─────────────────────────────────────────────────────────────────────
const torso = new THREE.Group();
torso.position.y = 0.82;
LUNA.add(torso);

const bodyMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.27,0.21,0.9,16), clothMat);
bodyMesh.castShadow = true; torso.add(bodyMesh);

const collarM = new THREE.Mesh(new THREE.TorusGeometry(0.155,0.040,9,20), clothOrMat);
collarM.rotation.x = Math.PI/2; collarM.position.y = 0.46; torso.add(collarM);

const neckM = new THREE.Mesh(new THREE.CylinderGeometry(0.087,0.105,0.24,10), skinMat);
neckM.position.y = 0.56; torso.add(neckM);

// ── KOPF ──────────────────────────────────────────────────────────────────────
const headGrp = new THREE.Group();
headGrp.position.y = 0.79; torso.add(headGrp);

const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.265,24,20), skinMat);
headMesh.scale.set(1,1.12,0.93); headMesh.castShadow = true; headGrp.add(headMesh);

const chinMesh = new THREE.Mesh(new THREE.SphereGeometry(0.15,14,12), skinMat);
chinMesh.scale.set(0.98,0.52,0.88); chinMesh.position.set(0,-0.22,0.06); headGrp.add(chinMesh);

// Wangenknochen
[-1,1].forEach(s=>{
  const ck = new THREE.Mesh(new THREE.SphereGeometry(0.09,8,8), skin2Mat);
  ck.scale.set(0.7,0.5,0.6); ck.position.set(s*0.20, 0.01, 0.18); headGrp.add(ck);
});

// ── AUGEN ─────────────────────────────────────────────────────────────────────
function makeEye(sx) {
  const g = new THREE.Group();
  g.position.set(sx*0.105, 0.062, 0.228);

  const sock  = new THREE.Mesh(new THREE.SphereGeometry(0.065,12,10), skin2Mat);
  sock.scale.set(1,0.80,0.46); g.add(sock);

  const white = new THREE.Mesh(new THREE.SphereGeometry(0.052,14,12), eyeWMat);
  white.position.z = 0.004; g.add(white);

  const iris  = new THREE.Mesh(new THREE.CircleGeometry(0.032,16), irisMat);
  iris.position.z = 0.052; g.add(iris);

  const pupil = new THREE.Mesh(new THREE.CircleGeometry(0.018,14), pupilMat);
  pupil.position.z = 0.054; g.add(pupil);

  // Glanzpunkt
  const hl = new THREE.Mesh(new THREE.CircleGeometry(0.0065,8), eyeWMat);
  hl.position.set(0.009,0.009,0.055); g.add(hl);

  // Lid (für Blinzeln)
  const lid = new THREE.Mesh(
    new THREE.SphereGeometry(0.056,12,10,0,Math.PI*2,0,Math.PI*0.52), skinMat);
  lid.rotation.x = Math.PI; lid.position.z = 0.010;
  lid.name = 'lid'; lid.scale.y = 0.04; g.add(lid);

  // Wimpern
  const lc = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-0.056,0.018,0.053),
    new THREE.Vector3(0,0.060,0.053),
    new THREE.Vector3(0.056,0.018,0.053));
  const lash = new THREE.Mesh(new THREE.TubeGeometry(lc,10,0.0072,5,false), hairMat);
  g.add(lash);

  headGrp.add(g); return g;
}
const eyeLeft  = makeEye(-1);
const eyeRight = makeEye(1);

// ── AUGENBRAUEN ───────────────────────────────────────────────────────────────
function makeBrow(sx) {
  const c = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(sx*0.038, 0.112, 0.218),
    new THREE.Vector3(sx*0.105, 0.140, 0.220),
    new THREE.Vector3(sx*0.172, 0.120, 0.212));
  const m = new THREE.Mesh(new THREE.TubeGeometry(c,10,0.0082,5,false), hairMat);
  headGrp.add(m); return m;
}
const browLeft  = makeBrow(-1);
const browRight = makeBrow(1);

// ── NASE ──────────────────────────────────────────────────────────────────────
const noseGrp = new THREE.Group();
noseGrp.position.set(0,-0.018,0.258); headGrp.add(noseGrp);
const noseBridge = new THREE.Mesh(new THREE.CylinderGeometry(0.015,0.022,0.112,8), skinMat);
noseBridge.rotation.x = Math.PI*0.07; noseBridge.position.y = 0.042; noseGrp.add(noseBridge);
const noseTip = new THREE.Mesh(new THREE.SphereGeometry(0.029,10,9), skinMat);
noseTip.scale.set(1.08,0.80,1.0); noseGrp.add(noseTip);
[-1,1].forEach(s=>{
  const ns = new THREE.Mesh(new THREE.SphereGeometry(0.016,8,7), skin2Mat);
  ns.scale.set(0.9,0.7,0.8); ns.position.set(s*0.024,-0.008,0); noseGrp.add(ns);
});

// ── MUND (vollständig morphbar) ───────────────────────────────────────────────
const mouthGrp = new THREE.Group();
mouthGrp.position.set(0,-0.100,0.255); headGrp.add(mouthGrp);

const upLip = new THREE.Mesh(
  new THREE.SphereGeometry(0.060,16,10,0,Math.PI*2,0,Math.PI*0.52), lipMat);
upLip.rotation.x = Math.PI; upLip.scale.set(1,0.44,0.56);
upLip.name = 'upLip'; mouthGrp.add(upLip);

const loLip = new THREE.Mesh(
  new THREE.SphereGeometry(0.065,16,10,0,Math.PI*2,0,Math.PI*0.52), lipMat);
loLip.scale.set(1.03,0.50,0.56);
loLip.position.y = -0.013; loLip.name = 'loLip'; mouthGrp.add(loLip);

const mInt = new THREE.Mesh(new THREE.SphereGeometry(0.046,12,10), mouthIMat);
mInt.scale.set(1,0.22,0.72); mInt.position.z = -0.004; mInt.name = 'mInt'; mouthGrp.add(mInt);

const teethMesh = new THREE.Mesh(new THREE.BoxGeometry(0.090,0.022,0.016), toothMat);
teethMesh.position.z = 0.010; teethMesh.name = 'teeth'; mouthGrp.add(teethMesh);

const tongueMesh = new THREE.Mesh(new THREE.SphereGeometry(0.026,10,8), tongueMat);
tongueMesh.scale.set(1.1,0.46,0.85); tongueMesh.position.z = -0.006; tongueMesh.name = 'tongue'; mouthGrp.add(tongueMesh);

// Grübchen
function makeDimple(sx) {
  const d = new THREE.Mesh(new THREE.SphereGeometry(0.010,7,6), skin2Mat);
  d.scale.set(1,0.42,0.42); d.position.set(sx*0.080,-0.030,0.240); headGrp.add(d);
}
makeDimple(-1); makeDimple(1);

// ── OHREN ─────────────────────────────────────────────────────────────────────
function makeEar(sx) {
  const ear = new THREE.Mesh(new THREE.SphereGeometry(0.055,10,9), skin2Mat);
  ear.scale.set(0.46,0.80,0.46); ear.position.set(sx*0.268,0.022,0); headGrp.add(ear);
  const er = new THREE.Mesh(new THREE.TorusGeometry(0.021,0.006,7,16), goldMat);
  er.position.set(sx*0.268,-0.050,0); er.rotation.y = Math.PI/2; headGrp.add(er);
}
makeEar(-1); makeEar(1);

// ── HAARE ─────────────────────────────────────────────────────────────────────
const hairGrp = new THREE.Group(); headGrp.add(hairGrp);
const hcap = new THREE.Mesh(
  new THREE.SphereGeometry(0.284,24,20,0,Math.PI*2,0,Math.PI*0.54), hairMat);
hcap.position.y = 0.046; hairGrp.add(hcap);
[-1,1].forEach(s=>{
  const sh = new THREE.Mesh(new THREE.SphereGeometry(0.170,14,11), hairMat);
  sh.scale.set(0.44,1.30,0.66); sh.position.set(s*0.248,-0.038,0); hairGrp.add(sh);
});
const fr = new THREE.Mesh(
  new THREE.CylinderGeometry(0.248,0.270,0.118,20,1,false,Math.PI*0.1,Math.PI*0.8), hairMat);
fr.position.set(0,0.130,0.110); fr.rotation.x = 0.26; hairGrp.add(fr);
const longH = new THREE.Mesh(new THREE.CylinderGeometry(0.090,0.165,0.75,12), hairMat);
longH.position.set(0,-0.50,-0.108); longH.rotation.x = 0.08; hairGrp.add(longH);

// ── ARME ──────────────────────────────────────────────────────────────────────
function makeArm(sx, parent) {
  const ag = new THREE.Group();
  ag.position.set(sx*0.33, 0.30, 0);

  const sh  = new THREE.Mesh(new THREE.SphereGeometry(0.094,10,9), clothMat); ag.add(sh);
  const ua  = new THREE.Mesh(new THREE.CylinderGeometry(0.060,0.052,0.40,10), clothMat);
  ua.position.y = -0.22; ag.add(ua);
  const elb = new THREE.Mesh(new THREE.SphereGeometry(0.056,9,8), skinMat);
  elb.position.y = -0.43; ag.add(elb);
  const la  = new THREE.Mesh(new THREE.CylinderGeometry(0.047,0.042,0.36,9), skinMat);
  la.position.y = -0.62; ag.add(la);
  const hand= new THREE.Mesh(new THREE.SphereGeometry(0.066,10,9), skinMat);
  hand.scale.set(1.06,0.70,0.76); hand.position.y = -0.81; ag.add(hand);
  for(let f=0;f<4;f++){
    const fg = new THREE.Mesh(new THREE.CylinderGeometry(0.010,0.008,0.070,6), skinMat);
    fg.position.set((f-1.5)*0.025,-0.868,0.020); ag.add(fg);
  }
  const th = new THREE.Mesh(new THREE.CylinderGeometry(0.012,0.010,0.060,6), skinMat);
  th.rotation.z = sx*Math.PI*0.22; th.position.set(sx*0.060,-0.845,0); ag.add(th);

  parent.add(ag); return ag;
}
const armLeft  = makeArm(-1, torso);
const armRight = makeArm( 1, torso);
armLeft.rotation.set(0.12,0,0.20);
armRight.rotation.set(0.12,0,-0.20);

// ── ROCK & BEINE ──────────────────────────────────────────────────────────────
const skirtM = new THREE.Mesh(new THREE.CylinderGeometry(0.30,0.42,0.60,18), clothMat);
skirtM.position.y = 0.50; LUNA.add(skirtM);
[-1,1].forEach(s=>{
  const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.060,0.052,0.74,10), clothMat);
  leg.position.set(s*0.092,0.13,0); LUNA.add(leg);
  const shoe= new THREE.Mesh(new THREE.SphereGeometry(0.074,10,8), shoeMat);
  shoe.scale.set(0.87,0.50,1.30); shoe.position.set(s*0.092,-0.22,0.04); LUNA.add(shoe);
});

// ── GITARREN-PROP ─────────────────────────────────────────────────────────────
const guitarGrp = new THREE.Group(); guitarGrp.visible = false; LUNA.add(guitarGrp);
const gtBM = new THREE.MeshPhongMaterial({color:0x7B3300, shininess:85});
const gBody  = new THREE.Mesh(new THREE.CylinderGeometry(0.20,0.24,0.082,18), gtBM);
gBody.rotation.x = Math.PI/2; gBody.position.set(0.27,0.08,0.22); guitarGrp.add(gBody);
const gWaist = new THREE.Mesh(new THREE.CylinderGeometry(0.13,0.13,0.082,16), gtBM);
gWaist.rotation.x = Math.PI/2; gWaist.position.set(0.27,0.30,0.22); guitarGrp.add(gWaist);
const gNeck  = new THREE.Mesh(new THREE.BoxGeometry(0.054,0.58,0.036),
  new THREE.MeshPhongMaterial({color:0x4A2200}));
gNeck.position.set(0.27,0.64,0.22); guitarGrp.add(gNeck);
for(let gi=0;gi<6;gi++){
  const gs = new THREE.Mesh(new THREE.CylinderGeometry(0.0018,0.0018,0.58,3),
    new THREE.MeshPhongMaterial({color:0xBBBBBB, shininess:110}));
  gs.position.set(0.27-0.011+gi*0.0045,0.64,0.240); guitarGrp.add(gs);
}

// ── KLAVIER-PROP ──────────────────────────────────────────────────────────────
const pianoGrp = new THREE.Group(); pianoGrp.visible = false; LUNA.add(pianoGrp);
const piBody = new THREE.Mesh(new THREE.BoxGeometry(0.64,0.064,0.20),
  new THREE.MeshPhongMaterial({color:0x060606, shininess:120}));
piBody.position.set(0,0.28,0.33); pianoGrp.add(piBody);
for(let ki=0;ki<8;ki++){
  const wk = new THREE.Mesh(new THREE.BoxGeometry(0.063,0.042,0.155),
    new THREE.MeshPhongMaterial({color:0xFFF8F2, shininess:70}));
  wk.position.set(-0.268+ki*0.077,0.334,0.34); pianoGrp.add(wk);
}
[0,1,3,4,5].forEach(pos=>{
  const bk = new THREE.Mesh(new THREE.BoxGeometry(0.038,0.052,0.096),
    new THREE.MeshPhongMaterial({color:0x060606, shininess:95}));
  bk.position.set(-0.230+pos*0.077,0.376,0.296); pianoGrp.add(bk);
});

// ── HINTERGRUND VISUALIZER BARS ───────────────────────────────────────────────
const bgBars = [];
for(let i=0; i<24; i++){
  const ang = (i/24)*Math.PI*2, r=3.8;
  const geo = new THREE.BoxGeometry(0.12, 0.5, 0.12);
  const mat = new THREE.MeshPhongMaterial({color:0xFF6B00, emissive:0xFF3000, emissiveIntensity:0.3, transparent:true, opacity:0.5});
  const bar = new THREE.Mesh(geo, mat);
  bar.position.set(Math.cos(ang)*r, 0.25, Math.sin(ang)*r-1);
  scene.add(bar);
  bgBars.push(bar);
}


// ════════════════════════════════════════════════════════════════════════════
//  ANIMATIONS-SYSTEM v7 – natürlichere Bewegung, Sakkaden, Phoneme
// ════════════════════════════════════════════════════════════════════════════
let lunaState = {
  emotion:    'neutral',
  speaking:   false,
  mouthOpen:  0,
  mouthTarget:0,
  armPose:    'rest',
  specData:   new Array(24).fill(0),
  // Blinzeln
  blinkTimer: 2.8 + Math.random()*3.2,
  blinkState: 0,  // 0=offen 1=schließen 2=öffnen
  blinkProg:  0,
  doubleBlink:false,
  // Augen-Sakkaden (natürlicher Blick)
  saccadeTimer:0,
  saccadeX:0, saccadeY:0,
  lookX:0,    lookY:0,
  // Mund-Phoneme
  mPhase:0, mWaveA:0, mWaveB:0,
  // Clap / Wave flags
  isClapping:false, isWaving:false,
};

// ── Emotion → Augenbrauen ────────────────────────────────────────────────────
const BROW_POSES = {
  neutral:    [0,    0,      0,     0   ],
  happy:      [-0.11, 0.005, -0.11, 0.005],
  thinking:   [ 0.20,-0.006, -0.055,0.009],
  correcting: [ 0.24,-0.006,  0.24,-0.006],  // beide runter → Stirnrunzeln
  explaining: [-0.065,0.008,  0.065,0.008],
  proud:      [-0.16, 0.012,  0.16, 0.012],
};
const EMOJ_MAP = {
  neutral:'😊 Bereit', happy:'😄 Super!', thinking:'🤔 Hmm…',
  correcting:'⚠️ Korrigiere', explaining:'👩‍🏫 Erkläre', proud:'🌟 Toll!'
};

function applyEmotion(e) {
  lunaState.emotion = e;
  const bp = BROW_POSES[e] || BROW_POSES.neutral;
  browLeft.rotation.z  = bp[0]; browLeft.position.y  = bp[1];
  browRight.rotation.z = bp[2]; browRight.position.y = bp[3];
  const etag = document.getElementById('emotion-tag');
  if(etag) etag.textContent = EMOJ_MAP[e] || '😊';
}

// ── Arm-Pose ─────────────────────────────────────────────────────────────────
function setLunaPose(pose) {
  lunaState.armPose    = pose;
  lunaState.isClapping = false;
  lunaState.isWaving   = false;
  guitarGrp.visible = false;
  pianoGrp.visible  = false;
  if(pose==='guitar'){
    armLeft.rotation.set(-0.44,0, 0.07); armRight.rotation.set( 0.60,0,-0.50);
    guitarGrp.visible = true;
  } else if(pose==='piano'){
    armLeft.rotation.set(-0.64,0,-0.04); armRight.rotation.set(-0.64,0, 0.04);
    pianoGrp.visible = true;
  } else if(pose==='pointing_right'){
    armLeft.rotation.set( 0.12,0, 0.20); armRight.rotation.set(-0.35,0,-0.80);
  } else if(pose==='both_up'){
    armLeft.rotation.set(-0.50,0,-0.42); armRight.rotation.set(-0.50,0, 0.42);
  } else if(pose==='clap'){
    lunaState.isClapping = true;
  } else if(pose==='wave'){
    lunaState.isWaving = true;
  } else {
    armLeft.rotation.set( 0.12,0, 0.20); armRight.rotation.set( 0.12,0,-0.20);
  }
}


// Alias for backward compatibility
function setArmPose(pose) { setLunaPose(pose); }
// ── Mund-Morphing (Phonem-Variation) ─────────────────────────────────────────
function updateMouth(dt) {
  lunaState.mPhase  += dt * (lunaState.speaking ? 9.5 : 2.5);
  lunaState.mWaveA  += dt * 5.3;
  lunaState.mWaveB  += dt * 7.7;
  // Smooth target
  lunaState.mouthOpen += (lunaState.mouthTarget - lunaState.mouthOpen) * Math.min(1, dt*16);
  // Phonem-artige Variation: 3 überlagerte Sinuswellen
  const variation = lunaState.speaking
    ? (0.62 + Math.sin(lunaState.mPhase)*0.24 + Math.sin(lunaState.mWaveA)*0.10 + Math.sin(lunaState.mWaveB)*0.08)
    : 1.0;
  const mo = lunaState.mouthOpen * variation;

  const ul = mouthGrp.getObjectByName('upLip');
  const ll = mouthGrp.getObjectByName('loLip');
  const mi = mouthGrp.getObjectByName('mInt');
  const tg = mouthGrp.getObjectByName('tongue');
  if(ul) ul.position.y =  mo * 0.024;
  if(ll) ll.position.y = -0.013 - mo * 0.030;
  if(mi) mi.scale.y    =  0.22  + mo * 1.0;
  if(tg) tg.position.y = -0.012 + mo * 0.010;
}

// ── Natürliches Blinzeln (inkl. Doppel-Blinzeln) ────────────────────────────
function updateBlink(dt) {
  lunaState.blinkTimer -= dt;
  if(lunaState.blinkTimer<=0 && lunaState.blinkState===0){
    lunaState.blinkState = 1; lunaState.blinkProg = 0;
    lunaState.doubleBlink = Math.random()<0.12;
    lunaState.blinkTimer  = 2.8 + Math.random()*4.0;
  }
  if(lunaState.blinkState!==0){
    lunaState.blinkProg += dt*11;
    const p  = Math.min(1, lunaState.blinkProg);
    const ly = (lunaState.blinkState===1) ? p : 1-p;
    const lL = eyeLeft.getObjectByName('lid');
    const lR = eyeRight.getObjectByName('lid');
    if(lL) lL.scale.y = ly;
    if(lR) lR.scale.y = ly;
    if(lunaState.blinkProg>=1){
      if(lunaState.blinkState===1){ lunaState.blinkState=2; lunaState.blinkProg=0; }
      else {
        lunaState.blinkState=0;
        if(lunaState.doubleBlink){ lunaState.doubleBlink=false; lunaState.blinkTimer=0.08; }
      }
    }
  }
}

// ── Sakkaden-Augenbewegung (natürlicher, lebendiger Blick) ───────────────────
function updateEyeLook(dt, ts) {
  lunaState.saccadeTimer -= dt;
  if(lunaState.saccadeTimer<=0){
    lunaState.saccadeTimer = 1.2 + Math.random()*2.8;
    lunaState.saccadeX = (Math.random()-0.5)*0.06;
    lunaState.saccadeY = (Math.random()-0.5)*0.035;
  }
  const driftX = Math.sin(ts*0.24)*0.022 + Math.sin(ts*0.11)*0.012;
  const driftY = Math.sin(ts*0.18)*0.015;
  const tX = lunaState.saccadeX + driftX;
  const tY = lunaState.saccadeY + driftY;
  lunaState.lookX += (tX - lunaState.lookX) * Math.min(1, dt*6);
  lunaState.lookY += (tY - lunaState.lookY) * Math.min(1, dt*6);
  eyeLeft.rotation.y  = lunaState.lookX; eyeRight.rotation.y = lunaState.lookX;
  eyeLeft.rotation.x  = lunaState.lookY; eyeRight.rotation.x = lunaState.lookY;
}

// ── ANIMATIONS-LOOP ───────────────────────────────────────────────────────────
let lastAnimT = 0;
function animate(t) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (t - lastAnimT)/1000);
  lastAnimT = t;
  const ts = t/1000;

  // Partikel
  const ppa = particleGeo.attributes.position.array;
  for(let i=0;i<pCount;i++){
    ppa[i*3+1] += dt * 0.04;
    if(ppa[i*3+1] > 8) ppa[i*3+1] = 0;
  }
  particleGeo.attributes.position.needsUpdate = true;
  particles.rotation.y = ts*0.012;

  // Visualizer-Bars (audio-reaktiv)
  for(let i=0;i<bgBars.length;i++){
    const sv  = lunaState.specData[i % lunaState.specData.length] || 0;
    const tgt = 0.18 + sv*2.0;
    bgBars[i].scale.y += (tgt - bgBars[i].scale.y)*0.16;
    bgBars[i].position.y = bgBars[i].scale.y * 0.18;
    bgBars[i].material.emissiveIntensity = 0.08 + sv*1.0;
  }

  // Stage-Ring Glow
  // ring glow
  const ringGlow = Math.sin(ts*2.4)*0.35; // no ringMaterial in v6
  if(typeof ringMaterial!=="undefined") ringMaterial.emissiveIntensity = 0.5 + Math.sin(ts*2.4)*0.35;

  // ── LUNA KÖRPER ─────────────────────────────────────────────────────────────

  // Atmung (Torso hebt sich)
  const breathAmp = lunaState.speaking ? 0.008 : 0.014;
  torso.position.y = 0.82 + Math.sin(ts*0.82)*breathAmp;
  torso.scale.x    = 1    + Math.sin(ts*0.82)*0.005;

  // Körper-Mikro-Schwingen
  LUNA.position.y  = lunaState.speaking ? Math.sin(ts*2.9)*0.013 : Math.sin(ts*0.48)*0.004;
  LUNA.rotation.y  = Math.sin(ts*0.16)*0.016;

  // Kopf-Nicken & Neigen
  const nodAmp  = lunaState.speaking ? 0.054 : 0.028;
  const hTiltX  = lunaState.emotion==='thinking' ? 0.11 : lunaState.emotion==='explaining' ? -0.025 : 0;
  headGrp.rotation.x = hTiltX + Math.sin(ts*3.4)*nodAmp;
  headGrp.rotation.z = Math.sin(ts*0.26)*0.020 + (lunaState.emotion==='correcting' ? Math.sin(ts*4.2)*0.055 : 0);
  headGrp.rotation.y = Math.sin(ts*0.17)*0.038;

  // Arm-Animationen
  if(lunaState.isClapping){
    const cl = Math.sin(ts*6.8)*0.20;
    armLeft.rotation.x  = -0.72 + Math.abs(cl)*0.15;
    armRight.rotation.x = -0.72 + Math.abs(cl)*0.15;
    armLeft.rotation.z  = -0.32 - cl;
    armRight.rotation.z =  0.32 + cl;
  } else if(lunaState.isWaving){
    armRight.rotation.x = -0.62 + Math.sin(ts*4.2)*0.16;
    armRight.rotation.z = -0.72 + Math.sin(ts*4.2)*0.26;
    armLeft.rotation.x  =  0.12; armLeft.rotation.z = 0.20;
  } else if(lunaState.speaking && lunaState.armPose==='rest'){
    // Organische Gesten beim Sprechen
    armRight.rotation.x =  0.12 + Math.sin(ts*2.5)*0.20;
    armRight.rotation.z = -0.20 + Math.sin(ts*2.0)*0.14;
    armLeft.rotation.x  =  0.12 + Math.cos(ts*2.2)*0.07;
    armLeft.rotation.z  =  0.20 + Math.cos(ts*1.7)*0.05;
  } else if(lunaState.armPose==='rest'){
    armRight.rotation.x = 0.12 + Math.sin(ts*0.5)*0.02;
    armLeft.rotation.x  = 0.12 + Math.cos(ts*0.5)*0.02;
  }

  updateEyeLook(dt, ts);
  updateBlink(dt);
  updateMouth(dt);

  // Licht-Animation
  keyLight.intensity  = 3.5 + Math.sin(ts*0.68)*0.5;
  // footLight not in v6

  renderer.render(scene, camera);
}
animate(0);


// ── Resize Handler ────────────────────────────────────────────────────────
function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w/h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
resize();
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe(container);

// ── ÖFFENTLICHE LUNA API ──────────────────────────────────────────────────────
window.LUNA_API = {
  setEmotion(e)    { applyEmotion(e); },
  setArmPose(p)    { setLunaPose(p); },
  // v6 compat alias
  setArmPoseV6:    function(p) { setLunaPose(p); },
  setSpeaking(v)   { lunaState.speaking = v; },
  setMouthOpen(v)  { lunaState.mouthTarget = Math.max(0,Math.min(1,v)); },
  updateSpectrum(a){ lunaState.specData = a; },
  celebrate() {
    applyEmotion('happy'); setLunaPose('clap');
    setTimeout(()=>{ lunaState.isClapping=false; setLunaPose('rest'); applyEmotion('neutral'); }, 4200);
  },
  correct() {
    applyEmotion('correcting'); setLunaPose('pointing_right');
    setTimeout(()=>{ setLunaPose('rest'); applyEmotion('explaining'); }, 3000);
  },
  wave() {
    lunaState.isWaving = true; applyEmotion('happy');
    setTimeout(()=>{ lunaState.isWaving=false; setLunaPose('rest'); applyEmotion('neutral'); }, 3200);
  },
  showInstrument(instr) {
    if(instr==='guitar') setLunaPose('guitar');
    else if(instr==='piano') setLunaPose('piano');
  },
};


// ════════════════════════════════════════════════════════════════════════════
//  LUNA SPRACHSYSTEM v8 – Audio-Unlock, zuverlässige TTS, Mikrofon
// ════════════════════════════════════════════════════════════════════════════
let speechQueue    = [];
let isSpeaking     = false;
let speechLang     = 'de-DE';
const LANG_THINKING = 'KI denkt...';
window._lastLunaMsg = null;
let mouthInterval  = null;
let preferredVoice = null;
let recognition    = null;
let micActive      = false;
let _audioUnlocked = false;      // Browser-Audio durch User-Klick freigegeben
let _pendingGreeting = null;     // Begrüßungstext warten bis Unlock

// ── Beste verfügbare Stimme wählen ──────────────────────────────────────────
function pickPreferredVoice() {
  if(!('speechSynthesis' in window)) return;
  const voices = window.speechSynthesis.getVoices();
  if(!voices.length) return;
  // Bevorzugte Stimmen – natürlichste zuerst
  const order = speechLang === 'de-DE'
    ? ['Google Deutsch', 'Marlene', 'Katja', 'Anna', 'Helena', 'Hedda',
       'Microsoft Katja', 'Microsoft Hedda', 'Sabine',
       'de-DE', 'de_DE', 'de-AT', 'de-CH', 'de']
    : ['Samantha', 'Google US English', 'Karen', 'Moira', 'Victoria',
       'Microsoft Zira', 'Microsoft Eva',
       'en-US', 'en_US', 'en-GB', 'en'];
  for(const p of order){
    const v = voices.find(x =>
      x.name===p || x.name.includes(p) || x.lang===p ||
      x.lang.startsWith(p.slice(0,5)) || x.lang.startsWith(p.slice(0,2)));
    if(v){ preferredVoice=v; console.log('[TTS] Stimme:',v.name,v.lang); return; }
  }
  // letzter Fallback: irgendeine Stimme passender Sprache
  const fb = voices.find(x => x.lang.startsWith(speechLang.slice(0,2)));
  if(fb){ preferredVoice=fb; console.log('[TTS] Fallback:',fb.name,fb.lang); }
}
if('speechSynthesis' in window){
  pickPreferredVoice();
  window.speechSynthesis.onvoiceschanged = pickPreferredVoice;
}

// ── Tatsächlich sprechen ─────────────────────────────────────────────────────
function lunaSay(text, emotion, priority) {
  if(!text) return;
  emotion  = emotion  || 'explaining';
  priority = priority || false;
  if(priority) speechQueue.unshift({text, emotion});
  else         speechQueue.push   ({text, emotion});
  if(!isSpeaking) processNextSpeech();
}

function processNextSpeech() {
  if(!speechQueue.length){
    isSpeaking = false;
    if(window.LUNA_API){ LUNA_API.setSpeaking(false); LUNA_API.setMouthOpen(0); }
    clearInterval(mouthInterval);
    showSpeechBubble('');
    return;
  }
  const {text, emotion} = speechQueue.shift();
  isSpeaking = true;
  if(window.LUNA_API){ LUNA_API.setSpeaking(true); LUNA_API.setEmotion(emotion); }
  showSpeechBubble(text);

  // Mund-Animation (lebendige Phonem-Variation)
  clearInterval(mouthInterval);
  let mPhase=0;
  mouthInterval = setInterval(()=>{
    mPhase += 0.38;
    const v = 0.18 + Math.abs(Math.sin(mPhase))*0.52 +
              Math.abs(Math.sin(mPhase*1.73))*0.18 +
              Math.abs(Math.sin(mPhase*2.91))*0.12;
    if(window.LUNA_API) LUNA_API.setMouthOpen(Math.min(1, v));
  }, 85);

  if(!('speechSynthesis' in window)){
    setTimeout(()=>{ clearInterval(mouthInterval); processNextSpeech(); },
      text.length*55+800);
    return;
  }

  window.speechSynthesis.cancel();
  const utt    = new SpeechSynthesisUtterance(text);
  utt.lang     = speechLang;

  // ── Natürliche Sprech-Parameter je nach Emotion ──────────────────────
  const emoParams = {
    success:    { rate: 1.08, pitch: 1.25, volume: 1.0  },  // aufgeregt, höher
    proud:      { rate: 1.05, pitch: 1.20, volume: 1.0  },  // warm, stolz
    explaining: { rate: 0.96, pitch: 1.08, volume: 0.95 },  // ruhig, klar
    correcting: { rate: 0.92, pitch: 1.00, volume: 0.95 },  // ernst, deutlich
    neutral:    { rate: 1.00, pitch: 1.10, volume: 0.95 },  // normal
    error:      { rate: 0.94, pitch: 0.98, volume: 0.9  },  // gedämpft
  };
  const ep = emoParams[emotion] || emoParams.neutral;

  // ── Leichte Zufalls-Variation damit Luna nicht roboterhaft klingt ────
  utt.rate   = ep.rate   + (Math.random() * 0.06 - 0.03);
  utt.pitch  = ep.pitch  + (Math.random() * 0.08 - 0.04);
  utt.volume = ep.volume;

  // ── Beste Stimme wählen ──────────────────────────────────────────────
  if(preferredVoice) utt.voice = preferredVoice;

  // Chrome-Bug-Fix: speechSynthesis bleibt manchmal nach ~15 s hängen
  const resumeFix = setInterval(()=>{
    if(window.speechSynthesis.paused) window.speechSynthesis.resume();
  }, 4500);

  utt.onend = ()=>{
    clearInterval(resumeFix); clearInterval(mouthInterval);
    if(window.LUNA_API) LUNA_API.setMouthOpen(0);
    setTimeout(processNextSpeech, 250);
  };
  utt.onerror = (e)=>{
    clearInterval(resumeFix); clearInterval(mouthInterval);
    console.warn('[TTS] Fehler:', e.error);
    setTimeout(processNextSpeech, 150);
  };
  window.speechSynthesis.speak(utt);
}

// ── Sprechblase ──────────────────────────────────────────────────────────────
function showSpeechBubble(text) {
  const el = document.getElementById('speech-bubble');
  if(!el) return;
  if(text){
    el.textContent   = text;
    el.style.display = 'block';
    clearTimeout(el._t);
    el._t = setTimeout(()=>{ el.style.display='none'; },
      Math.min(14000, text.length*75+2500));
  } else {
    el.style.display = 'none';
  }
}

// ── Audio-Unlock: muss nach echtem User-Klick aufgerufen werden ──────────────
// Browser blockieren speechSynthesis.speak() ohne vorherige User-Interaktion.
// Diese Funktion wird vom Willkommens-Screen beim Klick aufgerufen.
function _doUnlockAndGreet() {
  if(_audioUnlocked) return;
  _audioUnlocked = true;

  // 1. stille Utterance erzwingt die Browser-Erlaubnis
  const silent = new SpeechSynthesisUtterance('\u200b');
  silent.volume = 0; silent.rate = 2;
  window.speechSynthesis.speak(silent);

  silent.onend = ()=>{
    window.speechSynthesis.cancel();
    pickPreferredVoice();
    // 2. kurze Pause, dann echte Begrüßung
    setTimeout(()=>{
      const text = _pendingGreeting || (speechLang.startsWith('de')
        ? 'Hallo! Ich bin NoteIQ, deine persönliche KI-Musiklehrerin! Lass uns gemeinsam Musik machen!'
        : 'Hello! I am Luna, your AI music teacher! Let us make music together!');
      _pendingGreeting = null;
      lunaSay(text, 'explaining', true);
      if(window.LUNA_API){
        LUNA_API.setArmPose('pointing_right');
        setTimeout(()=>{ LUNA_API.setArmPose('rest'); }, 4500);
      }
    }, 300);
  };
  // Fallback falls onend nicht feuert
  setTimeout(()=>{
    if(!_pendingGreeting) return;
    const text = _pendingGreeting; _pendingGreeting = null;
    lunaSay(text, 'explaining', true);
  }, 1200);
}

// ── Mikrofon / Sprach-Erkennung ──────────────────────────────────────────────
// ── Aktives Zuhören – NoteIQ hört kontinuierlich zu ──────────────────────────
let _activeListening = false;   // Dauerhaftes Zuhören an/aus
let _listenRestart   = null;    // Restart-Timer nach onend
let _lastSpeech      = 0;       // Zeitstempel letzte Sprache
let _silenceTimer    = null;    // Timer für Stille-Erkennung

function initSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){ console.warn('[Mic] SpeechRecognition nicht verfügbar'); return; }

  recognition = new SR();
  recognition.lang           = speechLang;
  recognition.continuous     = true;    // DAUERHAFT zuhören
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = ()=>{
    micActive = true;
    const btn = document.getElementById('mic-button');
    if(btn){
      btn.classList.add('active');
      btn.title = _activeListening ? 'NoteIQ hört aktiv zu – klick zum Stoppen' : 'NoteIQ hört zu';
      btn.style.background = _activeListening ? 'rgba(0,221,136,0.25)' : '';
    }
    const hdr = document.getElementById('hdr-mic-status');
    if(hdr){ hdr.textContent = _activeListening ? '👂 Aktiv…' : '🎤 Hört…'; hdr.style.display=''; }
  };

  recognition.onend = ()=>{
    micActive = false;
    const btn = document.getElementById('mic-button');
    if(btn){ btn.classList.remove('active'); btn.style.background=''; }
    const hdr = document.getElementById('hdr-mic-status');
    if(hdr) hdr.style.display='none';
    const interim = document.getElementById('mic-interim-text');
    if(interim) interim.style.display='none';

    // Auto-Restart wenn aktives Zuhören an
    if(_activeListening){
      clearTimeout(_listenRestart);
      _listenRestart = setTimeout(()=>{
        try{ recognition.start(); }catch(e){}
      }, 400);
    }
  };

  recognition.onresult = (ev)=>{
    let final='', interim='';
    for(let i=ev.resultIndex;i<ev.results.length;i++){
      if(ev.results[i].isFinal) final   += ev.results[i][0].transcript;
      else                       interim += ev.results[i][0].transcript;
    }

    const intEl = document.getElementById('mic-interim-text');
    if(intEl && interim){
      intEl.textContent = '🎤 "' + interim + '"';
      intEl.style.display = 'block';
    }

    if(final.trim()){
      _lastSpeech = Date.now();
      if(intEl) intEl.style.display = 'none';

      const text = final.trim();

      // ── Kontext-bewusste Verarbeitung ────────────────────────────────
      // Kurze Filler-Wörter ignorieren
      const fillers = ['äh','ähm','hmm','hm','oh','ah','okay','ok','ja','nein','ne'];
      if(text.split(' ').length === 1 && fillers.includes(text.toLowerCase())) return;

      // Befehle erkennen (ohne KI-API zu brauchen)
      const lower = text.toLowerCase();
      if(lower.includes('nächste lektion') || lower.includes('weiter')) {
        addChat('Du', text); handleVoiceCommand('next_lesson'); return;
      }
      if(lower.includes('wiederholen') || lower.includes('nochmal')) {
        addChat('Du', text); handleVoiceCommand('repeat'); return;
      }
      if(lower.includes('stopp') || lower.includes('stop') || lower.includes('pause')) {
        addChat('Du', text); handleVoiceCommand('stop'); return;
      }

      // Normale Frage → KI
      const inp = document.getElementById('chat-input');
      if(inp) inp.value = text;
      addChat('Du', text);
      sendAIQuestion(text);
    }
  };

  recognition.onerror = (ev)=>{
    const ignorable = ['no-speech', 'audio-capture', 'aborted'];
    if(!ignorable.includes(ev.error))
      console.warn('[Mic]', ev.error);
    micActive = false;
    if(ev.error === 'not-allowed'){
      _activeListening = false;
      lunaSay('Bitte erlaube Mikrofon-Zugriff in deinem Browser!', 'correcting');
    }
    // Bei no-speech einfach weitermachen wenn aktiv
    if(_activeListening && ev.error === 'no-speech'){
      clearTimeout(_listenRestart);
      _listenRestart = setTimeout(()=>{ try{recognition.start();}catch(e){} }, 300);
    }
  };
}
initSpeechRecognition();

// ── Sprach-Befehle (ohne KI) ───────────────────────────────────────────────
function handleVoiceCommand(cmd) {
  if(cmd === 'next_lesson'){
    fetch('/advance_phase', {method:'POST', body:'{}'})
      .then(()=>lunaSay('Weiter geht's!', 'success'));
  } else if(cmd === 'repeat'){
    const last = document.querySelector('#chat-messages .luna-msg:last-child');
    if(last) lunaSay(last.textContent, 'explaining');
    else lunaSay('Was soll NoteIQ wiederholen?', 'neutral');
  } else if(cmd === 'stop'){
    window.speechSynthesis && window.speechSynthesis.cancel();
    speechQueue.length = 0; isSpeaking = false;
    lunaSay('Okay!', 'neutral');
  }
}

// ── Toggle Einzel-Mic ──────────────────────────────────────────────────────
function toggleMic() {
  if(!recognition){
    lunaSay('Dein Browser unterstützt leider keine Sprach-Erkennung.', 'explaining');
    return;
  }
  if(micActive){
    _activeListening = false;
    recognition.stop();
  } else {
    _activeListening = false;  // Einmaliges Zuhören
    window.speechSynthesis && window.speechSynthesis.cancel();
    try{ recognition.start(); }catch(e){}
  }
}

// ── Toggle AKTIVES Zuhören (dauerhaft) ────────────────────────────────────
function toggleActiveListen() {
  if(!recognition){
    lunaSay('Sprach-Erkennung nicht verfügbar.', 'explaining'); return;
  }
  _activeListening = !_activeListening;
  const btn = document.getElementById('active-listen-btn');
  if(btn){
    btn.textContent = _activeListening ? '👂 Aktiv AN' : '👂 Aktiv';
    btn.style.background = _activeListening ? 'rgba(0,221,136,0.2)' : '';
    btn.style.borderColor = _activeListening ? 'var(--green)' : '';
    btn.style.color = _activeListening ? 'var(--green)' : '';
  }
  if(_activeListening){
    lunaSay('NoteIQ hört aktiv zu — sprich einfach drauf los!', 'explaining');
    try{ recognition.start(); }catch(e){}
  } else {
    clearTimeout(_listenRestart);
    try{ recognition.stop(); }catch(e){}
    lunaSay('NoteIQ Zuhören deaktiviert.', 'neutral');
  }
}



// ════════════════════════════════════════════════════════════════════════════
//  AVATAR-3D ENGINE  –  eigenstaendiger Three.js Charakter-Avatar "Luna"
// ════════════════════════════════════════════════════════════════════════════
(function() {
'use strict';

const avCanvas    = document.getElementById('avatar-canvas');
const avContainer = document.getElementById('avatar-panel');
if(!avCanvas || !avContainer || typeof THREE === 'undefined') return;

// ── Renderer ────────────────────────────────────────────────────────────────
const avRenderer = new THREE.WebGLRenderer({canvas:avCanvas, antialias:true, alpha:true});
avRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
avRenderer.shadowMap.enabled = true;
avRenderer.setClearColor(0x080510, 1);

// ── Scene ───────────────────────────────────────────────────────────────────
const avScene = new THREE.Scene();

// ── Camera ──────────────────────────────────────────────────────────────────
const avCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 50);
avCamera.position.set(0, 1.55, 3.6);
avCamera.lookAt(0, 1.4, 0);

// ── Lights ──────────────────────────────────────────────────────────────────
avScene.add(new THREE.AmbientLight(0x302050, 1.0));

const keyL = new THREE.PointLight(0xFF9240, 3.0, 10);
keyL.position.set(-1.5, 3.5, 2.5); keyL.castShadow = true;
avScene.add(keyL);

const fillL = new THREE.PointLight(0x4088FF, 1.4, 8);
fillL.position.set(2, 2, 1.5);
avScene.add(fillL);

const rimL = new THREE.PointLight(0xFF3388, 1.2, 6);
rimL.position.set(0, 3, -2);
avScene.add(rimL);

const topL = new THREE.DirectionalLight(0xFFE8D0, 0.7);
topL.position.set(0, 5, 2);
avScene.add(topL);

// ── Floor + Glow ─────────────────────────────────────────────────────────────
const floorM = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 8),
  new THREE.MeshLambertMaterial({color: 0x050310})
);
floorM.rotation.x = -Math.PI/2;
floorM.receiveShadow = true;
avScene.add(floorM);

// Glow ring on floor
const ringM = new THREE.Mesh(
  new THREE.TorusGeometry(0.55, 0.04, 8, 48),
  new THREE.MeshBasicMaterial({color: 0xFF6B00, transparent: true, opacity: 0.35})
);
ringM.rotation.x = -Math.PI/2;
ringM.position.y = 0.01;
avScene.add(ringM);

// Particles
const pGeo = new THREE.BufferGeometry();
const pN = 120, pPos = new Float32Array(pN * 3);
for(let i = 0; i < pN; i++) {
  pPos[i*3]   = (Math.random()-.5)*3;
  pPos[i*3+1] = Math.random()*4;
  pPos[i*3+2] = (Math.random()-.5)*3 - 1;
}
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(pGeo,
  new THREE.PointsMaterial({color:0xFF6B00, size:0.03, transparent:true, opacity:0.5}));
avScene.add(particles);

// ── MATERIALS ────────────────────────────────────────────────────────────────
const M = {
  skin:   new THREE.MeshPhongMaterial({color:0xF0B88A, shininess:28}),
  skin2:  new THREE.MeshPhongMaterial({color:0xE4A878, shininess:18}),
  hair:   new THREE.MeshPhongMaterial({color:0x1E0802, shininess:70, specular:0x604020}),
  eyeW:   new THREE.MeshPhongMaterial({color:0xF8F2EE}),
  iris:   new THREE.MeshPhongMaterial({color:0x5C2A0E, emissive:0x120400}),
  pupil:  new THREE.MeshPhongMaterial({color:0x030303}),
  lip:    new THREE.MeshPhongMaterial({color:0xBB3050, shininess:80}),
  cloth:  new THREE.MeshPhongMaterial({color:0x130630, shininess:8}),
  clothO: new THREE.MeshPhongMaterial({color:0xFF6B00, shininess:24}),
  gold:   new THREE.MeshPhongMaterial({color:0xFFCC20, shininess:140}),
  tooth:  new THREE.MeshPhongMaterial({color:0xFFF8F0, shininess:100}),
  tongue: new THREE.MeshPhongMaterial({color:0xD05868}),
  mouthI: new THREE.MeshPhongMaterial({color:0x30081A}),
  shoe:   new THREE.MeshPhongMaterial({color:0x060614, shininess:95}),
  lash:   new THREE.MeshPhongMaterial({color:0x0A0606}),
};

// ── AVATAR ROOT ──────────────────────────────────────────────────────────────
const AV = new THREE.Group();
AV.position.y = 0;
avScene.add(AV);

// ── HELPER BUILDERS ──────────────────────────────────────────────────────────
function mkMesh(geo, mat, parent, pos, rot, sc) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true; m.receiveShadow = true;
  if(pos) m.position.set(...pos);
  if(rot) m.rotation.set(...rot);
  if(sc)  m.scale.set(...sc);
  parent.add(m);
  return m;
}

// ── TORSO ────────────────────────────────────────────────────────────────────
const torso = new THREE.Group();
torso.position.y = 0.82;
AV.add(torso);

mkMesh(new THREE.CylinderGeometry(0.25, 0.19, 0.88, 18), M.cloth, torso);

// Collar
const collar = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.038, 8, 20), M.clothO);
collar.rotation.x = Math.PI/2; collar.position.y = 0.45;
torso.add(collar);

// Neck
mkMesh(new THREE.CylinderGeometry(0.075, 0.095, 0.22, 10), M.skin, torso, [0, 0.56, 0]);

// ── HEAD GROUP ───────────────────────────────────────────────────────────────
const headGrp = new THREE.Group();
headGrp.position.y = 0.78;
torso.add(headGrp);

// Head shape
const headM = mkMesh(new THREE.SphereGeometry(0.255, 24, 20), M.skin, headGrp);
headM.scale.set(1, 1.10, 0.92);

// Chin
mkMesh(new THREE.SphereGeometry(0.14, 12, 10), M.skin, headGrp, [0, -0.20, 0.06], null, [0.96, 0.50, 0.86]);

// Cheekbones
[-1,1].forEach(s => {
  mkMesh(new THREE.SphereGeometry(0.085, 8, 8), M.skin2, headGrp, [s*0.195, 0.01, 0.18], null, [0.68, 0.48, 0.58]);
});

// Nose
mkMesh(new THREE.SphereGeometry(0.040, 10, 8), M.skin, headGrp, [0, -0.04, 0.248], null, [0.9, 0.7, 0.8]);

// ── EYES ─────────────────────────────────────────────────────────────────────
const eyes = [];
const upperLids = [], lowerLids = [];

function mkEye(sx) {
  const g = new THREE.Group();
  g.position.set(sx*0.100, 0.060, 0.222);
  headGrp.add(g);

  // Socket
  mkMesh(new THREE.SphereGeometry(0.060, 12, 10), M.skin2, g, null, null, [1, 0.78, 0.44]);

  // White
  mkMesh(new THREE.SphereGeometry(0.048, 14, 12), M.eyeW, g, [0, 0, 0.003]);

  // Iris
  mkMesh(new THREE.CircleGeometry(0.028, 16), M.iris, g, [0, 0, 0.048]);

  // Pupil
  mkMesh(new THREE.CircleGeometry(0.016, 14), M.pupil, g, [0, 0, 0.050]);

  // Highlight
  mkMesh(new THREE.CircleGeometry(0.007, 8),
    new THREE.MeshBasicMaterial({color:0xFFFFFF}), g, [-0.010, 0.010, 0.051]);

  // Upper Eyelid
  const uLid = mkMesh(new THREE.SphereGeometry(0.050, 12, 6), M.skin, g, [0, 0.010, 0.004], null, [1.06, 0.45, 0.50]);
  uLid.position.y = 0.028;
  upperLids.push(uLid);

  // Lashes
  mkMesh(new THREE.CylinderGeometry(0.002, 0.001, 0.018, 4), M.lash, g, [0, 0.056, 0.044], [0, 0, 0.3]);

  // Brow
  const browGeo = new THREE.CapsuleGeometry ? new THREE.CapsuleGeometry(0.004, 0.030, 3, 6) : new THREE.CylinderGeometry(0.004, 0.004, 0.035, 6);
  mkMesh(browGeo, M.lash, g, [0, 0.092, 0.022], [0, 0, sx*0.22]);

  eyes.push(g);
  return g;
}
mkEye(-1); mkEye(1);

// ── MOUTH (morphable) ────────────────────────────────────────────────────────
const mouthGrp = new THREE.Group();
mouthGrp.position.set(0, -0.100, 0.238);
headGrp.add(mouthGrp);

// Interior
const mouthInt = mkMesh(new THREE.SphereGeometry(0.038, 12, 8), M.mouthI, mouthGrp, null, null, [1, 0.6, 0.7]);

// Upper lip
const upLip = mkMesh(new THREE.SphereGeometry(0.040, 12, 6), M.lip, mouthGrp, [0, 0.010, 0.004], null, [1, 0.50, 0.70]);

// Lower lip
const loLip = mkMesh(new THREE.SphereGeometry(0.038, 12, 6), M.lip, mouthGrp, [0, -0.012, 0.004], null, [1, 0.52, 0.72]);

// Teeth
const teethM = mkMesh(new THREE.BoxGeometry(0.050, 0.014, 0.008), M.tooth, mouthGrp, [0, 0.007, 0.018]);

// Tongue
const tongueM = mkMesh(new THREE.SphereGeometry(0.022, 8, 6), M.tongue, mouthGrp, [0, -0.006, 0.012], null, [1, 0.5, 0.8]);

// ── HAIR ─────────────────────────────────────────────────────────────────────
const hairGrp = new THREE.Group();
hairGrp.position.y = 0.14;
headGrp.add(hairGrp);

// Main cap
mkMesh(new THREE.SphereGeometry(0.272, 22, 16), M.hair, hairGrp, null, null, [1, 0.72, 0.95]);

// Bangs
mkMesh(new THREE.SphereGeometry(0.18, 14, 8), M.hair, hairGrp, [0, -0.13, 0.19], null, [1.1, 0.52, 0.60]);

// Side strands
[-1,1].forEach(s => {
  mkMesh(new THREE.CylinderGeometry(0.06, 0.03, 0.38, 8), M.hair, hairGrp, [s*0.22, -0.28, 0.04],
    [0, 0, s*0.22]);
});

// Back bun
mkMesh(new THREE.SphereGeometry(0.13, 12, 10), M.hair, hairGrp, [0, -0.18, -0.25]);

// Hair shine
mkMesh(new THREE.SphereGeometry(0.12, 8, 6),
  new THREE.MeshPhongMaterial({color:0x6A3020, shininess:200, specular:0xFFAA60, transparent:true, opacity:0.35}),
  hairGrp, [0.05, 0.12, 0.12]);

// ── EARS + EARRINGS ───────────────────────────────────────────────────────────
[-1,1].forEach(s => {
  mkMesh(new THREE.SphereGeometry(0.042, 8, 6), M.skin2, headGrp, [s*0.263, 0.010, 0], null, [0.45, 0.72, 0.52]);
  mkMesh(new THREE.SphereGeometry(0.018, 8, 6), M.gold, headGrp, [s*0.270, -0.062, 0]);
});

// ── ARMS ─────────────────────────────────────────────────────────────────────
const armGrps = {L: null, R: null};

function mkArm(sx) {
  const grp = new THREE.Group();
  grp.position.set(sx*0.305, 0.36, 0);
  torso.add(grp);

  // Upper arm
  mkMesh(new THREE.CylinderGeometry(0.072, 0.058, 0.32, 10), M.cloth, grp, [0, -0.16, 0]);

  // Elbow joint
  mkMesh(new THREE.SphereGeometry(0.062, 8, 8), M.cloth, grp, [0, -0.32, 0]);

  // Forearm
  mkMesh(new THREE.CylinderGeometry(0.055, 0.042, 0.28, 10), M.cloth, grp, [0, -0.50, 0]);

  // Hand
  mkMesh(new THREE.SphereGeometry(0.058, 10, 8), M.skin, grp, [0, -0.66, 0], null, [1, 0.76, 0.80]);

  // Fingers (3 visible)
  [-0.03, 0, 0.03].forEach((xo, fi) => {
    mkMesh(new THREE.CylinderGeometry(0.012, 0.009, 0.06, 5), M.skin, grp, [xo, -0.72, 0.04 + fi*0.01], [0.3, 0, 0]);
  });

  return grp;
}
armGrps.L = mkArm(-1);
armGrps.R = mkArm(1);

// Arm rest angles
armGrps.L.rotation.z =  0.28;
armGrps.R.rotation.z = -0.28;

// ── LOWER BODY ───────────────────────────────────────────────────────────────
const hipGrp = new THREE.Group();
hipGrp.position.y = 0.36;
torso.add(hipGrp);

// Skirt
const skirtM = mkMesh(new THREE.CylinderGeometry(0.30, 0.42, 0.60, 20), M.cloth, hipGrp, [0, -0.60, 0]);

// Skirt trim
mkMesh(new THREE.TorusGeometry(0.40, 0.020, 6, 30), M.clothO, hipGrp, [0, -0.90, 0], [Math.PI/2, 0, 0]);

// Legs
[-1,1].forEach(s => {
  mkMesh(new THREE.CylinderGeometry(0.06, 0.05, 0.55, 10), M.skin, hipGrp, [s*0.10, -1.24, 0]);
  // Shoe
  mkMesh(new THREE.BoxGeometry(0.10, 0.06, 0.18), M.shoe, hipGrp, [s*0.10, -1.55, 0.04]);
  // Heel
  mkMesh(new THREE.CylinderGeometry(0.025, 0.020, 0.08, 6), M.shoe, hipGrp, [s*0.10, -1.55, -0.07]);
});

// ── ANIMATION STATE ──────────────────────────────────────────────────────────
let avState = {
  mood:      'neutral',
  speaking:  false,
  mouthOpen: 0,
  blink:     0,
  blinkT:    0,
  nextBlink: 2.5,
  breathT:   0,
  swayT:     0,
  armWaveT:  -1,
  mWavePhase:0,
};

const MOOD_CONFIG = {
  neutral:    {armL: 0.28,  armR:-0.28, headTilt: 0,     bodySwayAmp:0.018},
  happy:      {armL:-0.50,  armR:-0.50, headTilt: 0.08,  bodySwayAmp:0.030},
  explaining: {armL: 0.18,  armR:-0.80, headTilt:-0.05,  bodySwayAmp:0.022},
  correcting: {armL: 0.35,  armR: 0.20, headTilt:-0.10,  bodySwayAmp:0.012},
  proud:      {armL:-0.60,  armR:-0.60, headTilt: 0.06,  bodySwayAmp:0.025},
  thinking:   {armL: 0.30,  armR:-1.10, headTilt: 0.14,  bodySwayAmp:0.010},
};

const STATUS_TXT = {
  neutral:    'KI-Musiklehrerin \u00b7 Bereit',
  happy:      'Grossartig gespielt!',
  explaining: 'Erklaert gerade...',
  correcting: 'Korrigiert Haltung',
  proud:      'Sehr beeindruckend!',
  thinking:   'Analysiert...',
};

// Current arm angles (lerped)
let curArmL = 0.28, curArmR = -0.28;
let curHeadTilt = 0;

// ── RESIZE ───────────────────────────────────────────────────────────────────
function avResize() {
  const w = avContainer.clientWidth || 240;
  const h = avCanvas.clientHeight  || 400;
  avCamera.aspect = w / h;
  avCamera.updateProjectionMatrix();
  avRenderer.setSize(w, h, false);
}
avResize();
new ResizeObserver(avResize).observe(avContainer);

// ── ANIMATION LOOP ────────────────────────────────────────────────────────────
const avClock = new THREE.Clock();

function avAnimate() {
  requestAnimationFrame(avAnimate);
  const dt = avClock.getDelta();
  const t  = avClock.getElapsedTime();

  // ── Particles drift
  particles.rotation.y += 0.003 * dt * 60;

  // ── Ring glow pulse
  ringM.material.opacity = 0.20 + Math.sin(t * 1.4) * 0.12;

  // ── Breathing
  avState.breathT += dt;
  const breath = Math.sin(avState.breathT * 1.2) * 0.015;
  torso.position.y = 0.82 + breath;

  // ── Body sway
  avState.swayT += dt;
  const cfg = MOOD_CONFIG[avState.mood] || MOOD_CONFIG.neutral;
  AV.rotation.z = Math.sin(avState.swayT * 0.55) * cfg.bodySwayAmp;
  AV.rotation.y = Math.sin(avState.swayT * 0.38) * 0.025;

  // ── Head subtle movement
  const hNodAmp  = avState.speaking ? 0.040 : 0.018;
  headGrp.rotation.x = cfg.headTilt + Math.sin(avState.swayT * 0.72) * hNodAmp;
  headGrp.rotation.y = Math.sin(avState.swayT * 0.44) * 0.055;
  headGrp.rotation.z = Math.sin(avState.swayT * 0.31) * 0.022;

  // ── Arm lerp toward mood target
  const armSpeed = 3.0 * dt;
  curArmL += (cfg.armL - curArmL) * armSpeed;
  curArmR += (cfg.armR - curArmR) * armSpeed;

  if(avState.armWaveT > 0) {
    // Wave animation overrides R arm
    avState.armWaveT -= dt;
    armGrps.R.rotation.z = -0.80 + Math.sin(avState.armWaveT * 8) * 0.45;
  } else {
    armGrps.R.rotation.z = curArmR + Math.sin(t * 0.6) * 0.018;
  }
  armGrps.L.rotation.z = curArmL + Math.sin(t * 0.6 + 1.3) * 0.018;

  // ── Blink system
  avState.blinkT += dt;
  if(avState.blinkT > avState.nextBlink) {
    avState.blink    = 1;
    avState.blinkT   = 0;
    avState.nextBlink = 2.2 + Math.random() * 3.5;
  }
  if(avState.blink > 0) {
    avState.blink = Math.max(0, avState.blink - dt * 9);
    const lidY = avState.blink * 0.048;
    upperLids.forEach(lid => { lid.position.y = 0.028 - lidY; });
  }

  // ── Mouth morphing
  if(avState.speaking) {
    avState.mWavePhase += dt * 8.5;
    const mo = Math.abs(Math.sin(avState.mWavePhase)) * 0.6 +
               Math.abs(Math.sin(avState.mWavePhase * 1.8)) * 0.25 +
               Math.abs(Math.sin(avState.mWavePhase * 3.1)) * 0.15;
    avState.mouthOpen += (mo - avState.mouthOpen) * 0.25;
  } else {
    avState.mouthOpen *= 0.82;
  }

  const mo = avState.mouthOpen;
  mouthInt.scale.y = 0.4 + mo * 0.9;
  upLip.position.y =  0.010 + mo * 0.022;
  loLip.position.y = -0.012 - mo * 0.026;
  teethM.position.y = 0.007 + mo * 0.010;
  tongueM.position.y = -0.006 - mo * 0.014;
  tongueM.visible = mo > 0.3;

  // ── Keylight subtle flicker when speaking
  if(avState.speaking) {
    keyL.intensity = 3.0 + Math.sin(t * 12) * 0.35;
  } else {
    keyL.intensity += (3.0 - keyL.intensity) * 0.05;
  }

  avRenderer.render(avScene, avCamera);
}
avAnimate();

// ── PUBLIC API ────────────────────────────────────────────────────────────────
window.AVATAR3D = {
  setMood(mood) {
    avState.mood = mood;
    const st = document.getElementById('av-status');
    if(st) st.textContent = STATUS_TXT[mood] || STATUS_TXT.neutral;
    ['neutral','happy','explaining','correcting'].forEach(m => {
      const b = document.getElementById('avb-'+m);
      if(b) b.classList.toggle('active', m === mood);
    });
  },
  setSpeaking(v) { avState.speaking = !!v; },
  wave() { avState.armWaveT = 2.5; },
  celebrate() {
    this.setMood('happy');
    AV.position.y = 0;
    let wt = 0;
    const iv = setInterval(() => {
      wt += 0.05;
      AV.position.y = Math.abs(Math.sin(wt * 5)) * 0.15;
      if(wt > 1.2) { clearInterval(iv); AV.position.y = 0; }
    }, 50);
  },
};

// ── SYNC with main Luna state ─────────────────────────────────────────────────
setInterval(() => {
  const etag = document.getElementById('emotion-tag');
  if(etag) {
    const txt = etag.textContent;
    let mood = 'neutral';
    if(txt.includes('Super')||txt.includes('Toll')||txt.includes('proud'))    mood='happy';
    else if(txt.includes('Erkl')||txt.includes('explaining'))                 mood='explaining';
    else if(txt.includes('Korr')||txt.includes('correcting'))                 mood='correcting';
    else if(txt.includes('Stolz')||txt.includes('Star'))                      mood='proud';
    else if(txt.includes('Hmm')||txt.includes('think'))                       mood='thinking';
    if(mood !== avState.mood) window.AVATAR3D.setMood(mood);
  }
  const sb = document.getElementById('speech-bubble');
  const speaking = sb && sb.style.display !== 'none' && sb.textContent.trim().length > 0;
  if(speaking !== avState.speaking) window.AVATAR3D.setSpeaking(speaking);
}, 350);

})(); // end AVATAR3D IIFE

// Helper exposed globally
function avSetMood(mood) {
  if(window.AVATAR3D) AVATAR3D.setMood(mood);
  if(window.LUNA_API) LUNA_API.setEmotion(mood);
}

//  STATE MANAGEMENT & UI UPDATES
// ════════════════════════════════════════════════════════════════════════════
let STATE = null;
let LessonData = null;
let currentLessonIdx = 0;
let activeTab = 'lesson';
let activeRTab = 'audio';
let prevFeedback = [];
let sessionStats = {acc:[], posture:[], rhythm:[], poly:[], startTime:Date.now()};
let lastLunaComment = 0;
let lunaCommentInterval = 12000;

// Spektrum init
const SPEC_N = 32;
const specEl = document.getElementById('rp-spectrum');
if(specEl) {
  for(let i=0;i<SPEC_N;i++){
    const b=document.createElement('div');b.className='spec-bar';
    b.style.cssText='flex:1;min-width:2px;height:4px;border-radius:1px 1px 0 0;transition:height .04s;';
    specEl.appendChild(b);
  }
}
// Cam strings init (6 saiten)
const camStrEl = document.getElementById('cam-strings');
if(camStrEl) {
  for(let i=0;i<6;i++){
    const d=document.createElement('div');d.className='str-col';
    d.innerHTML=`<div class="str-note" id="cs-n${i}"></div>
      <div class="str-bar" id="cs-b${i}" style="width:14px;height:22px">
        <div class="str-fill" id="cs-f${i}" style="background:var(--success)"></div>
      </div>`;
    camStrEl.appendChild(d);
  }
}
// Poly strings right
const polyStrR = document.getElementById('poly-strings-r');
if(polyStrR) {
  const STR_L=['E','A','D','G','H','e'];
  for(let i=0;i<6;i++){
    const d=document.createElement('div');d.className='str-col';
    d.innerHTML=`<div class="str-note" id="ps-n${i}"></div>
      <div class="str-bar" id="ps-b${i}"><div class="str-fill" id="ps-f${i}" style="height:0%"></div></div>
      <div class="str-label">${STR_L[i]}</div>`;
    polyStrR.appendChild(d);
  }
}

function setLTab(t) {
  activeTab=t;
  ['lesson','chords','song','spanish'].forEach(id=>{
    const el=document.getElementById('ltab-'+id);
    if(el) el.style.display=id===t?'':'none';
  });
  document.querySelectorAll('.panel:first-child .tab').forEach(el=>{
    const tabId = el.getAttribute('onclick')?.match(/'([a-z]+)'/)?.[1];
    el.classList.toggle('active', tabId===t);
  });
  if(t==='spanish') renderFlairTab();
}

/* ── Spanish/Latin Flair Tab ─────────────────────────────────────────────── */
let SpecialLessons  = [];   // Aus STATE.special_lessons
let flairFilter     = 'all';
let currentPlayerLv = 1;

function updateSpanishTabVisibility(level) {
  currentPlayerLv = level;
  const tab = document.getElementById('tab-spanish');
  if(!tab) return;
  // Tab ab Level 3 anzeigen (mit Pulse-Badge)
  if(level >= 3) {
    tab.style.display = '';
    const badge = document.getElementById('spanish-tab-badge');
    // Badge nach erstmaligem Sehen entfernen
    const seen = localStorage.getItem('niq_flair_seen');
    if(seen) { if(badge) badge.remove(); }
  }
}

function filterFlair(cat) {
  flairFilter = cat;
  document.querySelectorAll('.flair-filter-btn').forEach(b=>{
    b.classList.toggle('active', b.dataset.filter===cat || (cat==='all' && b.dataset.filter==='all'));
  });
  renderFlairTab();
}

function renderFlairTab() {
  const el = document.getElementById('flair-lesson-list');
  if(!el) return;

  const notice = document.getElementById('spanish-unlock-notice');
  const msg    = document.getElementById('spanish-unlock-msg');

  if(currentPlayerLv < 3) {
    if(notice) notice.style.display = '';
    if(msg) msg.textContent = `Noch ${3 - currentPlayerLv} Level bis Flamenco & Latin freigeschaltet werden!`;
  } else {
    if(notice) notice.style.display = 'none';
  }

  const lessons = SpecialLessons.filter(l=>{
    if(flairFilter !== 'all' && l.category !== flairFilter) return false;
    return true;
  });

  if(!lessons.length) {
    el.innerHTML = '<div style="font-size:.65rem;color:var(--muted);padding:12px 0;text-align:center">Keine Lektionen verfügbar</div>';
    return;
  }

  // Gruppiert nach Kategorie
  const groups = {};
  lessons.forEach(l=>{ (groups[l.category]||=[]).push(l); });

  const ICONS = { Spanish:'💃', Latin:'🌴' };
  const COLORS = { Spanish:'#FFB400', Latin:'#00D490' };

  el.innerHTML = Object.entries(groups).map(([cat, ls]) => `
    <div class="spanish-section-header">
      <span>${ICONS[cat]||'🎸'}</span>
      <span class="spanish-section-title">${cat}</span>
      <span class="spanish-section-sub">${ls.length} Lektionen</span>
    </div>
    ${ls.map(l => {
      const locked = !l.unlocked;
      const catCls = cat.toLowerCase();
      return `
      <div class="lesson-card ${catCls} ${locked?'locked':''}"
           onclick="${locked?`showFlairLocked(${l.unlock_level})`:`selectFlairLesson(${l.idx})`}"
           title="${locked?'🔒 Level '+l.unlock_level+' erforderlich':l.title}">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div style="display:flex;align-items:center;gap:4px">
            <div class="lesson-level">Lv.${l.level}</div>
            <span class="lesson-flair-badge ${locked?'flair-locked':'flair-'+catCls}">
              ${locked ? '🔒 Lv.'+l.unlock_level : cat}
            </span>
          </div>
          <span style="font-size:.58rem;color:var(--yellow);font-family:var(--mono)">+${l.xp_reward||200} XP</span>
        </div>
        <div class="lesson-title" style="color:${locked?'var(--muted)':'#fff'}">
          ${l.title}
        </div>
        <div class="lesson-meta">${l.duration_min} Min · ${(l.chords||[]).length} Akkorde${locked?' · 🔒':''}</div>
      </div>`;
    }).join('')}
  `).join('');
}

function selectFlairLesson(globalIdx) {
  localStorage.setItem('niq_flair_seen','1');
  const badge = document.getElementById('spanish-tab-badge');
  if(badge) badge.remove();
  fetch('/select_lesson',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({idx:globalIdx})})
    .then(()=>{ setLTab('lesson'); });
}

function showFlairLocked(requiredLevel) {
  const diff = requiredLevel - currentPlayerLv;
  const msg = diff>0
    ? `Noch ${diff} Level um diese Lektion freizuschalten!\nSpiele mehr Standard-Lektionen um XP zu sammeln.`
    : `Diese Lektion ist ab Level ${requiredLevel} verfügbar.`;
  lunaSay(`Noch ${diff} Level bis diese Lektion freigeschaltet wird! Übe weiter!`, 'explaining');
  addChat('NoteIQ', msg.replace('\n',' '));
}
function setRTab(t) {
  activeRTab=t;
  ['audio','poly','stats','metro'].forEach(id=>{
    const el=document.getElementById('rtab-'+id);
    if(el) el.style.display=id===t?'':'none';
  });
  document.querySelectorAll('.panel.right .tab').forEach((el,i)=>{
    el.classList.toggle('active',['audio','poly','stats','metro'][i]===t);
  });
}

async function poll() {
  try {
    const r=await fetch('/state');
    if(!r.ok) return;
    STATE=await r.json();
    updateUI(STATE);
  } catch(e){}
}

function updateUI(s) {
  if(!s) return;
  const p=s.profile||{};
  const au=s.audio||{};
  const ch=s.chord||{};
  const poly=au.poly||{};
  const strum=au.strum||{};
  const song=s.song||{};
  const tut=s.tutorial||{};
  const ts=s.teacher||{};

  // Dynamische Fehler: Audio plötzlich verloren
  const sys = s.system_status || {};
  if(au.active === false && !window._audioErrShown) {
    window._audioErrShown = true;
    const errMsg = sys.audio_error || 'Mikrofon-Signal verloren';
    showErrToast('warn','Audio-Problem', errMsg, false);
  } else if(au.active) {
    window._audioErrShown = false;
  }

  // Header
  document.getElementById('hdr-chord').textContent=ch.data?.name||ch.key||'–';
  document.getElementById('hdr-level').textContent='Lv.'+p.level;
  document.getElementById('hdr-audio').textContent=au.active?'🎤':'🔇';
  const lName=ts.lesson_title||'';
  document.getElementById('hdr-lesson').textContent=lName?'📚 '+lName.slice(0,18):'–';
  document.getElementById('hdr-song-name').textContent=song.active?'♪ '+song.song_key:'';

  // 3D Teacher updates
  if(window.LUNA_API) {
    const spec=(au.spectrum||[]).slice(0,SPEC_N);
    window.LUNA_API.updateSpectrum(spec);
    // Emotion basierend auf State
    if(ts.emotion) window.LUNA_API.setEmotion(ts.emotion);
    if(ts.arm_pose) window.LUNA_API.setArmPose(ts.arm_pose);
    if(ts.speaking) window.LUNA_API.setSpeaking(true);
  }

  // Lesson tab
  const an_acc=ch.acc||0;
  const an_post=s.posture?.wrist_ok!==false;
  const an_rhy=strum.measured_bpm||0;
  const an_cln=poly.chord_score||0;
  sessionStats.acc.push(an_acc);
  sessionStats.posture.push(an_post?1:0);
  sessionStats.poly.push(an_cln);
  if(an_rhy) sessionStats.rhythm.push(an_rhy);

  setAnalysisScore('an-acc', an_acc*100, '%');
  setAnalysisScore('an-posture', an_post?100:0, '%', !an_post);
  setAnalysisScore('an-rhythm', an_rhy>0?an_rhy:0, ' BPM');

  // v14: echte Cleanliness statt chord_score
  const real_clean = au.cleanliness != null ? au.cleanliness : an_cln;
  setAnalysisScore('an-clean', real_clean*100, '%', au.buzz);

  // Buzz-Indikator
  const buzzEl = document.getElementById('an-buzz');
  if(buzzEl) {
    buzzEl.textContent = au.buzz ? '⚡ Schnarren' : '';
    buzzEl.style.color = au.buzz ? 'var(--red)' : 'var(--green)';
    buzzEl.style.display = au.buzz ? '' : 'none';
  }

  // Attack-Indikator (Aufleuchten beim Anschlag)
  const attackEl = document.getElementById('rp-onset');
  if(attackEl) {
    attackEl.style.display = au.attack ? '' : 'none';
    attackEl.textContent = au.attack ? '⚡' : '';
  }

  // Harmonic Ratio (Oberton-Qualität)
  const harmEl = document.getElementById('rp-harm');
  if(harmEl) harmEl.textContent = au.harm_ratio != null ? Math.round(au.harm_ratio*100)+'%' : '';

  // Obertöne-Karte im Analyse-Panel
  const anHarm = document.getElementById('an-harm');
  if(anHarm) {
    const hr = au.harm_ratio != null ? Math.round(au.harm_ratio*100) : 0;
    anHarm.textContent = hr + '%';
    anHarm.className = 'analysis-score ' + (hr > 70 ? 'good' : hr > 45 ? 'ok' : 'bad');
  }

  // Finger-Status aus posture-Daten (Wrist + Alignment)
  const anFinger = document.getElementById('an-finger');
  if(anFinger) {
    const wOk = s.posture?.wrist_ok !== false;
    const wAng = s.posture?.angle || 0;
    const alignOk = s.posture?.align_ok !== false;
    if(!wOk) {
      anFinger.textContent = 'Handgelenk ↑ ' + wAng.toFixed(0) + '°';
      anFinger.className = 'analysis-score bad';
    } else if(!alignOk) {
      anFinger.textContent = 'Daumen zu hoch';
      anFinger.className = 'analysis-score ok';
    } else {
      anFinger.textContent = 'OK';
      anFinger.className = 'analysis-score good';
    }
  }

  // v11: pädagogisches System aktualisieren
  if(ts && Object.keys(ts).length) {
    updatePedSystem(ts);
    // Curriculum-Liste bei erstem Mal rendern
    if(ts.curriculum && (!LessonData || LessonData.length===0)) {
      LessonData = ts.curriculum;
      renderLessonList(LessonData);
    }
    // Spanish/Latin Flair-Lektionen laden
    if(ts.special_lessons != null) {
      SpecialLessons = ts.special_lessons;
    }
  }

  // Chord tab
  document.getElementById('lb-chord-name').textContent=ch.data?.name||ch.key||'–';
  const diff=ch.data?.diff||1;
  document.getElementById('lb-chord-diff').textContent='★'.repeat(diff)+'☆'.repeat(3-diff);
  document.getElementById('lb-chord-tip').textContent=ch.data?.tip||'';
  setBarW('lb-acc-bar','lb-acc-val',ch.acc||0);
  setBarW('lb-hold-bar','lb-hold-val',ch.hold||0);
  const ml=(ch.mastered||[]).slice(0,8).join('  ');
  document.getElementById('lb-mastered').textContent=ml?'✓ '+ml:'';

  // Audio tab
  const note=au.note_name||'–', hz=au.pitch_hz||0, cents=au.cents_off||0;
  document.getElementById('rp-note').textContent=note;
  document.getElementById('rp-hz').textContent=`${hz.toFixed(0)} Hz  ${cents>=0?'+':''}${cents.toFixed(0)}ct`;
  const needlePct=Math.min(100,Math.max(0,(cents/50)*50+50));
  const needleEl=document.getElementById('rp-needle');
  if(needleEl) needleEl.style.left=needlePct+'%';
  // Spektrum
  const spec=au.spectrum||[];
  const sBars=specEl?specEl.children:[];
  for(let i=0;i<SPEC_N;i++){
    const v=spec[Math.floor(i/SPEC_N*spec.length)]||0;
    if(sBars[i]){
      sBars[i].style.height=Math.max(2,Math.round(v*28))+'px';
      const hue=20+i/SPEC_N*20;
      sBars[i].style.background=`hsl(${hue},90%,55%)`;
    }
  }
  setW('rb-bass',(au.bass||0)*100);
  setW('rb-mid',(au.mid||0)*100);
  setW('rb-high',(au.high||0)*100);
  document.getElementById('rp-notes').textContent=(au.notes||[]).join(' ');
  const cm=au.chord_match||'',cc=au.chord_conf||0;
  const rmEl=document.getElementById('rp-match');
  if(rmEl){rmEl.textContent=cm?`~${cm} ${Math.round(cc*100)}%`:'';rmEl.style.display=cm?'':'none';}
  const onEl=document.getElementById('rp-onset');
  if(onEl) onEl.style.display=au.onset?'':'none';

  // Poly tab
  const pres=poly.string_presence||[0,0,0,0,0,0];
  const pnotes=poly.string_notes||['–','–','–','–','–','–'];
  const muted_s=poly.muted_strings||[];
  for(let i=0;i<6;i++){
    const pct=Math.round((pres[i]||0)*100);
    const f=document.getElementById(`ps-f${i}`);
    if(f){f.style.height=pct+'%';f.style.background=muted_s.includes(i+1)?'var(--danger)':(pct>30?'var(--success)':'var(--muted)');}
    const n=document.getElementById(`ps-n${i}`);
    if(n) n.textContent=pnotes[i]!=='–'?pnotes[i]:'';
    // Cam strings
    const cf=document.getElementById(`cs-f${i}`);
    if(cf){cf.style.height=pct+'%';cf.style.background=muted_s.includes(i+1)?'var(--danger)':(pct>30?'var(--success)':'var(--muted)');}
    const cn=document.getElementById(`cs-n${i}`);
    if(cn) cn.textContent=pnotes[i]!=='–'?pnotes[i]:'';
  }
  setBarW('poly-score-bar-r','poly-score-val-r',poly.chord_score||0);
  const pnEl=document.getElementById('poly-notes-r');
  if(pnEl) pnEl.textContent=(poly.poly_notes||[]).join(' ')||'–';
  const pmEl=document.getElementById('poly-muted-r');
  if(pmEl) pmEl.textContent=muted_s.length?'Saite '+muted_s.join(', '):'–';
  const pint=document.getElementById('poly-interp');
  if(pint) {
    const sc=poly.chord_score||0;
    if(sc>0.8) pint.textContent='✓ Alle Saiten klingen gut!';
    else if(sc>0.5) pint.textContent='⚡ '+muted_s.length+' Saite(n) gedämpft – Finger-Position prüfen';
    else if(sc>0.2) pint.textContent='⚠️ Mehrere Saiten unrein – Griffdruck erhöhen';
    else pint.textContent='– Kein Ton erkannt – Gitarre spielen';
  }

  // Strumming
  const sdEl=document.getElementById('strum-indicator');
  if(sdEl){sdEl.textContent=strum.last_dir||'–';sdEl.className='strum-big'+(strum.onset_flash?' flash':'');}
  const slEl=document.getElementById('strum-live');
  if(slEl) slEl.textContent=strum.pattern_live||'–';
  const stEl=document.getElementById('strum-tgt');
  if(stEl) stEl.textContent=song.cur_pattern||'–';
  const bpmEl=document.getElementById('bpm-live');
  if(bpmEl) bpmEl.textContent=strum.measured_bpm>0?strum.measured_bpm.toFixed(0)+'':'–';

  // Metronom dot
  const mDot=document.getElementById('metro-dot');
  const mBpm=document.getElementById('metro-bpm-display');
  if(s.metro){
    if(mDot){
      const p=s.metro.pulse||0;
      mDot.style.background=p>0.5?`rgba(255,107,0,${0.4+p*0.6})`:'#1A1A35';
      mDot.style.boxShadow=p>0.5?`0 0 ${p*20}px rgba(255,107,0,0.6)`:'none';
      mDot.style.borderColor=p>0.5?'var(--or)':'var(--border)';
    }
    if(mBpm) mBpm.textContent=s.metro.bpm;
  }

  // Song
  const sbEl=document.getElementById('song-bar');
  if(sbEl) sbEl.style.display=song.active?'block':'none';
  if(song.active) {
    const prog=(song.pos||0)/Math.max(1,song.total||1);
    setW('sb-prog',prog*100);
    const bw=(100/(song.total||1))*(song.bar_pos||0);
    setW('sb-barpos',bw);
    setText('sb-name','♪ '+song.song_key);
    setText('sb-cur',song.cur_chord||'–');
    setText('sb-next','→ '+(song.next_chord||'–'));
    setText('sb-score','Score: '+Math.round((song.score||0)*100)+'%');
    const alEl=document.getElementById('sb-alarm');
    if(alEl) alEl.style.display=(song.bar_pos||0)>0.75?'':'none';
    // Left panel song info
    document.getElementById('song-active-info').style.display='';
    setW('sal-prog',(song.pos/Math.max(1,song.total))*100);
    setText('sal-prog-val',Math.round(song.pos/Math.max(1,song.total)*100)+'%');
    setText('sal-cur',song.cur_chord||'–');
    setText('sal-next',song.next_chord||'–');
    setW('sal-score',(song.score||0)*100);
    setText('sal-score-val',Math.round((song.score||0)*100)+'%');
    setText('sal-name',song.song_key);
  } else {
    const sai=document.getElementById('song-active-info');
    if(sai) sai.style.display='none';
  }

  // Stats
  const sc=document.getElementById('stats-rows');
  if(sc&&activeRTab==='stats') {
    const rows=[
      ['Name',p.name],['Instrument',p.instrument||'–'],
      ['Level',p.level],['XP',`${p.xp}/${p.next_xp}`],
      ['Ø Genauigkeit',(p.avg_acc||0).toFixed(0)+'%'],
      ['Gesamt',(p.total_min||0).toFixed(0)+' Min'],
      ['Pitch',`${(au.pitch_hz||0).toFixed(1)} Hz`],
      ['Strum-BPM',(strum.measured_bpm||0).toFixed(0)],
      ['Session-Acc',sessionStats.acc.length>0?Math.round(sessionStats.acc.slice(-10).reduce((a,b)=>a+b,0)/Math.min(10,sessionStats.acc.length)*100)+'%':'–'],
    ];
    sc.innerHTML=rows.map(([k,v])=>`<div class="stat-row"><span class="stat-k">${k}</span><span class="stat-v">${v}</span></div>`).join('');
  }

  // Library (Chord tab)
  if(activeTab==='chords') renderLib(s);
  // Song list (song tab)
  if(activeTab==='song') renderSongList(s);

  // Luna AI-Feedback – nur wenn neue Antwort (Änderungs-Detection)
  if(s.ai && s.ai.last && !s.ai.thinking) {
    const lastKey = s.ai.last.slice(0,40);
    if(lastKey !== window._lastLunaMsg) {
      window._lastLunaMsg = lastKey;
      addChat('Luna', s.ai.last);
      // Auch sprechen (falls kein Server-TTS)
      if(s.ai.last !== LANG_THINKING) {
        lunaSay(s.ai.last, 'explaining');
      }
    }
  }
  // Thinking-Indikator
  if(s.ai && s.ai.thinking) {
    const hdr = document.getElementById('hdr-mic-status');
    if(hdr){ hdr.textContent = '🧠 Denkt…'; hdr.style.display = ''; }
  }

  // Feedback toasts / overlay
  const fbs=s.feedback||[];
  updateFeedback(fbs);

  // Luna AI-Kommentar-Scheduler
  const now=Date.now();
  if(now-lastLunaComment>lunaCommentInterval) {
    scheduleLunaComment(s);
    lastLunaComment=now;
  }
}

function setAnalysisScore(id, val, suffix, isBad=false) {
  const el=document.getElementById(id);
  if(!el) return;
  const pct=Math.round(val);
  el.textContent=pct+suffix;
  el.className='analysis-score '+(pct>75?'good':pct>45?'ok':'bad');
  if(isBad) el.className='analysis-score bad';
}
function setBarW(barId, valId, v) {
  const pct=Math.min(100,Math.max(0,Math.round(v*100)));
  const b=document.getElementById(barId);
  const vEl=document.getElementById(valId);
  if(b) b.style.width=pct+'%';
  if(vEl) vEl.textContent=pct+'%';
}
function setW(id, pct) { const el=document.getElementById(id); if(el) el.style.width=Math.min(100,Math.max(0,pct))+'%'; }
function setText(id, t) { const el=document.getElementById(id); if(el) el.textContent=t; }

let libRendered=false;
function renderLib(s) {
  if(libRendered) return;
  libRendered=true;
  const data=s.library?.guitar_data||{};
  const mastered=s.chord?.mastered||[];
  const cur=s.chord?.key||'';
  const groups={};
  for(const [k,v] of Object.entries(data)){const g=v.group||'?';groups[g]=groups[g]||[];groups[g].push([k,v]);}
  let html='';
  for(const [g,items] of Object.entries(groups).sort()){
    html+=`<div class="lib-group">${g}</div><div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px">`;
    for(const [k,v] of items){
      const cls=mastered.includes(k)?'border-color:var(--success);background:#0A2A10;':k===cur?'border-color:var(--or);background:#2A0800;':'';
      html+=`<div onclick="selectChord('${k}')" title="${v.tip||''}" style="background:#1A1A35;border:1px solid var(--border);border-radius:5px;padding:3px 6px;cursor:pointer;font-size:.7rem;${cls}">${v.name||k}</div>`;
    }
    html+='</div>';
  }
  document.getElementById('lib-content').innerHTML=html;
  document.getElementById('lib-search').addEventListener('input',function(){
    const q=this.value.toLowerCase();
    document.querySelectorAll('#lib-content div[title]').forEach(el=>{
      el.style.display=(el.textContent.toLowerCase().includes(q)||el.title.toLowerCase().includes(q))?'':'none';
    });
  });
}

let songListRendered=false;
function renderSongList(s) {
  if(songListRendered) return;
  songListRendered=true;
  const songs = s.songs_available || [];
  const lv    = currentPlayerLv || 1;

  // Flair-Song-Kategorien (Schlüsselwörter im Song-Namen)
  const LATIN_KEYS   = ['Ipanema','Bamba','Guantanamera','Besame','Oye Como','Chan Chan','Aranjuez'];
  const SPANISH_KEYS = ['Malagueña','Flamenco','Soleares','Aranjuez','Phrygisch'];

  const isFlairSong = (sk) => {
    return LATIN_KEYS.some(k=>sk.includes(k)) || SPANISH_KEYS.some(k=>sk.includes(k));
  };
  const getFlairCategory = (sk) => {
    if(SPANISH_KEYS.some(k=>sk.includes(k))) return 'Spanish';
    if(LATIN_KEYS.some(k=>sk.includes(k)))   return 'Latin';
    return null;
  };

  // Standard-Songs und Flair-Songs trennen
  const stdSongs   = songs.filter(sk => !isFlairSong(sk));
  const flairSongs = songs.filter(sk =>  isFlairSong(sk));

  let html = stdSongs.map(sk=>`
    <div onclick="startSong('${sk.replace(/'/g,"\\'")}') "
         style="background:#1A1A35;border:1px solid var(--border);
                border-radius:8px;padding:8px 10px;margin-bottom:5px;
                cursor:pointer;font-size:.78rem;transition:.15s"
         onmouseover="this.style.borderColor='var(--song)'"
         onmouseout="this.style.borderColor='var(--border)'">
      <div style="font-weight:700;color:var(--accent3)">♪ ${sk}</div>
    </div>`).join('');

  // Flair-Sektion
  if(flairSongs.length) {
    html += `<div style="margin:10px 0 6px;padding:6px 0 4px;
                          border-top:1px solid rgba(255,180,0,.2)">
      <span style="font-size:.58rem;font-weight:800;letter-spacing:.1em;
                   text-transform:uppercase;color:#FFB400">
        🌹 Spanish / Latin Flair
      </span>
    </div>`;
    html += flairSongs.map(sk => {
      const cat    = getFlairCategory(sk);
      const locked = lv < 3;
      const border = cat==='Spanish' ? 'rgba(255,180,0,.3)' : 'rgba(0,200,120,.25)';
      const color  = cat==='Spanish' ? '#FFB400'            : '#00D490';
      return `
      <div onclick="${locked?`showFlairLocked(3)`:`startSong('${sk.replace(/'/g,"\\'")}') `}"
           style="background:${locked?'#111':'rgba(255,100,0,.05)'};
                  border:1px solid ${locked?'var(--border)':border};
                  border-radius:8px;padding:8px 10px;margin-bottom:5px;
                  cursor:pointer;font-size:.78rem;transition:.15s;
                  opacity:${locked?.5:1}"
           onmouseover="this.style.borderColor='${color}'"
           onmouseout="this.style.borderColor='${locked?'var(--border)':border}'">
        <div style="font-weight:700;color:${locked?'var(--muted)':color}">
          ${locked?'🔒':cat==='Spanish'?'💃':'🌴'} ${sk}
        </div>
        ${locked?`<div style="font-size:.58rem;color:var(--muted)">Ab Level 3 freigeschaltet</div>`:''}
      </div>`;
    }).join('');
  }

  document.getElementById('song-list-left').innerHTML = html;
}
// ════════════════════════════════════════════════════════════════════════════
//  PÄDAGOGISCHES SYSTEM v11 JS  –  Technique → Exercise → Song → XP
// ════════════════════════════════════════════════════════════════════════════

const PHASE_COLORS = {
  technique: 'var(--accent)',
  exercise:  'var(--blue)',
  song:      'var(--green)',
  xp:        'var(--yellow)',
};

function updatePedSystem(ts) {
  if(!ts) return;
  const sess = ts.session || {};

  // XP + Level
  const lv = sess.profile_level || ts.profile?.level || 1;
  updateSpanishTabVisibility(lv);
  const xp = sess.profile_xp  ?? ts.profile?.xp ?? 0;
  const xpN = sess.xp_next    ?? ts.profile?.next_xp ?? 120;
  const xpPct = sess.xp_pct != null ? sess.xp_pct*100 : Math.round(xp/Math.max(1,xpN)*100);
  const el = document.getElementById('lv-num');   if(el) el.textContent = lv;
  const xf = document.getElementById('xp-fill');  if(xf) xf.style.width = xpPct.toFixed(1)+'%';
  const xv = document.getElementById('xp-val');   if(xv) xv.textContent = xp+'/'+xpN+' XP';
  const xe = document.getElementById('xp-earned-label');
  if(xe && sess.xp_earned) xe.textContent = '+'+sess.xp_earned+' XP';

  // Phasen-Stepper
  const phIdx = sess.phase_idx ?? 0;
  const total = sess.total_phases ?? 4;
  [0,1,2,3].forEach(i => {
    const s = document.getElementById('pstep-'+i);
    if(!s) return;
    s.className = 'phase-step' + (i < phIdx ? ' done' : i===phIdx ? ' active' : '');
  });

  // Phase Card
  const card = document.getElementById('phase-card');
  if(card) {
    const phId = sess.phase_id || 'technique';
    card.className = 'phase-card ' + phId;
    const col = PHASE_COLORS[phId] || 'var(--accent)';
    card.style.borderLeftColor = col;
  }

  const setTxt = (id,v) => { const e=document.getElementById(id); if(e) e.textContent=v||''; };
  setTxt('phase-label',   sess.phase_label);
  setTxt('phase-title',   sess.phase_title);
  setTxt('phase-content', sess.phase_content);

  // Tips
  const tipsEl = document.getElementById('phase-tips');
  if(tipsEl) {
    const tips = sess.phase_tips || [];
    tipsEl.innerHTML = tips.map(t=>`<li>${t}</li>`).join('');
  }

  // Meta-Tags
  const metaEl = document.getElementById('phase-meta');
  if(metaEl) {
    let tags = '';
    if(sess.phase_bpm > 0) tags += `<span class="phase-tag bpm">🥁 ${sess.phase_bpm} BPM</span>`;
    if(sess.phase_strumming) tags += `<span class="phase-tag strum">🎸 ${sess.phase_strumming}</span>`;
    if(sess.phase_chords?.length) tags += `<span class="phase-tag chord">🎵 ${sess.phase_chords.length} Akkorde</span>`;
    metaEl.innerHTML = tags;
  }

  // Akkord-Chips
  const chordsEl = document.getElementById('phase-chords');
  if(chordsEl) {
    const chs = sess.phase_chords || [];
    chordsEl.innerHTML = chs.map(c=>
      `<span class="chord-chip" onclick="selectChordFromPhase('${c}')">${c}</span>`
    ).join('');
  }

  // Phasen-Fortschritt
  const ppf = sess.phase_progress ?? 0;
  const ppEl = document.getElementById('phase-prog-fill');
  const ppVl = document.getElementById('phase-prog-val');
  if(ppEl) ppEl.style.width = (ppf*100).toFixed(1)+'%';
  if(ppVl) ppVl.textContent = Math.round(ppf*100)+'%';

  // Lektion gesamt
  setBarW('lesson-prog-bar','lesson-prog-val', (sess.lesson_progress ?? ts.lesson_prog ?? 0)*100);

  // Weiter-Button
  const btn = document.getElementById('btn-phase-next');
  if(btn) {
    const phId = sess.phase_id || 'technique';
    btn.className = 'btn-phase-next ' + (phId==='song'?'song':phId==='xp'?'xp':'');
    const labels = {
      technique: '💪 Weiter zur Übung →',
      exercise:  '🎵 Weiter zum Song →',
      song:      '🏆 XP sammeln →',
      xp:        '📚 Nächste Lektion →',
    };
    btn.textContent = labels[phId] || 'Weiter →';
  }

  // Level-Up Overlay
  if(sess.level_up && sess.complete) {
    showLevelUp(sess.profile_level);
  }

  // Luna sprechen lassen beim Phasenwechsel
  if(sess.phase_speech && window._lastPhaseIdx !== phIdx) {
    window._lastPhaseIdx = phIdx;
    if(window.lunaSay) lunaSay(sess.phase_speech);
  }

  // Header aktualisieren
  const hlEl = document.getElementById('hdr-lesson');
  if(hlEl && sess.lesson_title) hlEl.textContent = '📚 '+(sess.lesson_title||'').slice(0,18);
}

async function advancePhase() {
  try {
    const r = await fetch('/advance_phase', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
    const d = await r.json();
    if(d.session) updatePedSystem({session: d.session});
    if(d.done) {
      // Lektion fertig → Glückwunsch
      if(window.CSS_AVATAR) CSS_AVATAR.celebrate();
      if(window.AVATAR3D)   AVATAR3D.celebrate();
    }
  } catch(e) { console.warn('advancePhase:', e); }
}

function selectChordFromPhase(chord) {
  const instr = STATE?.profile?.instrument || 'guitar';
  fetch('/select_chord', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key: chord})});
}

function showLevelUp(level) {
  const ov = document.getElementById('levelup-overlay');
  const msg = document.getElementById('levelup-msg');
  if(ov) { ov.className='levelup-overlay show'; ov.style.display='flex'; }
  if(msg) msg.textContent = 'Du hast Level '+level+' erreicht! 🎉';

  // Ab Level 3: Spanish/Latin Flair freischalten
  if(lv >= 3) {
    const tab = document.getElementById('tab-spanish');
    if(tab && tab.style.display === 'none') {
      tab.style.display = '';
      setTimeout(()=>{
        lunaSay('Du hast Level 3 erreicht! Jetzt sind Flamenco und Latin Flair-Lektionen freigeschaltet – schau mal in den Flair-Tab!', 'proud');
        addChat('NoteIQ', '🌹 Neu: Spanish & Latin Flair-Lektionen freigeschaltet! Klick auf den 🌹 Flair-Tab.');
      }, 1800);
    }
  }
}

function closeLevelUp() {
  const ov = document.getElementById('levelup-overlay');
  if(ov) ov.style.display = 'none';
}

window._lastPhaseIdx = -1;


function renderLessonList(curriculum) {
  const el=document.getElementById('lesson-list');
  if(!el||!curriculum) return;

  // Standard-Lektionen filtern (Spanish/Latin im Flair-Tab)
  const SPECIAL_CATS = new Set(['Spanish','Latin']);
  const standard = curriculum.filter(l => !SPECIAL_CATS.has(l.category));

  el.innerHTML = standard.map((les,i) => {
    const globalIdx = curriculum.indexOf(les);
    const isDone    = globalIdx < currentLessonIdx;
    const isActive  = globalIdx === currentLessonIdx;
    const isLocked  = les.unlocked === false;
    return `
    <div class="lesson-card ${isActive?'active':''} ${isDone?'done':''} ${isLocked?'locked':''}"
      onclick="${isLocked?`showFlairLocked(${les.unlock_level||les.level})`:`selectLesson(${globalIdx})`}"
      title="${isLocked?'🔒 Level '+les.level+' erforderlich':les.title}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div class="lesson-level">Lv.${les.level}</div>
        <span style="font-size:.58rem;color:var(--yellow);font-family:var(--mono)">
          +${les.xp_reward||100} XP
        </span>
      </div>
      <div class="lesson-title">
        ${isActive?'▶ ':''}${isLocked?'🔒 ':''}${les.title}
      </div>
      <div class="lesson-meta">${les.duration_min} Min · ${(les.chords||[]).length} Akkorde</div>
    </div>`;
  }).join('');
}

function updateFeedback(fbs) {
  const area=document.getElementById('feedback-area');
  if(!area) return;
  const texts=fbs.map(m=>m.text);
  if(JSON.stringify(texts)===JSON.stringify(prevFeedback)) return;
  prevFeedback=texts;
  area.innerHTML='';
  fbs.slice(-3).forEach(m=>{
    const d=document.createElement('div');
    d.className='feedback-msg'+(m.bold?' bold':m.text.includes('!')||m.text.includes('OK')?' ok':m.text.includes('Warn')||m.text.includes('!?')?' warn':'');
    d.textContent=m.text;area.appendChild(d);
    setTimeout(()=>d.remove(),3500);
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  LUNA KOMMENTAR-LOGIK (regelbasiert + AI)
// ════════════════════════════════════════════════════════════════════════════
const LUNA_COMMENTS = {
  de: {
    good_acc:     ["Wunderschön gespielt!", "Das klingt fantastisch!", "Perfekt – weiter so!", "Deine Finger werden immer geschickter!"],
    low_acc:      ["Druck die Fingerkuppe etwas fester auf die Saite.", "Versuch die Finger näher am Bund zu platzieren.", "Nimm dir Zeit – Qualität kommt vor Schnelligkeit."],
    bad_posture:  ["Dein Handgelenk ist zu hoch! Senk es bitte ab.", "Entspann deinen Arm – Verkrampfung kostet Klang.", "Daumen hinter den Hals, nicht drüber!"],
    good_rhythm:  ["Dein Rhythmus ist super!", "Ich höre einen schönen gleichmäßigen Beat!", "Genau im Takt – hervorragend!"],
    bad_rhythm:   ["Hör auf den Metronom – er hilft dir mit dem Timing.", "Etwas langsamer – erst Genauigkeit, dann Tempo.", "Zähle im Kopf: eins-zwei-drei-vier!"],
    muted_string: ["Eine Saite klingt gedämpft. Prüfe ob ein Finger sie berührt.", "Finger aufrecht halten damit die Nachbarsaiten frei schwingen.", "Schau genau welche Saite stumm bleibt."],
    onset:        ["Gut angeschlagen!", "Schöner fester Anschlag!"],
    idle:         ["Bereit wenn du es bist!", "Nimm dir kurz Zeit – dann weitermachen.", "Du machst das wirklich gut!", "Gitarre spielen ist eine Reise – genieß jeden Schritt!"],
    flamenco_good:["¡Olé! Das klingt nach echtem Flamenco!", "Die phrygische Kadenz sitzt perfekt!", "Du spielst jetzt wie ein Andalusier!"],
    latin_good:   ["¡Muy bien! Das klingt nach Bossa Nova!", "Der Clave-Rhythmus klingt schon ganz kubanisch!", "Das ist reines Latin-Feeling!"],
    flamenco_tip: ["Rasgueado: kleiner Finger zuerst, dann fächern!", "E-Phrygisch ist das Herzstück des Flamenco.", "Den Compás fühlen – nicht nur zählen!"],
    latin_tip:    ["Daumen für den Bass unabhängig üben!", "Bossa Nova ist Fingesse, nicht Kraft.", "Die Clave gibt deiner Begleitung den Groove!"],
  },
  en: {
    good_acc:     ["Beautiful playing!", "That sounds fantastic!", "Perfect – keep it up!", "Your fingers are getting more skilled!"],
    low_acc:      ["Press your fingertip a bit harder.", "Try placing your finger closer to the fret.", "Take your time – quality before speed."],
    bad_posture:  ["Your wrist is too high! Please lower it.", "Relax your arm – tension costs tone.", "Thumb behind the neck, not over it!"],
    good_rhythm:  ["Your rhythm is great!", "I hear a nice steady beat!", "Right in time – excellent!"],
    bad_rhythm:   ["Listen to the metronome.", "Slower first – accuracy then speed.", "Count in your head: one-two-three-four!"],
    muted_string: ["A string sounds muted. Check if a finger is touching it.", "Keep fingers upright so neighboring strings ring.", "See which string stays silent."],
    onset:        ["Good strum!", "Nice strong attack!"],
    idle:         ["Ready when you are!", "Take a moment – then continue.", "You're really doing well!", "Playing guitar is a journey – enjoy every step!"],
    flamenco_good:["¡Olé! That sounds like real Flamenco!", "The Phrygian cadence is perfect!", "You're playing like an Andalusian now!"],
    latin_good:   ["¡Muy bien! That sounds like Bossa Nova!", "The clave rhythm is sounding Cuban already!", "That's pure Latin feeling!"],
    flamenco_tip: ["Rasgueado: little finger first, then fan!", "E-Phrygian is the heart of Flamenco.", "Feel the Compás – don't just count!"],
    latin_tip:    ["Practice thumb bass independently!", "Bossa Nova is finesse, not force.", "The clave gives your playing that groove!"],
  },
};

function getLunaLang() {
  return (STATE?.lang||'de').startsWith('en')?'en':'de';
}

function scheduleLunaComment(s) {
  const lang=getLunaLang();
  const cmts=LUNA_COMMENTS[lang]||LUNA_COMMENTS.de;
  const au=s.audio||{};
  const poly=au.poly||{};
  const strum=au.strum||{};
  const ch=s.chord||{};
  const posture=s.posture||{};
  let chosen=null, emotion='explaining';
  const acc=ch.acc||0;
  const clean=poly.chord_score||0;
  const bpm=strum.measured_bpm||0;
  if(!posture.wrist_ok&&Math.random()<0.7) { chosen=rndOf(cmts.bad_posture); emotion='correcting'; if(window.LUNA_API) window.LUNA_API.correct(); }
  else if(poly.muted_strings&&poly.muted_strings.length>0&&Math.random()<0.5) { chosen=rndOf(cmts.muted_string); emotion='correcting'; }
  else if(acc>0.85&&Math.random()<0.5) { chosen=rndOf(cmts.good_acc); emotion='happy'; if(window.LUNA_API) setTimeout(()=>window.LUNA_API.celebrate(),200); }
  else if(acc<0.3&&acc>0.05&&Math.random()<0.6) { chosen=rndOf(cmts.low_acc); emotion='correcting'; }
  else { chosen=rndOf(cmts.idle); emotion='neutral'; }
  if(chosen) { lunaSay(chosen, emotion); addChat('Luna',chosen); }
  lunaCommentInterval=8000+Math.random()*15000;
  // Flair: Spanish/Latin Kontext-Kommentare
  const curLesson = LessonData?.[currentLessonIdx];
  const curCat    = curLesson?.category;
  if(curCat === 'Spanish' || curCat === 'Latin') {
    const lang_  = speechLang.startsWith('de') ? 'de' : 'en';
    const cats   = curCat === 'Spanish' ? 'flamenco' : 'latin';
    const pool   = (Math.random() > 0.5)
      ? (LUNA_COMMENTS[lang_]?.[cats+'_good'] || [])
      : (LUNA_COMMENTS[lang_]?.[cats+'_tip']  || []);
    if(pool.length) {
      const chosen = pool[Math.floor(Math.random()*pool.length)];
      lunaSay(chosen, curCat==='Spanish'?'proud':'explaining');
      addChat('NoteIQ', chosen);
      return;
    }
  }

}

function rndOf(arr) { return arr[Math.floor(Math.random()*arr.length)]; }

// ════════════════════════════════════════════════════════════════════════════
//  CHAT & SESSION FEEDBACK
// ════════════════════════════════════════════════════════════════════════════
const chatHistory=[];
function addChat(who, text) {
  const hist=document.getElementById('chat-history');
  if(!hist) return;
  const key=who+':'+text.slice(0,30);
  if(chatHistory.includes(key)) return;
  chatHistory.push(key);
  if(chatHistory.length>60) chatHistory.shift();
  const d=document.createElement('div');
  d.className='chat-msg '+(who==='Luna'?'luna':'user');
  d.textContent=(who==='Luna'?'Luna: ':'')+text;
  hist.appendChild(d);
  hist.scrollTop=hist.scrollHeight;
  if(hist.children.length>30) hist.removeChild(hist.children[0]);
}

async function sendAIQuestion(q, emotion) {
  emotion = emotion || guessEmotion(q);
  if(window.LUNA_API){ window.LUNA_API.setEmotion('thinking'); window.LUNA_API.setSpeaking(false); }
  // Optimistisch: Luna zeigt Denkanimation
  addChat('Luna', '...');
  const lastMsg = document.querySelector('#chat-messages .chat-msg.luna:last-child');
  try {
    const ch  = STATE?.chord  || {};
    const au  = STATE?.audio  || {};
    const les = STATE?.lesson || {};
    const ctx = [
      `Instrument:${STATE?.profile?.instrument||'guitar'}`,
      `Lektion:"${les.title||'?'}"`,
      `Phase:${les.phase||1}`,
      `Akkord:${ch.key||'?'}`,
      `Erkannt:${ch.detected||'?'}`,
      `Trefferquote:${((ch.hold_pct||0)*100).toFixed(0)}%`,
      `Note:${au.note_name||'?'}`,
      `Hz:${(au.pitch_hz||0).toFixed(0)}`,
      `BPM:${(au.strum_bpm||0).toFixed(0)}`,
      `Sauberkeit:${((au.cleanliness||1)*100).toFixed(0)}%`,
      `Schnarren:${au.buzz||false}`,
      `Obertöne:${((au.harm_ratio||0)*100).toFixed(0)}%`,
      `WristOK:${STATE?.posture?.wrist_ok!==false}`,
      `WristWinkel:${(STATE?.posture?.angle||0).toFixed(0)}°`,
      `XP:${STATE?.profile?.xp||0}`,
      `Level:${STATE?.profile?.level||1}`,
    ].join(', ');
    await fetch('/ask_ai',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q, ctx, emotion})});
    lunaCommentInterval = 3000;
    // "..." ersetzen – wird von Python/AI über lunaSay befüllt
    if(lastMsg && lastMsg.textContent === 'Luna: ...') lastMsg.remove();
  } catch(e) {
    if(lastMsg && lastMsg.textContent === 'Luna: ...') lastMsg.remove();
    addChat('Luna', 'Keine Verbindung zur NoteIQ-KI. Bitte API-Key prüfen.');
  }
  if(window.LUNA_API) window.LUNA_API.setEmotion('explaining');
}

// Rationale Emotion aus Frage-Text ableiten
function guessEmotion(q) {
  const lo = q.toLowerCase();
  if(lo.includes('warum') || lo.includes('why') || lo.includes('wie')  || lo.includes('how'))  return 'explaining';
  if(lo.includes('falsch')|| lo.includes('wrong')|| lo.includes('fehler')) return 'correcting';
  if(lo.includes('gut')   || lo.includes('good') || lo.includes('toll'))   return 'proud';
  if(lo.includes('hilf')  || lo.includes('help') || lo.includes('tipp'))   return 'explaining';
  return 'neutral';
}

async function sendChat() {
  const inp = document.getElementById('chat-input');
  if(!inp || !inp.value.trim()) return;
  const q = inp.value.trim(); inp.value = '';
  addChat('Du', q);
  await sendAIQuestion(q);
}

function requestFeedback() {
  const sfb=document.getElementById('session-fb');
  if(!sfb) return;
  sfb.style.display='block';
  const accs=sessionStats.acc;
  const postArr=sessionStats.posture;
  const polyArr=sessionStats.poly;
  const avgAcc=accs.length?Math.round(accs.reduce((a,b)=>a+b,0)/accs.length*100):0;
  const avgPost=postArr.length?Math.round(postArr.reduce((a,b)=>a+b,0)/postArr.length*100):100;
  const avgPoly=polyArr.length?Math.round(polyArr.reduce((a,b)=>a+b,0)/polyArr.length*100):0;
  const grade=avgAcc>80&&avgPost>75&&avgPoly>70?'A':(avgAcc>60?'B':(avgAcc>40?'C':'D'));
  const gradeColors={A:'var(--success)',B:'#AADD00',C:'#FFAA00',D:'var(--danger)'};
  const gradeEl=document.getElementById('sfb-grade');
  if(gradeEl){gradeEl.textContent=grade;gradeEl.style.color=gradeColors[grade]||'#fff';}
  const mins=Math.round((Date.now()-sessionStats.startTime)/60000);
  const lang=getLunaLang();
  const feedback=lang==='en'?{
    acc:{label:'Chord Accuracy',val:avgAcc+'%',note:avgAcc>70?'Great chord gripping!':'Focus on pressing firmly with fingertips.'},
    posture:{label:'Posture',val:avgPost+'%',note:avgPost>80?'Excellent posture throughout!':'Keep wrist low and thumb behind neck.'},
    clean:{label:'Sound Cleanliness',val:avgPoly+'%',note:avgPoly>65?'Clean sound, all strings ringing!':'Some strings muted – check finger position.'},
    time:{label:'Session Duration',val:mins+' min',note:'Great dedication!'},
  }:{
    acc:{label:'Akkord-Genauigkeit',val:avgAcc+'%',note:avgAcc>70?'Toller Griff!':'Drücke mit der Fingerkuppe fester auf.'},
    posture:{label:'Haltung',val:avgPost+'%',note:avgPost>80?'Ausgezeichnete Haltung!':'Handgelenk tief halten, Daumen hinter den Hals.'},
    clean:{label:'Klang-Sauberkeit',val:avgPoly+'%',note:avgPoly>65?'Sauberer Klang!':'Einige Saiten gedämpft – Fingerposition prüfen.'},
    time:{label:'Session-Dauer',val:mins+' Min',note:'Toll – dranbleiben!'},
  };
  const content=document.getElementById('sfb-content');
  if(content){
    content.innerHTML=Object.values(feedback).map(f=>`
      <div class="fb-section">
        <h4>${f.label}: ${f.val}</h4>
        <p>${f.note}</p>
      </div>`).join('');
  }
  if(window.LUNA_API) window.LUNA_API.celebrate();
  const lunaFeedText=lang==='en'?`Your grade is ${grade}! ${feedback.acc.note} ${feedback.posture.note}`
    :`Deine Note ist ${grade}! ${feedback.acc.note} ${feedback.posture.note}`;
  lunaSay(lunaFeedText,'proud');
}

function closeFeedback() { const el=document.getElementById('session-fb'); if(el) el.style.display='none'; }

// ════════════════════════════════════════════════════════════════════════════
//  CONTROLS
// ════════════════════════════════════════════════════════════════════════════
function selectChord(k) { fetch('/select_chord',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})}); }
function startSong(k) { fetch('/start_song',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})}); setLTab('song'); }
function stopSong() { fetch('/stop_song',{method:'POST'}); }
function selectLesson(idx) { currentLessonIdx=idx; fetch('/select_lesson',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({idx})}); }
function changeBPM(d) { fetch('/set_bpm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({delta:d})}); }
function toggleMetro() { fetch('/toggle_metro',{method:'POST'}); }
function startNewLesson() { fetch('/lesson_restart',{method:'POST'}); closeFeedback(); }

// ════════════════════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════════════
//  ONBOARDING WIZARD v16 + ERROR HANDLING
// ════════════════════════════════════════════════════════════════════════════

/* ── Onboarding State ─────────────────────────────────────────────────────── */
let obStep       = 0;
const OB_TOTAL   = 5;
let obMicGranted = false;
let obInitState  = null;   // /state Response beim Start
let obLangChoice = null;   // Sprachauwahl im Onboarding

/* ── Schritt navigieren ──────────────────────────────────────────────────── */
function obNext() {
  if(obStep === 1) { runSystemCheck(); }   // System-Check beim zweiten Mal
  if(obStep < OB_TOTAL - 1) {
    obGoTo(obStep + 1);
  } else {
    obFinish();
  }
}
function obBack() {
  if(obStep > 0) obGoTo(obStep - 1);
}
function obGoTo(n) {
  document.getElementById('ob-step-' + obStep)?.classList.remove('active');
  document.getElementById('ob-dot-'  + obStep)?.classList.replace('active','done');
  obStep = n;
  document.getElementById('ob-step-' + n)?.classList.add('active');
  document.getElementById('ob-dot-'  + n)?.classList.add('active');

  // Buttons
  const back = document.getElementById('ob-back-btn');
  const next = document.getElementById('ob-next-btn');
  const ctr  = document.getElementById('ob-step-counter');
  if(back) back.style.display = n > 0 ? '' : 'none';
  if(ctr)  ctr.textContent    = `${n+1} / ${OB_TOTAL}`;

  // Letzter Schritt: "Loslegen"
  if(next) {
    if(n === OB_TOTAL - 1) {
      next.textContent = '🎸 Loslegen!';
      next.style.background = 'var(--green)';
    } else {
      next.textContent = 'Weiter →';
      next.style.background = '';
    }
  }

  // Schritt-spezifische Aktionen
  if(n === 1) { setTimeout(runSystemCheck, 300); }
  if(n === 3) { obFillProfile(); }
}

/* ── System-Check ─────────────────────────────────────────────────────────── */
async function runSystemCheck() {
  const el = document.getElementById('ob-checks-list');
  if(!el) return;

  // Lade aktuellen State
  let st = null;
  try {
    const r = await fetch('/state');
    st = await r.json();
    obInitState = st;
  } catch(e) {
    el.innerHTML = obCheckHTML('fail','🔌','Server-Verbindung',
      'Konnte nicht auf http://localhost:'+window.location.port+' verbinden.',
      'Seite neu laden', 'location.reload()');
    return;
  }

  const sys = st?.system_status || {};
  const checks = [];

  // Audio / Mikrofon
  if(sys.audio_ok) {
    checks.push(obCheckHTML('ok','🎙️','Mikrofon (Server-seitig)',
      'Mikrofon aktiv – Ton-Analyse läuft'));
  } else if(sys.audio_errcode === 'NO_PYAUDIO') {
    checks.push(obCheckHTML('fail','🎙️','Mikrofon (PyAudio fehlt)',
      sys.audio_error || 'pip install pyaudio',
      'Anleitung', "showErrToast('error','Installation','pip install pyaudio',false)"));
  } else if(sys.audio_errcode === 'NO_MIC') {
    checks.push(obCheckHTML('warn','🎙️','Kein Mikrofon gefunden',
      'Schließe ein Mikrofon oder USB-Audio-Interface an.'));
  } else if(sys.audio_error) {
    checks.push(obCheckHTML('warn','🎙️','Audio-Warnung',
      sys.audio_error));
  } else {
    checks.push(obCheckHTML('warn','🎙️','Mikrofon (Simulation)',
      'Läuft im Simulations-Modus – kein echtes Mikrofon erkannt.'));
  }

  // Kamera (wird durch den Fakt dass wir State kriegen impliziert)
  if(sys.cam_ok !== false) {
    checks.push(obCheckHTML('ok','📹','Kamera','Kamera-Feed aktiv'));
  } else {
    checks.push(obCheckHTML('warn','📹','Kamera nicht verfügbar',
      'Keine Kamera gefunden – Hand-Tracking nicht möglich. Audio-Analyse läuft trotzdem.'));
  }

  // Hand-Tracking
  if(sys.tracker_ok) {
    checks.push(obCheckHTML('ok','🖐️','Hand-Tracking (MediaPipe)',
      sys.tracker_mode || 'MediaPipe Hand-Landmarker aktiv – präzise'));
  } else if(sys.tracker_error) {
    checks.push(obCheckHTML('warn','🖐️','Hand-Tracking (Fallback)',
      (sys.tracker_mode || 'Skin-Farb-Erkennung aktiv') +
      (sys.tracker_error ? ' – ' + sys.tracker_error : '')));
  }

  // KI / Groq
  if(sys.groq_ok) {
    checks.push(obCheckHTML('ok','🤖','Groq AI (llama-3.3-70b)',
      'Kostenlose KI-Antworten aktiv – ultra-schnell'));
  } else if(sys.ai_ok) {
    checks.push(obCheckHTML('ok','🤖','OpenAI (Fallback)',
      'OpenAI GPT-4o aktiv'));
  } else {
    checks.push(obCheckHTML('warn','🤖','KI offline',
      'Kein API-Key gesetzt. Setze GROQ_API_KEY (kostenlos: console.groq.com)',
      'Mehr Info', "showErrToast('warn','KI-Einrichtung','export GROQ_API_KEY=dein_key',false)"));
  }

  // TTS
  if(sys.tts_eleven) {
    checks.push(obCheckHTML('ok','🗣️','ElevenLabs TTS',
      'Natürliche Stimme aktiv – menschlich klingend'));
  } else if(sys.tts_openai) {
    checks.push(obCheckHTML('ok','🗣️','OpenAI TTS (nova)',
      'Hochwertige Stimme aktiv'));
  } else {
    checks.push(obCheckHTML('warn','🗣️','Browser-Stimme (Fallback)',
      'Keine Server-TTS. Browser Web Speech API wird verwendet.',
      'ElevenLabs einrichten', "window.open('https://elevenlabs.io','_blank')"));
  }

  // Browser-Kompatibilität
  const hasSpeech = !!window.SpeechSynthesis || !!window.speechSynthesis;
  const hasSR     = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  checks.push(obCheckHTML(
    hasSR ? 'ok' : 'warn',
    '🌐', 'Browser-Kompatibilität',
    (hasSR ? 'Sprach-Erkennung ✓' : 'Keine Sprach-Erkennung (Chrome/Edge empfohlen)') +
    ' · ' + (hasSpeech ? 'Sprach-Ausgabe ✓' : 'Keine Sprach-Ausgabe')
  ));

  el.innerHTML = checks.join('');
}

function obCheckHTML(status, icon, name, msg, fixLabel, fixAction) {
  const fix = fixLabel
    ? `<span class="ob-check-fix" onclick="${fixAction}">${fixLabel}</span>`
    : '';
  return `
  <div class="ob-check ${status}">
    <div class="ob-check-icon">${icon}</div>
    <div class="ob-check-label">
      <div class="ob-check-name">${name}</div>
      <div class="ob-check-msg">${msg}</div>
    </div>
    ${fix}
  </div>`;
}

/* ── Mikrofon-Erlaubnis anfordern ─────────────────────────────────────────── */
async function obRequestMic() {
  const check = document.getElementById('ob-mic-check');
  const msg   = document.getElementById('ob-mic-msg');
  const ctx   = document.getElementById('ob-audio-ctx-check');
  if(!check || !msg) return;

  check.className = 'ob-check checking';
  msg.textContent = 'Warte auf Browser-Erlaubnis…';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach(t => t.stop());  // Sofort wieder freigeben
    obMicGranted = true;
    check.className = 'ob-check ok';
    msg.textContent = '✓ Mikrofon-Zugriff erlaubt';
    check.querySelector('.ob-check-fix')?.remove();
    if(ctx) { ctx.style.display = ''; ctx.className = 'ob-check ok'; }

    // Weiter-Button aktivieren / hervorheben
    const nextBtn = document.getElementById('ob-next-btn');
    if(nextBtn) { nextBtn.textContent = 'Weiter →'; nextBtn.style.animation = 'wov-bounce .6s ease'; }

  } catch(err) {
    check.className = 'ob-check fail';
    if(err.name === 'NotAllowedError') {
      msg.textContent = 'Zugriff verweigert – bitte in Browser-Einstellungen erlauben';
    } else if(err.name === 'NotFoundError') {
      msg.textContent = 'Kein Mikrofon gefunden – bitte Gerät anschließen';
    } else {
      msg.textContent = 'Fehler: ' + err.message;
    }
    showErrToast('warn','Mikrofon-Zugriff verweigert',
      'Klick auf das 🔒-Symbol in der Adressleiste und erlaube Mikrofon-Zugriff.', true);
  }
}

/* ── Profil-Anzeige im Onboarding ──────────────────────────────────────────── */
function obFillProfile() {
  const el = document.getElementById('ob-profile-display');
  if(!el || !obInitState) return;
  const p = obInitState.profile || {};
  const t = obInitState.teacher || {};
  el.innerHTML = `
    <div class="ob-check ok">
      <div class="ob-check-icon">👤</div>
      <div class="ob-check-label">
        <div class="ob-check-name">${p.name || 'Spieler'}</div>
        <div class="ob-check-msg">Level ${p.level||1} · ${p.xp||0} XP · ${t.instrument||'guitar'==='guitar'?'🎸 Gitarre':'🎹 Klavier'}</div>
      </div>
    </div>
    <div class="ob-check ok">
      <div class="ob-check-icon">📚</div>
      <div class="ob-check-label">
        <div class="ob-check-name">Lektions-Fortschritt</div>
        <div class="ob-check-msg">Lektion ${(t.lesson_idx||0)+1} von ${(t.curriculum||[]).length} · ${Math.round((p.total_min||0))} Min gespielt</div>
      </div>
    </div>`;

  // Sprach-Buttons markieren basierend auf Server-Sprache
  const obLang = obInitState?.lang || 'de';
  obLangChoice = obLang;
  document.querySelectorAll('[data-lang]').forEach(b => {
    b.classList.toggle('selected', b.dataset.lang === obLang);
  });
}

/* ── Sprache setzen ───────────────────────────────────────────────────────── */
function obSetLang(lang) {
  obLangChoice = lang;
  document.querySelectorAll('[data-lang]').forEach(b => {
    b.classList.toggle('selected', b.dataset.lang === lang);
  });
  speechLang = lang === 'de' ? 'de-DE'
             : lang === 'es' ? 'es-ES'
             : lang === 'fr' ? 'fr-FR'
             : 'en-US';
  pickPreferredVoice();
}

/* ── Onboarding abschließen ──────────────────────────────────────────────── */
function obFinish() {
  // Onboarding-Overlay ausblenden
  const ov = document.getElementById('onboarding-overlay');
  if(ov) {
    ov.classList.add('hide');
    setTimeout(() => { ov.style.display = 'none'; }, 450);
  }

  // Einmalig in localStorage merken (nächster Start überspringt Onboarding)
  try { localStorage.setItem('niq_ob_done', '1'); } catch(e) {}

  // Audio-Unlock + Begrüßung
  _doUnlockAndGreet();

  // System-Status-Bar einblenden
  setTimeout(updateSysStatusBar, 800);

  // Fehlermeldungen aus dem System-Check anzeigen
  if(obInitState?.system_status) {
    const sys = obInitState.system_status;
    if(!sys.audio_ok && sys.audio_errcode !== 'NO_PYAUDIO') {
      showErrToast('warn', 'Mikrofon', sys.audio_error || 'Audio nicht verfügbar', true);
    }
    if(!sys.groq_ok && !sys.ai_ok) {
      showErrToast('warn', 'KI offline',
        'Setze GROQ_API_KEY für KI-Antworten. Kostenlos: console.groq.com', true);
    }
    if(!sys.tracker_ok && sys.tracker_error) {
      showErrToast('info', 'Hand-Tracking',
        sys.tracker_mode || 'Skin-Modus aktiv', true);
    }
  }

  // Polling starten
  setInterval(poll, 200);

  // Begrüßungstext
  const lang = obLangChoice || obInitState?.lang || 'de';
  const greetings = {
    de: 'Willkommen! Ich bin NoteIQ. Ich höre und sehe dein Spiel in Echtzeit. Stell mir gerne Fragen per Mikrofon oder im Chat. Los geht's!',
    en: 'Welcome! I am NoteIQ. I hear and see your playing in real time. Feel free to ask me questions via microphone or chat. Let's go!',
    es: '¡Bienvenido! Soy NoteIQ. Escucho y veo tu toque en tiempo real. ¡Pregúntame por micrófono o chat!',
    fr: 'Bienvenue ! Je suis NoteIQ. J'entends et vois votre jeu en temps réel. Posez-moi des questions par micro ou chat !',
  };
  _pendingGreeting = greetings[lang] || greetings.de;
}

/* ── Poll mit Verbindungs-Fehlerbehandlung ───────────────────────────────── */
let _pollErrCount   = 0;
let _pollErrToastId = null;
let _pollActive     = false;

async function poll() {
  if(_pollActive) return;   // Kein paralleler Request
  _pollActive = true;
  try {
    const r = await fetch('/state', { signal: AbortSignal.timeout(3000) });
    if(!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    STATE = s;

    // Verbindung wieder OK
    if(_pollErrCount > 0) {
      _pollErrCount = 0;
      removeErrToast(_pollErrToastId);
      _pollErrToastId = null;
      showErrToast('ok','Verbindung wiederhergestellt','Server antwortet wieder.', true);
    }

    updateUI(s);
    updateSysStatusBar(s.system_status);

  } catch(err) {
    _pollErrCount++;
    if(_pollErrCount === 3 && !_pollErrToastId) {
      _pollErrToastId = showErrToast('error','Verbindung unterbrochen',
        'Server antwortet nicht. Prüfe ob das Python-Skript noch läuft.', false);
    }
    // Nach 30 Fehlern: Reload vorschlagen
    if(_pollErrCount === 30) {
      showErrToast('error','Server nicht erreichbar',
        'Das Programm scheint beendet. <a href="/" style="color:var(--accent3)">Seite neu laden</a>',
        false);
    }
  } finally {
    _pollActive = false;
  }
}

/* ── Error-Toasts ─────────────────────────────────────────────────────────── */
let _toastIdCounter = 0;
function showErrToast(type, title, msg, autoClose) {
  const id  = 'et-' + (++_toastIdCounter);
  const con = document.getElementById('err-toast-container');
  if(!con) return id;

  const icons = { error:'❌', warn:'⚠️', ok:'✅', info:'ℹ️' };
  const cls   = type === 'warn' ? 'err-toast warn-toast' : 'err-toast';

  const div = document.createElement('div');
  div.id        = id;
  div.className = cls;
  div.innerHTML = `
    <div class="err-toast-icon">${icons[type]||'ℹ️'}</div>
    <div class="err-toast-body">
      <div class="err-toast-title">${title}</div>
      <div class="err-toast-msg">${msg}</div>
    </div>
    <div class="err-toast-close" onclick="removeErrToast('${id}')">×</div>`;
  con.appendChild(div);

  if(autoClose) setTimeout(() => removeErrToast(id), type === 'ok' ? 3000 : 8000);
  return id;
}

function removeErrToast(id) {
  const el = document.getElementById(id);
  if(el) { el.style.opacity = '0'; el.style.transform = 'translateX(20px)'; setTimeout(()=>el.remove(), 300); }
}

/* ── Onboarding erneut anzeigen (Hilfe-Button) ──────────────────────────── */
// Tastatur: '?' öffnet Onboarding/Hilfe
document.addEventListener('keydown', function(e) {
  if(e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
    const chatInput = document.getElementById('chat-input');
    if(document.activeElement === chatInput) return; // nicht im Chat-Feld
    showOnboarding();
  }
});

function showOnboarding() {
  const ov = document.getElementById('onboarding-overlay');
  if(!ov) return;
  ov.style.display = 'flex';
  ov.classList.remove('hide');
  ov.style.opacity = '1';
  // Direkt zur Bedienungsanleitung (Schritt 4)
  obStep = 3;
  obGoTo(4);
  // Schließen-Button
  const next = document.getElementById('ob-next-btn');
  if(next) {
    next.textContent = '✓ Schließen';
    const origOnclick = next.onclick;
    next.onclick = () => {
      ov.classList.add('hide');
      setTimeout(() => { ov.style.display = 'none'; }, 450);
      next.textContent = '🎸 Loslegen!';
      next.onclick = obNext;
    };
  }
}

/* ── System-Status-Bar (untere Leiste) ────────────────────────────────────── */
function updateSysStatusBar(sys) {
  const bar = document.getElementById('sys-status-bar');
  if(!bar) return;
  sys = sys || STATE?.system_status;
  if(!sys) return;

  bar.classList.add('visible');
  const dot = (ok, warn) =>
    `<span class="sys-dot ${ok?'ok':warn?'warn':'fail'}"></span>`;

  bar.innerHTML = [
    dot(sys.audio_ok)+'Mikrofon',
    dot(sys.cam_ok !== false)+'Kamera',
    dot(sys.tracker_ok, sys.tracker_error)+'Hand-Tracking',
    dot(sys.groq_ok || sys.ai_ok, true)+'KI',
    dot(sys.tts_eleven || sys.tts_openai, true)+'Stimme',
    `<span style="margin-left:auto;color:var(--border2)">NoteIQ v16.0</span>`,
  ].join('<span style="color:var(--border2);margin:0 6px">·</span>');
}

/* ── Overlay-Schließen (Alt-Kompatibilität) ──────────────────────────────── */
function _closeOverlay() {
  const ov = document.getElementById('wov');
  if(ov) { ov.style.opacity='0'; setTimeout(()=>{ ov.style.display='none'; },480); }
}

/* ══════════════════════════════════════════════════════════════════════════
   INIT: Onboarding starten oder überspringen
══════════════════════════════════════════════════════════════════════════ */
(async () => {
  // State laden
  let initS = null;
  try {
    const r = await fetch('/state');
    initS   = await r.json();
  } catch(e) {
    // Server noch nicht bereit – kurz warten und nochmal
    await new Promise(res => setTimeout(res, 800));
    try {
      const r = await fetch('/state');
      initS   = await r.json();
    } catch(e2) {
      showErrToast('error','Server nicht erreichbar',
        'Konnte keine Verbindung zu NoteIQ herstellen. Ist das Python-Skript gestartet?', false);
    }
  }

  if(initS) {
    STATE          = initS;
    obInitState    = initS;
    LessonData     = initS.teacher?.curriculum;
    SpecialLessons = initS.teacher?.special_lessons || [];
    updateSpanishTabVisibility(initS.teacher?.profile_level || 1);
    const lang = initS.lang || 'de';
    speechLang = lang==='de'?'de-DE':lang==='es'?'es-ES':lang==='fr'?'fr-FR':'en-US';
    pickPreferredVoice();
    renderLessonList(LessonData);
  }

  // Onboarding überspringen wenn schon durchgeführt (oder Skip-Flag)
  const skipOb = localStorage.getItem('niq_ob_done') === '1';
  const urlSkip = new URLSearchParams(window.location.search).get('skip_ob');

  if(skipOb || urlSkip) {
    // Direktstart
    const ov = document.getElementById('onboarding-overlay');
    if(ov) ov.style.display = 'none';
    _pendingGreeting = (initS?.lang||'de') === 'de'
      ? 'Hallo! Ich bin NoteIQ, bereit zum Spielen!'
      : 'Hello! NoteIQ is ready to play!';
    setInterval(poll, 200);
    setTimeout(()=>updateSysStatusBar(initS?.system_status), 500);
    // Direkt-Audio-Unlock beim ersten Klick
    document.addEventListener('click', ()=>_doUnlockAndGreet(), { once:true });
  } else {
    // Onboarding zeigen – erster Schritt
    obGoTo(0);
    // Keine setInterval hier – wird in obFinish() gestartet
  }
})();
</script>
</body>
</html>"""


# ══ NEU v6: TEACHER SYSTEM + HTTP HANDLER ══════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  TEACHER STATE MANAGER  (v6 neu)
# ══════════════════════════════════════════════════════════════════════════════
class TeacherSystem:
    """
    v11: Integriert LessonSession (Technique→Exercise→Song→XP).
    Steuert 3D-Lehrer Luna: Emotionen, Phasen-Fortschritt, XP-System.
    """
    def __init__(self, profile, lang="de"):
        self.profile    = profile
        self.lang       = lang
        self.instrument = profile.instrument
        self.curriculum_list = LESSON_CURRICULUM.get(self.instrument, [])
        self.lesson_idx : int   = profile.lesson_idx
        self.lesson_prog: float = 0.0
        self.emotion    : str   = "neutral"
        self.arm_pose   : str   = "rest"
        self.speaking   : bool  = False
        self._last_emotion_t    = 0.
        self._good_streak       = 0
        self._bad_streak        = 0
        self._lesson_acc_buf    : collections.deque = collections.deque(maxlen=30)
        self._posture_buf       : collections.deque = collections.deque(maxlen=20)
        self._poly_buf          : collections.deque = collections.deque(maxlen=20)
        self._lesson_start      = time.time()
        self._lesson_done       = False
        # v11: CurriculumManager + LessonSession
        self._cm   = CurriculumManager()
        self._pm   = None   # wird von InstructorV6 gesetzt
        self._sess : Optional[LessonSession] = None
        self._start_session(self.lesson_idx)

    # Alias für Kompatibilität
    @property
    def curriculum(self): return self.curriculum_list

    def _start_session(self, idx: int):
        les_data = self._cm.get_lesson(self.instrument, idx)
        if les_data:
            # Fallback: minimaler ProgressMgr wenn _pm noch nicht gesetzt
            pm = self._pm or ProgressMgr()
            self._sess = LessonSession(les_data, self.profile, self._cm, pm, self.lang)

    def switch_instrument(self, instr):
        self.instrument = instr
        self.curriculum_list = LESSON_CURRICULUM.get(instr, [])
        self.lesson_idx = 0; self.lesson_prog = 0.0
        self._start_session(0)

    def select_lesson(self, idx):
        self.lesson_idx = max(0, min(idx, len(self.curriculum_list)-1))
        self.lesson_prog = 0.0
        self._lesson_start = time.time()
        self._lesson_done  = False
        self._lesson_acc_buf.clear()
        self._start_session(self.lesson_idx)

    def advance_phase(self) -> bool:
        """Manuell nächste Phase aufrufen. Returns True wenn Lektion fertig."""
        if self._sess:
            done = self._sess.advance_phase()
            if done:
                self.lesson_idx = self.profile.lesson_idx
                self._start_session(self.lesson_idx)
            return done
        return False

    @property
    def cur_lesson(self) -> Dict:
        if not self.curriculum_list: return {}
        return self.curriculum_list[min(self.lesson_idx, len(self.curriculum_list)-1)]

    def update(self, acc: float, wrist_ok: bool, poly_score: float,
               strum_bpm: float, target_bpm: float, onset: bool) -> Dict:
        now = time.time()
        self._lesson_acc_buf.append(acc)
        self._posture_buf.append(1.0 if wrist_ok else 0.0)
        self._poly_buf.append(poly_score)
        avg_acc  = float(np.mean(self._lesson_acc_buf)) if self._lesson_acc_buf else 0.
        avg_post = float(np.mean(self._posture_buf))    if self._posture_buf    else 1.
        avg_poly = float(np.mean(self._poly_buf))       if self._poly_buf       else 0.

        # Streak tracking
        if acc > 0.7: self._good_streak += 1; self._bad_streak = 0
        else:         self._bad_streak  += 1; self._good_streak = 0

        # v11: LessonSession Fortschritt
        if self._sess:
            self._sess.record_acc(acc)
            self.lesson_prog = self._sess.lesson_progress
        else:
            # Fallback: zeitbasiert
            les = self.cur_lesson
            dur = les.get("duration_min", 10) * 60
            elapsed = now - self._lesson_start
            time_prog = min(1., elapsed / dur)
            acc_prog  = min(1., avg_acc * 1.2)
            self.lesson_prog = min(1., (time_prog * 0.4 + acc_prog * 0.6))

        # Emotion bestimmen
        if now - self._last_emotion_t > 3.0:
            self._last_emotion_t = now
            if not wrist_ok:
                self.emotion = "correcting"; self.arm_pose = "pointing_right"
            elif self._good_streak > 20 and acc > 0.85:
                self.emotion = "proud";      self.arm_pose = "clap"
            elif avg_acc > 0.7 and avg_poly > 0.6:
                self.emotion = "happy";      self.arm_pose = "rest"
            elif self._bad_streak > 15:
                self.emotion = "explaining"; self.arm_pose = "pointing_right"
            else:
                self.emotion = "neutral";    self.arm_pose = "rest"

        return self.get_state()

    def get_state(self) -> Dict:
        les = self.cur_lesson
        topics = les.get("topics", [])
        speech_intro = les.get("speech",{}).get(self.lang, les.get("speech",{}).get("de",""))

        # v11: Session-State einbinden
        sess_st = self._sess.to_state() if self._sess else {}

        return {
            "lesson_idx":      self.lesson_idx,
            "lesson_title":    les.get("title",""),
            "lesson_prog":     round(self.lesson_prog, 3),
            "lesson_topics":   topics,
            "lesson_chords":   les.get("chords",[]),
            "lesson_speech":   speech_intro,
            "emotion":         self.emotion,
            "arm_pose":        self.arm_pose,
            "speaking":        self.speaking,
            # v11 Pädagogik
            "session":         sess_st,
            "curriculum":      [{"id":l["id"],"title":l["title"],"level":l["level"],
                                  "duration_min":l["duration_min"],"chords":l["chords"],
                                  "xp_reward":l.get("xp_reward",100),
                                  "category":l.get("category","Standard"),
                                  "unlock_level":l.get("unlock_level",l["level"]),
                                  "unlocked":(self.profile.level>=l.get("unlock_level",l["level"]))}
                                 for l in self.curriculum_list],
            "special_lessons": [{"id":l["id"],"title":l["title"],"level":l["level"],
                                   "duration_min":l["duration_min"],"chords":l["chords"],
                                   "xp_reward":l.get("xp_reward",100),
                                   "category":l.get("category",""),
                                   "unlock_level":l.get("unlock_level",l["level"]),
                                   "unlocked":(self.profile.level>=l.get("unlock_level",l["level"])),
                                   "idx":i}
                                  for i,l in enumerate(self.curriculum_list)
                                  if l.get("category") in {"Spanish","Latin"}],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════════
_instructor = None
_latest_jpg = None
_frame_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            html = _build_html(_instructor.state if _instructor else {})
            self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))
        elif p == "/state":
            s = json.dumps(_instructor.state if _instructor else {},
                           ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json", s)
        elif p == "/stream":
            self._mjpeg_stream()
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def _mjpeg_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    jpg = _latest_jpg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        def ok(): self._send(200, "application/json", b'{"ok":true}')

        if p == "/select_chord" and _instructor:
            try:
                data = json.loads(body); k = data.get("key","")
                instr = _instructor.profile.instrument
                if k in CHORDS[instr]:
                    _instructor.trainer.set_chord(k)
                    _instructor.fb.push(f"Akkord: {k}", OR["orange"])
            except: pass
            ok()
        elif p == "/start_song" and _instructor:
            try:
                data = json.loads(body); k = data.get("key","")
                _instructor.start_song(k)
            except: pass
            ok()
        elif p == "/stop_song" and _instructor:
            _instructor.song.stop()
            _instructor.fb.push("Song beendet", OR["muted"])
            ok()
        elif p == "/tut_next" and _instructor:
            _instructor.tut.next_step(); ok()
        elif p == "/tut_dismiss" and _instructor:
            _instructor.tut.dismiss(); ok()
        elif p == "/ask_ai" and _instructor:
            try:
                data = json.loads(body)
                q    = data.get("q","Tipp?")
                ctx  = data.get("ctx","")
                if not ctx:
                    au  = _instructor.audio
                    ctx = (
                        f"Instrument:{_instructor.profile.instrument},"
                        f"Akkord:{_instructor.trainer.cur_key},"
                        f"Note:{au.note_name},Hz:{au.pitch_hz:.1f},"
                        f"Poly:{au.poly.poly_notes},"
                        f"Strum:{au.strum.measured_bpm:.0f}BPM,"
                        f"Sauberkeit:{au.cleanliness:.2f},"
                        f"Schnarren:{au.buzz_detected},"
                        f"Obertöne:{au.harmonic_ratio:.2f},"
                        f"WristOK:{_instructor._wrist_ok},"
                        f"WristWinkel:{_instructor._wrist_ang:.1f}°,"
                        f"AlignOK:{getattr(_instructor,'_align_ok',True)},"
                        f"XP:{_instructor.profile.xp},Level:{_instructor.profile.level},"
                        f"Lektion-Kategorie:{_instructor.trainer.cur_chord.get('category','Standard')}"
                    )
                emotion = data.get("emotion", "explaining")
                _instructor.ai.ask(q, ctx, emotion)
            except: pass
            ok()
        elif p == "/select_lesson" and _instructor:
            try:
                data = json.loads(body); idx = int(data.get("idx",0))
                _instructor.teacher.select_lesson(idx)
                les = _instructor.teacher.cur_lesson
                if les.get("chords"):
                    first = les["chords"][0]
                    instr = _instructor.profile.instrument
                    if first in CHORDS[instr]: _instructor.trainer.set_chord(first)
                _instructor.fb.push(f"Lektion: {les.get('title','')}", OR["song"])
            except: pass
            ok()
        elif p == "/set_bpm" and _instructor:
            try:
                data = json.loads(body); d = int(data.get("delta",0))
                _instructor.metro.set(_instructor.metro.bpm + d)
                _instructor.fb.push(f"BPM: {_instructor.metro.bpm}", OR["muted"])
            except: pass
            ok()
        elif p == "/toggle_metro" and _instructor:
            _instructor.metro.on = not _instructor.metro.on; ok()
        elif p == "/lesson_restart" and _instructor:
            _instructor.teacher.select_lesson(_instructor.teacher.lesson_idx); ok()
        elif p == "/advance_phase" and _instructor:
            done = _instructor.teacher.advance_phase()
            les = _instructor.teacher.cur_lesson
            if les.get("chords"):
                ph_chords = (_instructor.teacher._sess.current_phase.get("chords",[])
                             if _instructor.teacher._sess else les["chords"])
                instr = _instructor.profile.instrument
                target = ph_chords[0] if ph_chords else (les["chords"][0] if les["chords"] else "")
                if target and target in CHORDS.get(instr,{}):
                    _instructor.trainer.set_chord(target)
            phase_label = (_instructor.teacher._sess.current_phase.get("label","")
                           if _instructor.teacher._sess else "")
            _instructor.fb.push(f"Phase: {phase_label}", OR["song"])
            if done:
                _instructor.fb.push("Lektion abgeschlossen! XP erhalten!", OR["green"])
            self._send(200,"application/json",
                       json.dumps({"ok":True,"done":done,
                                   "session":(_instructor.teacher._sess.to_state()
                                              if _instructor.teacher._sess else {})},
                                  ensure_ascii=False).encode())
        elif p == "/lesson_state" and _instructor:
            sess_st = (_instructor.teacher._sess.to_state()
                       if _instructor.teacher._sess else {})
            self._send(200,"application/json",
                       json.dumps(sess_st,ensure_ascii=False).encode())
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def _send(self, code, ct, data):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(data))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser hat Verbindung getrennt
        except Exception as e:
            print(f"[HTTP] Sende-Fehler: {e}")


def start_server():
    import socket as _sock
    # Prüfen ob Port frei ist
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", WEB_PORT))
        except OSError:
            print(f"[WebUI] ⚠ Port {WEB_PORT} belegt – versuche {WEB_PORT+1}")
            global WEB_PORT
            WEB_PORT += 1
    try:
        srv = HTTPServer(("0.0.0.0", WEB_PORT), Handler)
        srv.socket.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[WebUI] ✓ http://localhost:{WEB_PORT}")
    except Exception as e:
        print(f"[WebUI] ✗ Server-Fehler: {e}")
        raise

# ══ NEU v6: INSTRUCTOR V6 + MAIN ════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUCTOR v6  (erweitert v5 um TeacherSystem)
# ══════════════════════════════════════════════════════════════════════════════
class InstructorV6(Instructor):
    """
    Erweitert den v5 Instructor um:
    - TeacherSystem (Luna's Lektion-Logik)
    - Erweiterte _update_state mit teacher-Daten
    - Lektion-gesteuerte Akkord-Auswahl
    - Session-Feedback
    """
    def __init__(self, profile, lang="de"):
        super().__init__(profile, lang)
        self.teacher = TeacherSystem(profile, lang)
        # v11: ProgressMgr an TeacherSystem übergeben für LessonSession
        self.teacher._pm = self.progress
        self.teacher._start_session(self.teacher.lesson_idx)
        # Setze ersten Akkord aus Curriculum
        self._apply_lesson_chord()

    def _apply_lesson_chord(self):
        les = self.teacher.cur_lesson
        chords = les.get("chords", [])
        instr  = self.profile.instrument
        if chords and chords[0] in CHORDS[instr]:
            self.trainer.set_chord(chords[0])

    def switch_instr(self, instr):
        super().switch_instr(instr)
        self.teacher.switch_instrument(instr)
        self._apply_lesson_chord()

    def process(self, frame):
        frame = super().process(frame)
        return frame

    def _update_state(self, acc):
        # Erst v5 state
        super()._update_state(acc)
        # Teacher update
        strum_bpm   = self.audio.strum.measured_bpm
        poly_score  = self.audio.poly.chord_score
        wrist_ok    = self._wrist_ok
        teacher_st  = self.teacher.update(
            acc         = acc,
            wrist_ok    = wrist_ok,
            poly_score  = poly_score,
            strum_bpm   = strum_bpm,
            target_bpm  = float(self.metro.bpm),
            onset       = self.audio.onset,
        )
        self.state["teacher"] = teacher_st


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL SETUP v6
# ══════════════════════════════════════════════════════════════════════════════
def terminal_setup_v6(pm):
    print("\n╔═══════════════════════════════════════════════════════════════════════╗")
    print("║  🎸  NoteIQ v16.0  ·  KI-Musiklehrer  ·  Spanish / Latin Flair  🎹  ║")
    print("║  Groq AI · ElevenLabs TTS · Hand-Analyse · Onboarding · Error-Check  ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝\n")
    names = list(pm.profiles.keys())
    if names:
        print("Vorhandene Profile:")
        for i, n in enumerate(names, 1):
            p = pm.profiles[n]
            print(f"  [{i}] {n:15s}  Lv.{p.level:2d}  |  {p.instrument:7s}  |  {p.total_min:.0f} Min")
        print("  [N] Neues Profil\n")
        c = input("Wahl: ").strip().upper()
        if c.isdigit() and 1 <= int(c) <= len(names):
            p = pm.profiles[names[int(c)-1]]
            lang = input("Sprache [de/en]: ").strip().lower() or "de"
            return p, (lang if lang in LANG else "de")
    print()
    name  = input("Dein Name: ").strip() or "Player1"
    ic    = input("Instrument [1=Gitarre, 2=Klavier]: ").strip()
    instr = "piano" if ic=="2" else "guitar"
    lang  = input("Sprache [de/en]: ").strip().lower() or "de"
    return pm.get(name, instr), (lang if lang in LANG else "de")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN v6
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _instructor, _latest_jpg

    pm = ProgressMgr()
    profile, lang = terminal_setup_v6(pm)
    print(f"\n✓ Willkommen, {profile.name}! NoteIQ ist bereit.")

    start_server()
    print(f"  Browser öffnen: http://localhost:{WEB_PORT}")
    print("  Luna begrüßt dich gleich auf der Seite!\n")

    # ── Kamera-Initialisierung mit detailliertem Feedback ────────────────
    cap = None
    cam_error = None
    print("[Kamera] Suche Kamera...")
    for idx in range(5):
        try:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ret, test_frame = c.read()
                if ret and test_frame is not None:
                    cap = c
                    print(f"[Kamera] ✓ Kamera {idx} ({test_frame.shape[1]}x{test_frame.shape[0]})")
                    break
                else:
                    c.release()
            else:
                c.release()
        except Exception as e:
            cam_error = str(e)

    _instructor = InstructorV6(profile, lang)

    # Audio-Status melden
    if _instructor.audio.error:
        print(f"[Audio] ⚠ {_instructor.audio.error}")
        _instructor.fb.push(f"⚠ Audio: {_instructor.audio.error[:40]}", OR["danger"], 8.)
    if _instructor.tracker.tracker_error:
        print(f"[Tracker] ⚠ {_instructor.tracker.tracker_error}")

    if not cap:
        cam_error = cam_error or "Keine Kamera gefunden oder Kamera blockiert"
        print(f"[Kamera] ⚠ {cam_error}")
        print("[Kamera] ℹ Nur Web-UI verfügbar – alle Audio-Funktionen laufen weiter")
        # Placeholder-Frame mit Status-Anzeige
        ph = np.ones((720,1280,3), np.uint8)*15
        cv2.rectangle(ph, (0,0),(1280,720),(20,20,40),-1)
        _txt(ph,"NoteIQ v16.0",(80,300),1.2,OR["orange"],2)
        _txt(ph,"Kamera nicht verfuegbar",(80,360),0.7,OR["danger"],1)
        _txt(ph,"Alle Audio-Funktionen aktiv",(80,400),0.6,OR["muted"],1)
        _txt(ph,f"http://localhost:{WEB_PORT}",(80,450),0.65,OR["blue"],1)
        _, buf = cv2.imencode('.jpg', ph, [cv2.IMWRITE_JPEG_QUALITY, 80])
        _latest_jpg = buf.tobytes()
        try:
            while True:
                _instructor.audio.update(
                    target_chord   = _instructor.trainer.cur_chord,
                    target_pattern = _instructor.song.cur_pattern,
                    metro_bpm      = float(_instructor.metro.bpm))
                _instructor._update_state(0.)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Unterbrochen]")
        finally:
            _instructor.audio.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow(CV_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CV_WINDOW, 1280, 720)

    def on_mouse(ev, x, y, fl, _):
        if ev==cv2.EVENT_LBUTTONDOWN and _instructor.cal.active:
            _instructor.cal.click(x,y)
            if _instructor.cal.done:
                _instructor.fb.push(_instructor.t("cal_done"), OR["success"])
    cv2.setMouseCallback(CV_WINDOW, on_mouse)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: time.sleep(0.04); continue
            frame = cv2.flip(frame, 1)
            frame = _instructor.process(frame)

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with _frame_lock: _latest_jpg = buf.tobytes()

            cv2.imshow(CV_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27): break
            elif key==ord('h'): _instructor._show_help  = not _instructor._show_help
            elif key==ord('s'): _instructor._show_stats = not _instructor._show_stats
            elif key==ord('b'): _instructor._show_lib   = not _instructor._show_lib
            elif key==ord('g'): _instructor._show_song  = not _instructor._show_song
            elif key==ord('l'): _instructor.switch_lang()
            elif key==ord('m'): _instructor.metro.on = not _instructor.metro.on
            elif key in (ord('+'),ord('=')):
                _instructor.metro.set(_instructor.metro.bpm+5)
                _instructor.fb.push(f"BPM: {_instructor.metro.bpm}", OR["muted"])
            elif key==ord('-'):
                _instructor.metro.set(_instructor.metro.bpm-5)
                _instructor.fb.push(f"BPM: {_instructor.metro.bpm}", OR["muted"])
            elif key==ord('c'):
                _instructor.cal.start()
                _instructor.fb.push(_instructor.t("calibrate"), OR["orange"], 8.)
            elif key==ord('1'): _instructor.switch_instr("guitar")
            elif key==ord('2'): _instructor.switch_instr("piano")
            elif key==ord('p'):
                _instructor.trainer.pos += 1
                _instructor.fb.push(_instructor.t("next_chord",chord=_instructor.trainer.cur_key), OR["muted"])
            elif key==ord('r'): _instructor.tracker._wrist.clear()
            elif key==ord('a'):
                ctx=(f"Instrument:{profile.instrument},"
                     f"Akkord:{_instructor.trainer.cur_key},"
                     f"Note:{_instructor.audio.note_name},"
                     f"Hz:{_instructor.audio.pitch_hz:.1f}")
                _instructor.ai.ask(f"Tipp für {_instructor.trainer.cur_key}", ctx)
            elif key==ord('t'):
                instr=_instructor.profile.instrument
                keys=list(CHORDS[instr].keys())
                print(f"\nVerfügbare Akkorde ({instr}):")
                for i,k in enumerate(keys):
                    print(f"  {k}", end="  " if (i+1)%8 else "\n")
                choice=input("\nAkkord: ").strip()
                if choice in CHORDS[instr]:
                    _instructor.trainer.set_chord(choice)
                    _instructor.fb.push(f"Akkord: {choice}", OR["orange"])
            elif ord('1')<=key<=ord('9') and _instructor._show_song:
                instr=_instructor.profile.instrument
                songs=list(SONGS.get(instr,{}).keys())
                idx2=key-ord('1')
                if idx2<len(songs):
                    _instructor.start_song(songs[idx2])
                    _instructor._show_song=False

    except KeyboardInterrupt:
        print("\n[Unterbrochen]")
    finally:
        _instructor.audio.stop()
        acc = float(np.mean(_instructor._session_acc)) if _instructor._session_acc else 0.
        mins = _instructor.elapsed_min()
        pm.record(profile, acc*100, mins, _instructor.trainer.mastered)
        cap.release(); cv2.destroyAllWindows()
        print(f"\n✓ Session gespeichert: {mins:.1f} Min  |  {acc*100:.0f}% Ø")
        print("Auf Wiedersehen! 🎸🎹 Luna freut sich auf das nächste Mal!")


if __name__ == "__main__":
    main()
