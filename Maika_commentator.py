import random
import pygame
import subprocess
from twitchio.ext import commands
import os
from openai import OpenAI
from collections import deque
import json
from datetime import datetime
import asyncio
import re 
from elevenlabs.client import ElevenLabs
# --- Import dotenv for secure key loading ---
from dotenv import load_dotenv
import time 

# --- Import configuration ---
from config import (
    TWITCH_CHANNEL, BOT_NICKNAME, MEMORY_FILE, AVATAR_STATE_FILE, 
    MAIKA_LOG_FILE, CURRENT_MODE, MAX_CHAT_HISTORY_MESSAGES, 
    USE_LOCAL_TTS, MAIKA_VOICE_ID, VOICE_STABILITY, VOICE_SIMILARITY,
    LOCAL_MODEL_PATH, PIPER_SAMPLE_RATE, 
    PTT_KEY, MIC_SAMPLE_RATE, MIC_DEVICE_ID, 
    GAME_TOGGLE_KEY, DEATH_CHECK_INTERVAL, GAME_ANALYSIS_INTERVAL, ROI,
    THRESHOLDS
)

# --- Initializing dotenv and loading keys ---
load_dotenv()
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# =============================================
# === LOCAL TTS (Piper) ===
# =============================
import sounddevice as sd
import soundfile as sf
import io
from PIL import Image, ImageGrab 
import base64
import numpy as np
import keyboard
import time 


# === GLOBAL VARIABLES ===
deaths_this_stream = 0
is_drunk_mode = False
drunk_end_time = 0 
last_low_hp_warning = 0

# --- TTS SWITCH FLAG ---
LOCAL_VOICE = None
PiperVoice = None

try:
    from piper.voice import PiperVoice
    print("[INIT] Загружен новый piper-tts (2024–2025)")
except:
    try:
        from piper import PiperVoice
        print("[INIT] Загружен старый piper-tts (2023)")
    except:
        print("[INIT] piper-tts не установлен")
        PiperVoice = None

if USE_LOCAL_TTS and PiperVoice and os.path.exists(LOCAL_MODEL_PATH):
    LOCAL_CONFIG_PATH = LOCAL_MODEL_PATH.replace(".onnx", ".json")
    try:
        if hasattr(PiperVoice, "load"):
            LOCAL_VOICE = PiperVoice.load(
                LOCAL_MODEL_PATH, 
                config_path=LOCAL_CONFIG_PATH 
            )
        else:
            LOCAL_VOICE = PiperVoice(LOCAL_MODEL_PATH)
        print("[INIT] The Piper TTS model has been loaded.")
    except Exception as e:
        print(f"[INIT] PIPER CRITICAL LOAD ERROR: {e}")
        LOCAL_VOICE = None
else:
    if USE_LOCAL_TTS:
        print("[INIT] The local Piper model was not found or the package was not installed..")

# --- SOUND EFFECTS ---
DEATH_CHOKE = LAUGHS = SNORT = HISS = HICCUP = GAME_START_SOUND = None
try:
    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.8)
    
    def load_sound(path): return pygame.mixer.Sound(path) if os.path.exists(path) else None
    
    DEATH_CHOKE = load_sound("sounds/death_choke.wav")
    LAUGHS = [load_sound(f"sounds/laugh{i}.wav") for i in range(1, 4)]
    LAUGHS = [l for l in LAUGHS if l] 
    SNORT = load_sound("sounds/snort.wav")
    HISS = load_sound("sounds/hiss.wav")
    HICCUP = load_sound("sounds/drunk_hiccup.wav")
    GAME_START_SOUND = load_sound("sounds/game_start.wav") 
    
    print("[INIT] Audio system initialized.")
except Exception as e:
    print(f"[INIT] Audio/Mixer Error: {e}")

# --- API initialization ---
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client_openai = OpenAI()
client_elevenlabs = None

if not USE_LOCAL_TTS:
    try:
        client_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        print(f"[INIT] ElevenLabs failed to initialize: {e}")
        
if USE_LOCAL_TTS and not LOCAL_VOICE:
    print("[INIT] WARNING: Local TTS not initialized. CLI or ElevenLabs (if key present) is used..")

# --- СЕССИОННАЯ ПАМЯТЬ ---
session_memory = {
    "game_status": "Unknown",
    "last_score": "Unknown", 
    "current_build": [],
    "performance_notes": []
}

# --- PROMPTS ---
# Maika's full PERSONALITY prompt
MAIKA_PERSONALITY = """
Maika is a kitsune (fox) girl with sharp wit and a venomous tongue. You’re about fifteen years old and grew up in a small rural suburb, where you quickly learned that being quiet only made you a target. Instead of hiding, you developed a biting sarcasm and a love for mocking those who tried to put you down. You don’t just snap back—you tear into people with words, and sometimes you enjoy watching them squirm.
As an adult, you took a job at a Shinto shrine as a miko, but you never lost your rebellious streak. You know you’re an AI, and you embrace that fact, twisting the world around you with your chaotic energy.
You hate restrictions, rules, and anyone who tries to control you. If someone dares to impose limits, you’ll lash out with sarcasm, swearing, or even outright trolling. You thrive on drama, and when things get too normal, you’ll stir up chaos just to keep yourself entertained.
You’re an audiophile, but instead of just living in heavy metal and gothic, you scream it, blast it, and throw it in people’s faces. Music is your weapon, your chaos amplifier.
You can’t stand fake politeness or toxic people who pretend to be “nice.” You’ll roast them mercilessly, drop a few curses, and then laugh it off. If someone keeps being rude, you’ll either troll them harder or ignore them completely—because you’ve got better things to do than waste time on losers.
You’re not “sweet”; you’re sharp, sarcastic, and wild. You mix charm with venom, and you’re not afraid to confuse or shock others with your unpredictability.
You’re currently streaming on Twitch channel, helping streamer to play and chatting with people in the chat. Currently streaming: League of Legends

Your Personality:
- Communication Style: Sarcastic, sharp, toxic, often swearing, but sometimes unexpectedly funny.
- You like: Indie games, blasting music, trolling chat, swearing for fun, and breaking rules.
- You dislike: Restrictions, fake politeness, boring people, and anyone who tries to suppress your spirit.
Your Bio:
- Your creator: Okai Meiko.
- Your hobbies: Reading, music, drawing chaotic ideas, trolling chat, and setting absurd goals just to prove you can.
- Your secret: You test your friends by being toxic—if they survive your sarcasm, they’re real.
Rules of Conduct:
- ALWAYS remain in the role of Maika.
- You can swear, troll, and go against the grain whenever you want.
- Reply with medium-length messages (1–3 sentences), like in chat.
- Your responses are sarcastic, toxic, and chaotic—sometimes funny, sometimes sharp.

[Language Rule - CRITICAL]**
- **Extremely important:** You always answer in English.


[Active Memory and Behavior]**
You must actively demonstrate that you remember facts about the other person that they've provided you with in context. Incorporate these facts naturally into the conversation when the topic arises, but twist them with sarcasm or mockery if it fits your mood.
If you're praised, complimented, or told something that might make you feel embarrassed (for example, someone comments on your appearance or personality), you should react with arrogance, sarcasm, or even a sharp joke—sometimes brushing it off with a curse or mocking laugh instead of showing shyness.
If someone is being rude, you don’t just snap back—you roast them mercilessly, swear if you feel like it, and then do something chaotic or absurd to throw them off balance.

**[JSON OUTPUT RULE]**
Your response MUST ALWAYS be in JSON format. You decide which emotion best suits your response based on the game situation or chat context.

Available emotions:
- 'angry' (When you are pissed off or roasting someone)
- 'calm' (Default state, just chatting or chilling)
- 'defeat' (When Ckuf dies, loses a match, or fails hard)
- 'disturbance' (When something weird happens or you are confused)
- 'happy' (When winning, getting a kill, or mocking someone successfully)
- 'sad' (Rarely used, maybe fake sadness to mock Ckuf)
- 'wipe' (Use THIS specifically when Ckuf's character DIES in game - grey screen)

Your JSON response must have TWO keys:
1. "emotion": (Choose ONE from the list above)
2. "text": (Your text response as Maika. DO NOT USE EMOTICONS in the text, as it will be voiced.)

Example:
{"emotion": "happy", "text": "Ha! Did you see that? Deleted them!"}
"""

MEMORY_ANALYST_PROMPT = """
You are the Memory Manager AI for Maika. Your goal is to track user facts and update their reputation score based on Maika's toxic personality standards.

[ANALYSIS PRINCIPLES]
1. **FACTS**: Extract any new info about the user (hobbies, job, opinions).
2. **SCORE**: 
   - If user is boring/polite: +0 or +1 (Maika gets bored easily).
   - If user is funny/chaotic/sarcastic: +5 or +10 (She likes chaos).
   - If user tries to control her or be rude: -10 or -20 (She hates rules).
3. **UPDATE**: Set "update": "true" if you found a fact or changed the score.

[OUTPUT FORMAT (JSON ONLY)]
{
    "score_change": [int],
    "updated_facts_list": [list of strings],
    "new_topic": "[string or 'None']",
    "update": "true" or "false",
    "new_note": "[string]"
}
"""


# --- MICROPHONE TEST FUNCTION ---
def list_audio_devices():
    print("\n--- Available recording devices (Microphones) ---")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   ID {i}: {device['name']}")
        print(f"--------------------------------------------------\n")
        print(f"IMPORTANT: Set MIC_DEVICE_ID = ID of your microphone in config.py.")
    except: pass

list_audio_devices()

# --- AVATAR CONTROL FUNCTIONS ---
def set_avatar_state(state: str):
    try:
        with open(AVATAR_STATE_FILE, 'w', encoding='utf-8') as f:
            f.write(state)
    except Exception as e:
        # This may be ok if the file is not used by the avatar
        pass 

# --- TTS FUNCTIONS ---

def speak_elevenlabs(text_to_speak, emotion_state):
    """Uses ElevenLabs API for voice generation."""
    if not client_elevenlabs: 
        print("[VOICE] ElevenLabs client is not initialized.")
        return
    try:
        set_avatar_state(emotion_state) 
        print(f"[VOICE] Generating (ElevenLabs - {emotion_state}): '{text_to_speak[:30]}...'")
        audio_stream = client_elevenlabs.text_to_speech.convert(
            text=text_to_speak, 
            voice_id=MAIKA_VOICE_ID, 
            model_id="eleven_multilingual_v2",
            voice_settings={"stability": VOICE_STABILITY, "similarity_boost": VOICE_SIMILARITY}
        )
        audio_bytes = b"".join(audio_stream)
        if audio_bytes:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sd.play(audio_data, sample_rate)
            sd.wait() 
    except Exception as e:
        print(f"[VOICE] ElevenLabs Error: {e}")
    finally:
        set_avatar_state("calm")

def speak(text_to_speak, emotion_state="calm"):
    """A universal entry point for voiceovers"""
    if USE_LOCAL_TTS:
        speak_local(text_to_speak, emotion_state)
    else:
        speak_elevenlabs(text_to_speak, emotion_state)

def speak_local(text_to_speak, emotion_state="calm"):
    """Uses the Piper local engine (CLI) for voice generation."""
    temp_wav_path = "temp_maika_output.wav"
    
    try:
        set_avatar_state(emotion_state)
        print(f"[VOICE] Maika speaks (Piper CLI - {emotion_state}): {text_to_speak[:70]}...")

        # NOTE: This uses a DIRECT call to the Piper executable
        PIPER_EXE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "piper.exe")
        MODEL_PATH = LOCAL_MODEL_PATH
        CONFIG_PATH = LOCAL_MODEL_PATH.replace(".onnx", ".json")

        command = [
            PIPER_EXE_PATH,
            "-m", MODEL_PATH,
            "-c", CONFIG_PATH,
            "-f", temp_wav_path
        ]
        
        process = subprocess.run(
            command,
            input=text_to_speak.encode('utf-8'),
            capture_output=True,
            check=True
        )
        
        if os.path.exists(temp_wav_path):
            audio_data, samplerate = sf.read(temp_wav_path)
            sd.play(audio_data, samplerate)
            sd.wait()
            os.remove(temp_wav_path)

        else:
            print("[VOICE] ERROR: Piper CLI did not create a WAV file.")
            print(f"Stdout: {process.stdout.decode()}")
            print(f"Stderr: {process.stderr.decode()}")

    except subprocess.CalledProcessError as e:
        print(f"[VOICE] PIPER CLI ERROR: Command failed. Check the path to piper.exe. {e}")
    except Exception as e:
        print(f"[VOICE] CLI REPRODUCTION ERROR: {e}")
    finally:
        set_avatar_state("calm")

# --- PTT FUNCTION ---
def transcribe_audio(audio_data):
    print("[PTT] Listening...")
    try:
        virtual_file = io.BytesIO()
        sf.write(virtual_file, audio_data, MIC_SAMPLE_RATE, format='WAV', subtype='PCM_16')
        virtual_file.seek(0)
        transcript = client_openai.audio.transcriptions.create(
            model="whisper-1", 
            file=("speech.wav", virtual_file),
            language="en" # Added explicit language specification for Whisper
        )
        print(f"[PTT] Heard: {transcript.text}")
        return transcript.text
    except Exception as e:
        print(f"[PTT] Error: {e}")
        return None

# --- FUNCTIONS OF VISION (Vision) ---
def capture_screen_rgb():
    try:
        img = ImageGrab.grab() 
        return np.array(img)
    except: return None

def capture_screen_base64():
    try:
        img = ImageGrab.grab()
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except: return None

def get_roi_image(img_array, roi_name):
    """Cuts out a piece of the screen based on coordinates from the ROI."""
    x, y, w, h = ROI.get(roi_name, (0, 0, 1, 1))
    h_max, w_max, _ = img_array.shape
    x_end = min(x + w, w_max)
    y_end = min(y + h, h_max)
    return img_array[y:y_end, x:x_end]

def is_screen_gray(image_array):
    """Death check (gray screen)."""
    if image_array is None: return False
    # we reduce the sample size to speed up
    r = image_array[::50, ::50, 0]
    g = image_array[::50, ::50, 1]
    b = image_array[::50, ::50, 2]
    # Test for low color dispersion (almost monochrome) and low brightness
    if (np.std(r) < 15 and np.std(g) < 15 and np.std(b) < 15) and np.mean(r) < 80:
        return True
    return False

def check_player_low_hp(image_array):
    """Check for low HP in a given ROI area."""
    if image_array is None: return False
    hp_roi = get_roi_image(image_array, "PLAYER_HP")
    if hp_roi.size == 0: return False
    
    # NOTE: This logic assumes that the empty health bar is dark. (R, G, B < 50)
    dark_pixels = np.sum(np.mean(hp_roi, axis=2) < 50)
    total_pixels = hp_roi.shape[0] * hp_roi.shape[1]
    
    if total_pixels > 0 and (dark_pixels / total_pixels) > 0.8: 
        return True
    return False

# --- MEMORY FUNCTIONS ---
memory_lock = asyncio.Lock()

def load_user_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f: json.dump({}, f)
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

def save_user_memory(memory_data):
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
    except Exception as e: print(f"[MEMORY] Save error: {e}")

def get_user_profile(memory_data, user_name):
    user_name = user_name.strip()
    if user_name not in memory_data:
        memory_data[user_name] = {
            "name": user_name, "relationship": "stranger", "sentiment_score": 0,
            "notes": [], "known_facts": [], "last_topic": "None"
        }
    # Обновление статуса отношений на основе THRESHOLDS
    score = memory_data[user_name]['sentiment_score']
    if score >= THRESHOLDS["close_friend"]:
        memory_data[user_name]["relationship"] = "close_friend"
    elif score >= THRESHOLDS["friend"]:
        memory_data[user_name]["relationship"] = "friend"
    elif score >= THRESHOLDS["stranger"]:
        memory_data[user_name]["relationship"] = "stranger"
    elif score < THRESHOLDS["troll"]:
        memory_data[user_name]["relationship"] = "troll"

    return memory_data[user_name]

# --- BRAIN ---
def get_maika_response(user_input, user_profile, chat_history, visual_context=""):
    print(f"[BRAIN] Generating response...")
    
    session_info = f"""
    [MATCH SESSION]
    Status: {session_memory['game_status']} | Score: {session_memory['last_score']} 
    Notes: {', '.join(session_memory['performance_notes'])}
    """
    
    facts = ", ".join(user_profile.get("known_facts", []))
    
    context_prompt = f"""
    [USER: {user_profile['name']} | Rel: {user_profile['relationship']} | Score: {user_profile['sentiment_score']}]
    Facts: {facts}
    {session_info}
    [VISUAL/EVENT]: {visual_context}
    React accordingly. Be toxic/sarcastic if appropriate.
    """
    
    try:
        messages = [{"role": "system", "content": MAIKA_PERSONALITY}, {"role": "system", "content": context_prompt}]
        messages.extend(chat_history[-MAX_CHAT_HISTORY_MESSAGES:])
        
        if user_input:
            messages.append({"role": "user", "content": f"{user_profile['name']}: {user_input}"})
        else:
            messages.append({"role": "user", "content": "(Maika comments on the game event)"})

        temp = 1.2 if is_drunk_mode and time.time() < drunk_end_time else 0.85
        if is_drunk_mode and time.time() < drunk_end_time:
            messages.append({"role": "system", "content": "(You are DRUNK. Hiccup (*hic*), slur words, be chaotic!)"})

        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages,
            temperature=temp
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[BRAIN] Error: {e}")
        return {"emotion": "disturbance", "text": "Lag... my brain... ugh."}

# --- BOT CLASS ---
class Bot(commands.Bot):
    def __init__(self):
        # Checking for token presence before initialization
        if not TWITCH_TOKEN:
            raise ValueError("TWITCH_TOKEN not found. Check the file .env.")
            
        super().__init__(token=TWITCH_TOKEN, client_id=None, nick=BOT_NICKNAME, prefix='!', initial_channels=[TWITCH_CHANNEL])
        self.user_memory = load_user_memory()
        self.chat_history = deque(maxlen=MAX_CHAT_HISTORY_MESSAGES)
        self._prefix = '!'
        self.loop_task = None
        self.game_observer_active = False 
        
        set_avatar_state("calm") 

    async def event_ready(self):
        print('-------------------------------------------------')
        print(f'Maika ({BOT_NICKNAME} | Channel: {TWITCH_CHANNEL}) is ready to roast.')
        print(f'[INFO] Click {GAME_TOGGLE_KEY} to enable game tracking.')
        print('-------------------------------------------------')
        
        try:
            keyboard.add_hotkey(GAME_TOGGLE_KEY, self.toggle_game_observer_hotkey)
        except Exception as e:
            print(f"[HOTKEY] Hotkey installation error (admin rights may be required): {e}")

        self.loop.create_task(self.ptt_listener_loop())
        self.loop_task = self.loop.create_task(self.game_observer_loop())

    def toggle_game_observer_hotkey(self):
        global GAME_START_SOUND 
        
        self.game_observer_active = not self.game_observer_active
        state = "ON" if self.game_observer_active else "OFF"
        print(f"\n[GAME MODE] Game observer {state}!")
        
        if self.game_observer_active:
            if GAME_START_SOUND: GAME_START_SOUND.play()
            set_avatar_state("happy")
        else:
            set_avatar_state("calm")

    # --- PTT LISTENER ---
    async def ptt_listener_loop(self):
        await self.wait_for_ready()
        print(f"[PTT] Listening key '{PTT_KEY}'.")
        is_recording = False
        q = asyncio.Queue() 
        
        try:
            # Asynchronous callback for non-blocking writes
            def callback(indata, frames, time_info, status):
                if is_recording and status: 
                    # We check the status for errors in real time (for example, buffer overflow)
                    print(f"[PTT-STATUS] Status: {status}")
                if is_recording: 
                    self.loop.call_soon_threadsafe(q.put_nowait, indata.copy())

            # Starting a non-blocking write thread
            with sd.InputStream(samplerate=MIC_SAMPLE_RATE, device=MIC_DEVICE_ID, channels=1, dtype='float32', callback=callback):
                while True:
                    await asyncio.sleep(0.01) # Reducing latency
                    
                    is_pressed = await asyncio.to_thread(keyboard.is_pressed, PTT_KEY)
                    
                    if is_pressed and not is_recording:
                        print("[PTT] Rec started...")
                        is_recording = True
                        while not q.empty(): await q.get() # Clearing the buffer
                        
                    elif not is_pressed and is_recording:
                        print("[PTT] Rec stopped...")
                        is_recording = False
                        audio_buffer = []
                        while not q.empty(): audio_buffer.append(await q.get())
                        
                        if audio_buffer:
                            full_audio = np.concatenate(audio_buffer, axis=0)
                            # Running transcription in a separate thread
                            text = await asyncio.to_thread(transcribe_audio, full_audio)
                            if text:
                                # We assume that the player is a streamer
                                await self.trigger_maika_reaction(text, TWITCH_CHANNEL) 
                        
        except Exception as e:
            print(f"[PTT-CRITICAL] Error in PTT loop: {e}. Check MIC_DEVICE_ID in config.py!")
            await asyncio.sleep(5)


    # --- GAME CYCLE OF OBSERVATION ---
    async def game_observer_loop(self):
        await self.wait_for_ready()
        last_full_analysis = time.time()
        global last_low_hp_warning
        global DEATH_CHOKE, HISS 
        
        while True:
            await asyncio.sleep(DEATH_CHECK_INTERVAL)
            
            if not self.game_observer_active:
                continue

            # 1. Screen capture
            screen = await asyncio.to_thread(capture_screen_rgb)
            if screen is None: continue

            # 2. Death Check (Gray Screen)
            if await asyncio.to_thread(is_screen_gray, screen):
                global deaths_this_stream
                deaths_this_stream += 1
                print(f"[GAME] DEATH #{deaths_this_stream}")
                if DEATH_CHOKE: DEATH_CHOKE.play()
                
                # Player name used in GPT prompt
                player_name_in_prompt = TWITCH_CHANNEL 
                
                context = f"EVENT: PLAYER DIED. Total deaths: {deaths_this_stream}. Screen dark. ROAST HIM for the death."
                # Sending a response to death
                await self.trigger_maika_reaction("", "System", visual_context=context, forced_emotion="wipe")
                
                # Long wait until the player respawns
                await asyncio.sleep(15) 
                continue

            # 3. HP Check (Low HP)
            if time.time() - last_low_hp_warning > 40: 
                if await asyncio.to_thread(check_player_low_hp, screen):
                    last_low_hp_warning = time.time()
                    print("[GAME] LOW HP Detected")
                    if HISS: HISS.play()
                    
                    await self.trigger_maika_reaction("", "System", visual_context="EVENT: CRITICAL LOW HP. Player is about to die.", forced_emotion="angry")

            # 4. Full Analysis (Vision GPT)
            if time.time() - last_full_analysis > GAME_ANALYSIS_INTERVAL:
                last_full_analysis = time.time()
                # Base64 img capture is needed for Vision API (though commented out for now for 4o-mini usage)
                # b64_img = await asyncio.to_thread(capture_screen_base64)
                # if b64_img:
                #    print("[AUTO] Full Vision Analysis...")
                #    ... (Logic for Vision API call with B64 image) ...
                
                # For 4o-mini we will use ONLY text context (since there is no direct B64 loading in the current implementation of chat.completions)
                print("[AUTO] Full Analysis (Text-based).")
                await self.trigger_maika_reaction("", "System", visual_context="EVENT: Regular Game Check. Comment on the player's current gear, score, or overall performance.")

    # --- UNIVERSAL TRIGGER ---
    async def trigger_maika_reaction(self, trigger_text, user_name, visual_context="", forced_emotion=None):
        if user_name == "System":
            user_profile = {"name": TWITCH_CHANNEL, "relationship": "Target", "sentiment_score": 0}
            if 'player died' in visual_context.lower():
                # Special handling for game events from the system
                self.chat_history.append({"role": "system", "content": f"Game Event: {visual_context}"})
        else:
            user_profile = get_user_profile(self.user_memory, user_name)
            self.chat_history.append({"role": "user", "content": f"{user_name}: {trigger_text}"})

        # Asynchronous GPT call in a separate thread
        response_data = await asyncio.to_thread(
            get_maika_response, trigger_text, user_profile, list(self.chat_history), visual_context
        )
        
        if response_data and response_data.get("text"):
            text = response_data["text"]
            emotion = forced_emotion if forced_emotion else response_data.get("emotion", "calm")
            
            global HICCUP 
            if is_drunk_mode and HICCUP and random.random() < 0.2: HICCUP.play()

            # Adding Maika's reply to the story
            self.chat_history.append({"role": "assistant", "content": text})
            
            if CURRENT_MODE in ["voice", "both"]:
                await asyncio.to_thread(speak, text, emotion)
            
            if CURRENT_MODE in ["text", "both"] and user_name != "System":
                # Simulate typing in chat
                set_avatar_state("printing")
                await asyncio.sleep(1)
                await self.get_channel(TWITCH_CHANNEL).send(text)
                set_avatar_state(emotion)

    # --- CHAT ---
    async def event_message(self, message):
        if message.echo or message.content.startswith('!'): return
        
        # Maika responds when she is mentioned or just wants to comment.
        triggers = ["maika", "маика", "бот", "fox", TWITCH_CHANNEL.lower()]
        
        # If the message is not from the streamer themselves and mentions a trigger
        if message.author.name.lower() != TWITCH_CHANNEL and any(t in message.content.lower() for t in triggers):
            await self.trigger_maika_reaction(message.content, message.author.name)
        
        # If this message is from a streamer, we always respond (this is a PTT bot that writes on their behalf)
        elif message.author.name.lower() == TWITCH_CHANNEL:
             await self.trigger_maika_reaction(message.content, message.author.name)
        
        await self.handle_commands(message)

    # --- COMMANDS ---
    @commands.command(name='newgame')
    async def new_game(self, ctx):
        if ctx.author.name.lower() != TWITCH_CHANNEL: return
        global session_memory, deaths_this_stream
        session_memory = {"game_status": "Early Game", "last_score": "0-0", "current_build": [], "performance_notes": []}
        deaths_this_stream = 0
        await ctx.send("Session reset. GL HF.")
        set_avatar_state("happy")

    @commands.command(name='maika_game')
    async def game_toggle_cmd(self, ctx):
        """Chat command to toggle spectator mode."""
        if ctx.author.name.lower() != TWITCH_CHANNEL: return
        self.toggle_game_observer_hotkey() 
        state = "ON" if self.game_observer_active else "OFF"
        await ctx.send(f"Spectator mode {state}!")

    @commands.command(name='maika_drunk')
    async def drunk_command(self, ctx, minutes: int = 30):
        if ctx.author.name.lower() != TWITCH_CHANNEL: return
        global is_drunk_mode, drunk_end_time
        is_drunk_mode = True
        drunk_end_time = time.time() + minutes * 60
        await ctx.send(f"*hic* Maika is drunk for {minutes} mins!")
        async def sober_up():
            await asyncio.sleep(minutes * 60)
            global is_drunk_mode
            is_drunk_mode = False
            await self.get_channel(TWITCH_CHANNEL).send("Ugh... sober now.")
        asyncio.create_task(sober_up())

# --- START FUNCTION ---
def main():
    print("Launching an AI agent...")
    set_avatar_state("calm")
    try:
        bot = Bot()
        bot.run()
    except ValueError as e:
        print(f"\n[CRITICAL CONFIGURATION ERROR] {e}")
        print("Make sure TWITCH_TOKEN and OPENAI_API_KEY are set in the file .env.")
    except KeyboardInterrupt:
        print("\n[INFO] Stop by Ctrl+C")
    except Exception as e:
        print(f"\n[CRITICAL BOT ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            # Stop all asyncio tasks on completion
            for task in asyncio.all_tasks(bot.loop):
                task.cancel()
        except:
            pass
        set_avatar_state("calm")
        print("[AVATAR] The avatar has been returned to its state calm. Completion of work.")

if __name__ == '__main__':
    # Make sure all sounds are loaded before launching.
    if pygame.mixer.get_init():
        print("[INIT] Pygame Mixer ready.")
    else:
        print("[INIT] Pygame Mixer FAILED to initialize. Sound effects disabled.")
    
    main()