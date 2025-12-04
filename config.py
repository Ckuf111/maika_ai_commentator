import os

# --- AI AND BOT SETTINGS ---
TWITCH_CHANNEL = "Your twitch channel"
BOT_NICKNAME = "maika_ai"
MEMORY_FILE = "maika_memory.json"
AVATAR_STATE_FILE = "maika_state.txt" # For external avatar program
MAIKA_LOG_FILE = "maika_log.txt"

# --- OPERATING MODE SETTINGS ---
# "both" (voice and chat), "voice" (only voice), "text" (only chat)
CURRENT_MODE = "both" 
MAX_CHAT_HISTORY_MESSAGES = 15 # Maximum number of messages for GPT context

# --- TTS SETTINGS ---
# Use True for local TTS (Piper), False for cloud (ElevenLabs)
USE_LOCAL_TTS = True 

# ElevenLabs Settings (Used if USE_LOCAL_TTS = False)
MAIKA_VOICE_ID = "h2dQOVyUfIDqY2whPOMo" 
VOICE_STABILITY = 0.40 
VOICE_SIMILARITY = 0.80 

# Local TTS settings (Piper)
# IMPORTANT: The path must be relative or absolute to the .onnx file
LOCAL_MODEL_PATH = "voices/en_US-kristin-medium.onnx"
PIPER_SAMPLE_RATE = 22050 # Sampling frequency for the Piper model

# --- PTT (Push-to-Talk) SETTINGS ---
PTT_KEY = 'home' # Hotkey to start voice recording
MIC_SAMPLE_RATE = 22050 # Microphone sampling rate for recording (recommended)
MIC_DEVICE_ID = 1 # ID of your recording device (microphone). Check the device list!

# --- GAME AND VISION CONTROL SETTINGS ---
GAME_TOGGLE_KEY = 'f10' # Hotkey for toggling spectating
DEATH_CHECK_INTERVAL = 2 # How often to check the death screen (in seconds)
GAME_ANALYSIS_INTERVAL = 60 # How often to perform a full screen analysis with Vision GPT (in seconds)

# Region of Interest (ROI) - Specific to your game!
# (X_START, Y_START, WIDTH, HEIGHT)
# IMPORTANT: These coordinates must be adjusted to suit your monitor and game.
ROI = {
    # Player health bar area (example for 1920x1080)
    "PLAYER_HP": (760, 980, 400, 15), 
    # ACCOUNT/KDA area
    "KDA": (1700, 0, 200, 100) 
}

# --- MAIKA'S RELATIONSHIP THRESHOLDS (For internal memory) ---
THRESHOLDS = {
    "troll": -50,
    "stranger": 0,
    "friend": 200,
    "close_friend": 500
}