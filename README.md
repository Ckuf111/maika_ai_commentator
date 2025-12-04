Maika_Commentator_Project

A project for the toxic commentator sitting next to you, commenting on your game. This code is configured for League of Legends, but can be reconfigured for any other game.
P.S. I'm far from a programmer, and I'm more than confident that this code can be further developed, optimized, and generally made better and more interesting.


🛠️ Installation and Setup

Prerequisites

You will need the following installed:

Python 3.10+

Git

Twitch Account (for the bot)

OpenAI API Key (for GPT and Whisper)

ElevenLabs API Key (optional, for cloud TTS)

Step 1: Clone the Repository

git clone [https://github.com/ckuf/maika_ai_commentator.git](https://github.com/ckuf/maika_ai_commentator.git)
cd maika_ai_commentator


Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment (venv) to manage dependencies cleanly.

# Create the environment
python -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
.\venv\Scripts\activate


Step 3: Install Dependencies

Use the generated requirements.txt file to install all necessary Python packages:

pip install -r requirements.txt


Step 4: Configure API Keys

Copy the template file to create your environment file:

cp .env_template .env


Edit the new .env file and fill in your Twitch, OpenAI, and ElevenLabs API keys.

Step 5: Configure config.py

Adjust the settings in config.py to match your stream:

Set TWITCH_CHANNEL and BOT_NICKNAME.

Configure MIC_DEVICE_ID (run the script once to see the list of devices if needed).

Calibrate the ROI coordinates for your game regions (HP bar, KDA, etc.).

Step 6: Run the Bot

Start the main script:

python maika_commentator.py