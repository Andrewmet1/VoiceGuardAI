# VoiceGuardAI

VoiceGuardAI is a comprehensive mobile application for AI voice scam detection with real-time analysis capabilities.

## Repository Structure

- `/api` - FastAPI server for voice analysis
  - `web_test.py` - Main API server code
  - `requirements.txt` - Python dependencies

- `/models` - Trained AI models
  - `voiceguard_model.pth` - VoiceGuard model for voice spoofing detection

- `/mobile-app` - React Native mobile application (to be added from MacBook)

## Setup Instructions

### API Server

1. Install Python dependencies:
   ```
   cd api
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```
   python web_test.py
   ```

### Mobile App

The mobile app code will be added from the MacBook. After adding:

1. Install dependencies:
   ```
   cd mobile-app
   npm install --legacy-peer-deps
   ```

2. Start the development server:
   ```
   npx expo start
   ```

## Features

- Real-time AI voice detection for unknown callers
- Background monitoring with minimal battery impact
- Voice Activity Detection to isolate caller's voice
- Local on-device processing for privacy
- Call history with AI detection results
- Detailed settings for user customization
