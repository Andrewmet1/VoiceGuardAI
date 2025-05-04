import os
import sys
import base64
import json
import logging
import numpy as np
import requests
import socket
import time
from typing import Optional, Tuple
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import tempfile
import subprocess
import traceback

# Import torch but delay loading heavy libraries
import torch

# Delay these imports - they'll be imported when needed
# import librosa
# import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('voiceguard_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants and paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # /api
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # /VoiceGuardAI_Test
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
VOICEGUARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "voiceguard_model.pth")

# Create required directories
os.makedirs(TEMP_DIR, exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000
HF_API_URL = "https://api-inference.huggingface.co/models/Heem2/Deepfake-audio-detection"
HF_TOKEN = "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"
logger.info(f"✅ HF_API_TOKEN: {HF_TOKEN[:10]}***" if HF_TOKEN else "❌ HF_API_TOKEN not loaded")

# Setup FastAPI with CORS
app = FastAPI(title="VoiceGuard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log paths
logger.info("Project structure:")
logger.info(f"  - Current dir: {CURRENT_DIR}")
logger.info(f"  - Project root: {PROJECT_ROOT}")
logger.info(f"  - Static dir: {STATIC_DIR}")
logger.info(f"  - Temp dir: {TEMP_DIR}")
logger.info(f"  - VoiceGuard model: {VOICEGUARD_MODEL_PATH}")

# Verify directories exist
if not os.path.exists(STATIC_DIR):
    logger.error(f"Static directory not found at: {STATIC_DIR}")
    raise RuntimeError(f"Static directory not found at: {STATIC_DIR}")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    """Serve the main page."""
    logger.info("Serving main page")
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"index.html not found at {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

def prepare_audio_for_huggingface(audio_path: str) -> str:
    """Prepares and encodes a real WAV file for Hugging Face."""
    try:
        import librosa
        import soundfile as sf
        
        # Load and resample audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        encoded_wav = os.path.join(TEMP_DIR, "encoded_for_hf.wav")
        
        # Write as PCM 16-bit WAV with explicit format
        sf.write(encoded_wav, waveform, samplerate=SAMPLE_RATE, format="WAV", subtype="PCM_16")
        
        # Read and encode WAV file
        with open(encoded_wav, "rb") as f:
            wav_bytes = f.read()
            
        # Return base64 encoded string
        return base64.b64encode(wav_bytes).decode("utf-8")
        
    except Exception as e:
        logger.error(f"Error encoding HF WAV: {str(e)}")
        raise

def test_hf_connection() -> bool:
    """Test connection to Hugging Face API."""
    try:
        logger.info("Testing Hugging Face API connection...")
        
        # Create a simple sine wave as test audio (1 second, 440Hz)
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        test_wav_path = os.path.join(TEMP_DIR, "test_audio.wav")
        import soundfile as sf
        sf.write(test_wav_path, test_audio, sample_rate, format="WAV", subtype="PCM_16")
        
        # Encode for API
        test_b64 = prepare_audio_for_huggingface(test_wav_path)
        test_payload = {"inputs": test_b64}
        
        # Test API
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        logger.info(f"🔐 Authorization header: {headers['Authorization'][:20]}...")
        logger.info("Sending test request to Hugging Face API...")
        response = requests.post(HF_API_URL, headers=headers, json=test_payload)
        
        logger.info(f"Test response status: {response.status_code}")
        logger.info(f"Test response body: {response.text}")
        
        if response.status_code == 200:
            logger.info("✅ Hugging Face API connection successful!")
            return True
        else:
            logger.warning(f"⚠️ Hugging Face API returned non-200 status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to Hugging Face API: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Add a new endpoint to test HF connection on demand
@app.get("/api/test-hf-connection")
async def test_hf_connection_endpoint():
    """Test connection to Hugging Face API."""
    result = test_hf_connection()
    if result:
        return {"status": "success", "message": "Connection to Hugging Face API successful"}
    else:
        return {"status": "error", "message": "Failed to connect to Hugging Face API"}

# Initialize HF connection status
hf_api_ready = None

# Modify the query_huggingface_api function to test connection if needed
def query_huggingface_api(audio_data: dict) -> Tuple[Optional[float], Optional[float]]:
    """Query Hugging Face API for deepfake detection."""
    global hf_api_ready
    
    # If we haven't tested the connection yet, do it now
    if hf_api_ready is None:
        hf_api_ready = test_hf_connection()
    
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"🎯 Attempt {attempt}/{MAX_RETRIES}: Querying Hugging Face API...")
            
            # Log request details
            logger.info(f"Request URL: {HF_API_URL}")
            logger.info(f"Request payload type: {type(audio_data)}")
            logger.info(f"Request payload keys: {audio_data.keys()}")
            
            # Make request
            response = requests.post(HF_API_URL, headers=headers, json=audio_data)
            
            # Log response
            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            if response.status_code == 200:
                # Parse response
                result = response.json()
                logger.info(f"Parsed response: {result}")
                
                # Extract scores - handle different label formats
                try:
                    # Try standard format first
                    genuine_score = next(r["score"] for r in result if r["label"] == "REAL")
                    spoof_score = next(r["score"] for r in result if r["label"] == "FAKE")
                except StopIteration:
                    # Try alternative format
                    try:
                        genuine_score = next(r["score"] for r in result if r["label"] == "HumanVoice")
                        spoof_score = next(r["score"] for r in result if r["label"] == "AIVoice")
                    except StopIteration:
                        # If all else fails, extract from any format
                        scores = {item["label"].lower(): item["score"] for item in result}
                        genuine_score = scores.get("real", scores.get("humanvoice", 0.5))
                        spoof_score = scores.get("fake", scores.get("aivoice", 0.5))
                
                logger.info(f"✅ Scores - Genuine: {genuine_score*100:.1f}%, Spoof: {spoof_score*100:.1f}%")
                return genuine_score, spoof_score
                
            elif response.status_code == 503:
                logger.warning(f"🕓 Model not ready (503). Attempt {attempt}/{MAX_RETRIES}. Waiting {RETRY_DELAY} seconds...")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("❌ Maximum retries reached. Giving up.")
                    return None, None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ Exception during request: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
    
    return None, None

# Initialize VoiceGuard model
voiceguard_model = None

# Define model architecture for later use
class VoiceGuardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # First conv block
        self.conv1 = torch.nn.Conv1d(20, 64, 3, padding=1)  # Input channels = 20 (MFCC features)
        self.bn1 = torch.nn.BatchNorm1d(64)
        
        # Second conv block
        self.conv2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        # Third conv block
        self.conv3 = torch.nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(256)
        
        # Pooling and dropout
        self.pool = torch.nn.MaxPool1d(2)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)  # Squeeze time dimension to 1
        self.dropout = torch.nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(256, 128)  # After adaptive pooling, input is 256
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        # Input shape: [batch_size, n_mfcc=20, time_steps=256]
        
        # Conv blocks with batch norm
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # [batch_size, 64, 128]
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # [batch_size, 128, 64]
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  # [batch_size, 256, 32]
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)  # [batch_size, 256, 1]
        
        # Flatten to match fc1 input size
        x = x.view(x.size(0), -1)  # [batch_size, 256]
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))  # [batch_size, 128]
        x = self.dropout(torch.relu(self.fc2(x)))  # [batch_size, 64]
        x = self.fc3(x)  # [batch_size, 2]
        
        return x

# Function to load the model only when needed
def load_voiceguard_model():
    global voiceguard_model
    
    # If model is already loaded, return it
    if voiceguard_model is not None:
        return voiceguard_model
    
    # Check if model file exists
    if not os.path.exists(VOICEGUARD_MODEL_PATH):
        logger.warning(f"⚠️ VoiceGuard model not found at: {VOICEGUARD_MODEL_PATH}")
        return None
    
    try:
        # Create model and load state dict
        model = VoiceGuardModel()
        
        # Load state dict
        state_dict = torch.load(VOICEGUARD_MODEL_PATH, map_location='cpu')
        
        # Debug state dict keys and shapes
        logger.info("VoiceGuard state dict keys and shapes:")
        for key, tensor in state_dict.items():
            logger.info(f"  - {key}: {tensor.shape}")
            
        # Compare model and state dict keys
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        missing_keys = model_keys - state_dict_keys
        extra_keys = state_dict_keys - model_keys
        
        if missing_keys:
            logger.error(f"Missing keys in state dict: {missing_keys}")
        if extra_keys:
            logger.error(f"Extra keys in state dict: {extra_keys}")
            
        try:
            # Load state dict and set to eval mode
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("✅ VoiceGuard model loaded successfully")
            voiceguard_model = model
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load state dict: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error: {repr(e)}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load VoiceGuard model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full error: {repr(e)}")
        return None

def convert_to_wav(input_file: str, output_file: str, sample_rate: int = SAMPLE_RATE) -> bool:
    """Convert audio file to WAV format with specified sample rate."""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', str(sample_rate),
            output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return False

def prepare_audio_for_voiceguard(audio_path: str) -> torch.Tensor:
    """Process audio for VoiceGuard model - returns MFCC features."""
    try:
        import librosa
        
        # Load and resample audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        
        # Convert to tensor
        mfcc_tensor = torch.FloatTensor(mfcc)
        
        # Add batch dimension
        mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Shape: [1, 20, T]
        
        return mfcc_tensor
    except Exception as e:
        logger.error(f"Error preparing audio for VoiceGuard: {str(e)}")
        raise

def predict_voiceguard(audio_tensor: torch.Tensor) -> tuple[float, float]:
    """Get predictions from VoiceGuard model."""
    try:
        with torch.no_grad():
            # Get model predictions
            outputs = load_voiceguard_model()(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract genuine and spoof probabilities
            genuine_prob = float(probabilities[0][0]) * 100
            spoof_prob = float(probabilities[0][1]) * 100
            
            logger.info(f"VoiceGuard scores - Genuine: {genuine_prob:.3f}%, Spoof: {spoof_prob:.3f}%")
            return genuine_prob, spoof_prob
    except Exception as e:
        logger.error(f"Error getting VoiceGuard predictions: {str(e)}")
        return None, None

@app.post("/api/predict")
async def predict(file: UploadFile):
    """Analyze audio file using both VoiceGuard and Hugging Face."""
    try:
        # Save uploaded file
        input_path = os.path.join(TEMP_DIR, "input_audio")
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Convert to WAV format
        wav_path = os.path.join(TEMP_DIR, "audio_16k.wav")
        if not convert_to_wav(input_path, wav_path):
            raise HTTPException(status_code=500, detail="Failed to convert audio format")
        
        # Process audio for Hugging Face (primary, required)
        hf_b64 = prepare_audio_for_huggingface(wav_path)
        hf_payload = {"inputs": hf_b64}
        hf_genuine, hf_spoof = query_huggingface_api(hf_payload)
        
        # If Hugging Face fails, return error with 503 status
        if hf_genuine is None or hf_spoof is None:
            logger.error("Hugging Face API unavailable - returning 503")
            raise HTTPException(
                status_code=503,
                detail="Voice analysis service unavailable - the model is warming up"
            )
        
        # Process audio for VoiceGuard if available (secondary, optional)
        vg_genuine = None
        vg_spoof = None
        
        # Try to load the VoiceGuard model if it's not already loaded
        model = load_voiceguard_model()
        if model:
            try:
                audio_tensor = prepare_audio_for_voiceguard(wav_path)
                vg_genuine, vg_spoof = predict_voiceguard(audio_tensor)
                logger.info(f"VoiceGuard scores - Genuine: {vg_genuine:.1f}%, Spoof: {vg_spoof:.1f}%")
            except Exception as e:
                logger.error(f"VoiceGuard prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue without VoiceGuard
        
        # Create response
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate overall prediction based on both models
        # Higher weight to Hugging Face (80%) and lower weight to VoiceGuard (20%) if available
        hf_spoof_score = float(hf_spoof) * 100  # Convert to percentage
        
        # Determine if this is likely AI-generated (spoof) or human
        is_spoof = hf_spoof_score > 50
        prediction = "spoof" if is_spoof else "human"
        
        # Calculate confidence score (0-100)
        confidence = hf_spoof_score if is_spoof else (100 - hf_spoof_score)
        
        # Create message based on prediction
        message = "AI-Generated Voice Detected" if is_spoof else "Human Voice Detected"
        
        # Format response to match frontend expectations
        response = {
            "timestamp": timestamp,
            "prediction": prediction,
            "confidence": confidence,
            "message": message,
            "model_details": {
                "huggingface": {
                    "enabled": True,
                    "genuine": float(hf_genuine) * 100,  # Convert to percentage
                    "spoof": float(hf_spoof) * 100  # Convert to percentage
                },
                "voiceguard": {
                    "enabled": vg_genuine is not None and vg_spoof is not None,
                    "genuine": float(vg_genuine) if vg_genuine is not None else 0,
                    "spoof": float(vg_spoof) if vg_spoof is not None else 0
                }
            }
        }
        
        # Add top-level voiceguard scores for mobile app compatibility
        response["voiceguard"] = {
            "genuine_score": float(vg_genuine) if vg_genuine is not None else 0,
            "spoof_score": float(vg_spoof) if vg_spoof is not None else 0
        }
        
        # Store original format for recent analyses
        recent_analysis = {
            "timestamp": timestamp,
            "huggingface": {
                "genuine": float(hf_genuine),
                "spoof": float(hf_spoof)
            }
        }
        
        # Add VoiceGuard results if available
        if vg_genuine is not None and vg_spoof is not None:
            recent_analysis["voiceguard"] = {
                "genuine": float(vg_genuine),
                "spoof": float(vg_spoof)
            }
        
        # Add to recent analyses
        recent_analyses.insert(0, recent_analysis)
        if len(recent_analyses) > 5:
            recent_analyses.pop()
        
        return response
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (like 503)
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup temp files
        for file in ['input_audio', 'audio_16k.wav', 'encoded_for_hf.wav']:
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except:
                pass

@app.get("/api/recent")
async def get_recent():
    """Get recent analysis results."""
    return recent_analyses

# Store recent analyses (max 5)
recent_analyses = []

if __name__ == "__main__":
    # Find an available port if 8009 is in use
    port = 8009
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
    except OSError:
        # Port is in use, find an available one
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 0))
            port = s.getsockname()[1]
        logger.info(f"Port 8009 is in use, using fallback port {port}")
    
    import uvicorn
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
