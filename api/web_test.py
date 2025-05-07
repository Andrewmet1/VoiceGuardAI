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
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, WavLMModel, AutoFeatureExtractor

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

# Get token directly from environment variable (hardcoded for debugging)
HF_TOKEN = "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"
logger.info(f"✅ HF_API_TOKEN: {HF_TOKEN[:10]}***" if HF_TOKEN else "❌ HF_API_TOKEN not loaded")

# Import torch but delay loading heavy libraries
import torch

# Delay these imports - they'll be imported when needed
# import librosa
# import soundfile as sf

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
        
        # Load and resample audio - use 10 seconds for better detection
        max_duration = 10  # seconds
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
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

# Initialize HF models for local inference
# First model: Deepfake-audio-detection
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "hf_model_cache")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# If we have a local cached version, use it; otherwise use the model ID
if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    HF_MODEL_PATH = LOCAL_MODEL_DIR
    logger.info(f"Using locally cached model at {LOCAL_MODEL_DIR}")
else:
    HF_MODEL_PATH = "Heem2/Deepfake-audio-detection"
    logger.info(f"Using model ID: {HF_MODEL_PATH}")
    
    # Try to download and cache the model locally for future use
    try:
        logger.info("Attempting to download and cache model for future use...")
        # Download and save to our custom cache directory
        Wav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_PATH, cache_dir=LOCAL_MODEL_DIR)
        AutoModelForAudioClassification.from_pretrained(HF_MODEL_PATH, cache_dir=LOCAL_MODEL_DIR)
        logger.info(f"✅ Successfully cached model to {LOCAL_MODEL_DIR}")
    except Exception as e:
        logger.warning(f"⚠️ Could not cache model: {str(e)}")

# Second model: WavLM-Large for deepfake detection
SECOND_MODEL_ID = "microsoft/wavlm-large"
LOCAL_MODEL_DIR_2 = os.path.join(PROJECT_ROOT, "hf_model_cache_2")
os.makedirs(LOCAL_MODEL_DIR_2, exist_ok=True)

try:
    logger.info(f"Loading primary Hugging Face model from {HF_MODEL_PATH}...")
    hf_processor = Wav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_PATH)
    hf_model = AutoModelForAudioClassification.from_pretrained(HF_MODEL_PATH)
    hf_model.eval()
    logger.info(f"✅ Successfully loaded primary Hugging Face model locally")
    
    logger.info(f"Loading WavLM model from {SECOND_MODEL_ID}...")
    try:
        # Load WavLM model and processor
        wavlm_processor = AutoFeatureExtractor.from_pretrained(SECOND_MODEL_ID)
        wavlm_model = WavLMModel.from_pretrained(SECOND_MODEL_ID)
        wavlm_model.eval()
        logger.info(f"✅ Successfully loaded WavLM model")
        use_second_model = True
    except Exception as e:
        logger.warning(f"⚠️ Could not load WavLM model: {str(e)}")
        use_second_model = False
    
    use_local_hf_model = True
except Exception as e:
    logger.error(f"❌ Failed to load Hugging Face models locally: {str(e)}")
    logger.info("Falling back to Hugging Face API")
    use_local_hf_model = False
    use_second_model = False

# Initialize HF connection status (for API fallback)
hf_api_ready = None

# Function to run local inference with Hugging Face model
def run_local_hf_inference(wav_path: str) -> Tuple[Optional[str], Optional[float]]:
    """Run local inference using Hugging Face audio classification model."""
    try:
        import librosa
        
        # Load audio with 10 seconds for better detection
        max_duration = 10  # seconds
        waveform, sr = librosa.load(wav_path, sr=16000, mono=True, duration=max_duration)
        
        # Process audio with the model's processor
        inputs = hf_processor(waveform, sampling_rate=16000, return_tensors="pt")
        
        # Run model inference
        with torch.no_grad():
            logits = hf_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_idx = torch.argmax(probs).item()
            result = hf_model.config.id2label[predicted_idx]
            confidence = float(probs[predicted_idx])
        
        # Map model output to expected format ("AI" or "Human")
        logger.info(f"✅ Primary HF model result: {result} with confidence {confidence:.3f}")
        
        # Standardize the result label to match expected format
        standardized_result = "Human" if "human" in result.lower() else "AI"
        logger.info(f"Standardized result: {standardized_result}")
        
        # Run second model if available
        second_result = None
        second_confidence = None
        if use_second_model:
            second_result, second_confidence = run_second_model_inference(wav_path)
            if second_result and second_confidence is not None:
                logger.info(f"Second model result: {second_result} with confidence {second_confidence:.3f}")
            
            # If both models agree it's AI, increase confidence
            if standardized_result == "AI" and second_result == "AI":
                logger.info("Both models agree this is AI-generated audio")
                confidence = max(confidence, 0.95)  # Increase confidence when both agree it's AI
            # If models disagree and second model says it's AI with high confidence
            elif standardized_result == "Human" and second_result == "AI" and second_confidence > 0.8:
                logger.info("Second model detected AI with high confidence, overriding primary model")
                standardized_result = "AI"
                confidence = second_confidence
        
        return standardized_result, confidence
    except Exception as e:
        logger.error(f"❌ Local HF model inference failed: {str(e)}")
        traceback.print_exc()
        return None, None

# Function to run inference with WavLM model
def run_second_model_inference(wav_path: str) -> Tuple[Optional[str], Optional[float]]:
    """Run inference using WavLM model for deepfake detection."""
    try:
        import librosa
        import torch
        import numpy as np
        
        # Load audio with 10 seconds for better detection
        max_duration = 10  # seconds
        waveform, sr = librosa.load(wav_path, sr=16000, mono=True, duration=max_duration)
        
        # Process audio with WavLM processor
        inputs = wavlm_processor(waveform, sampling_rate=16000, return_tensors="pt")
        
        # Run WavLM model inference
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
            # Get the hidden states from the model
            hidden_states = outputs.last_hidden_state
            
            # Calculate statistics on the hidden states
            # Higher variance and entropy often indicate real human speech
            # Lower variance and more predictable patterns often indicate AI-generated speech
            hidden_mean = hidden_states.mean().item()
            hidden_std = hidden_states.std().item()
            hidden_max = hidden_states.max().item()
            hidden_min = hidden_states.min().item()
            
            # Calculate a synthetic score based on these statistics
            # This is a heuristic approach - in production you'd want to train a classifier on these features
            range_ratio = (hidden_max - hidden_min) / (hidden_std + 1e-6)
            entropy_estimate = hidden_std / (abs(hidden_mean) + 1e-6)
            
            # Normalize entropy estimate to a reasonable range (0-1)
            # For WavLM, entropy estimates can be very high for human speech
            normalized_entropy = min(1.0, entropy_estimate / 100.0)
            
            # Real human speech tends to have higher entropy and variability
            # AI speech tends to be more regular with lower entropy
            # Invert the normalized entropy to get AI score (higher = more likely AI)
            ai_score = 1.0 - normalized_entropy
            human_score = normalized_entropy
            
            logger.info(f"✅ WavLM analysis - mean: {hidden_mean:.3f}, std: {hidden_std:.3f}, range: {hidden_max-hidden_min:.3f}")
            logger.info(f"WavLM entropy estimate: {entropy_estimate:.3f}, normalized: {normalized_entropy:.3f}")
            logger.info(f"WavLM scores - Human: {human_score:.3f}, AI: {ai_score:.3f}")
            
            # Determine if it's AI or human based on our heuristic
            if ai_score > 0.65:  # Threshold can be adjusted based on testing
                standardized_result = "AI"
                confidence = ai_score
            else:
                standardized_result = "Human"
                confidence = human_score
                
            return standardized_result, confidence
    except Exception as e:
        logger.error(f"❌ WavLM model inference failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

# Modify the query_huggingface_api function to test connection if needed
def query_huggingface_api(audio_data: dict) -> Tuple[Optional[str], Optional[float]]:
    """Query Hugging Face API for deepfake detection.
    Returns a tuple of (result, confidence) where result is either "AI" or "Human".
    Returns (None, None) if the API call fails.
    """
    global hf_api_ready
    
    # If we haven't tested the connection yet, do it now
    if hf_api_ready is None:
        hf_api_ready = test_hf_connection()
    
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    # Ensure we have a token
    if not HF_TOKEN:
        logger.error("❌ HF_API_TOKEN not set. Cannot query Hugging Face API.")
        return None, None
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Log audio payload size
    audio_size = len(audio_data.get("inputs", "")) if isinstance(audio_data.get("inputs"), str) else 0
    logger.info(f"Audio payload size: {audio_size} bytes")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"🎯 Attempt {attempt}/{MAX_RETRIES}: Querying Hugging Face API...")
            
            # Make request
            response = requests.post(HF_API_URL, headers=headers, json=audio_data)
            
            # Log response
            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            if response.status_code == 200:
                # Parse response
                result = response.json()
                logger.info(f"Parsed response: {result}")
                
                # Extract scores from the format [{"label": "AIVoice", "score": 0.633}, {"label": "HumanVoice", "score": 0.367}]
                ai_score = 0.0
                human_score = 0.0
                
                for item in result:
                    label = item.get("label", "").lower()
                    score = item.get("score", 0.0)
                    
                    if "ai" in label or "fake" in label:
                        ai_score = score
                    elif "human" in label or "real" in label:
                        human_score = score
                
                # Determine final result based on highest score
                if ai_score > human_score:
                    final_result = "AI"
                    confidence = ai_score
                else:
                    final_result = "Human"
                    confidence = human_score
                
                # Log the final result with rounded confidence for readability
                logger.info(f"✅ Final result: {final_result} with confidence {round(confidence*100, 1)}%")
                return final_result, confidence
                
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

@app.post("/api/predict")
async def predict(file: UploadFile):
    """Analyze audio file using Hugging Face API and VoiceGuard model."""
    try:
        # Save uploaded file
        input_path = os.path.join(TEMP_DIR, "input_audio")
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Convert to WAV format
        wav_path = os.path.join(TEMP_DIR, "audio_16k.wav")
        if not convert_to_wav(input_path, wav_path):
            raise HTTPException(status_code=500, detail="Failed to convert audio format")
        
        # Use local Hugging Face model if available, otherwise fall back to API
        if use_local_hf_model:
            # Run local Hugging Face inference
            logger.info("Using local Hugging Face model for inference")
            hf_result, hf_confidence = run_local_hf_inference(wav_path)
        else:
            # Process audio for Hugging Face API
            logger.info("Using Hugging Face API for inference")
            hf_b64 = prepare_audio_for_huggingface(wav_path)
            hf_payload = {"inputs": hf_b64}
            hf_result, hf_confidence = query_huggingface_api(hf_payload)
        
        # If Hugging Face inference fails (both local and API), return error
        if hf_result is None or hf_confidence is None:
            logger.error("Hugging Face model unavailable - returning 503")
            raise HTTPException(
                status_code=503,
                detail={"error": "HF model unavailable"}
            )
        
        # Process audio for VoiceGuard if available (secondary path)
        vg_genuine = None
        vg_spoof = None
        
        # Try to load the VoiceGuard model if it's not already loaded
        model = load_voiceguard_model()
        if model:
            try:
                audio_tensor = prepare_audio_for_voiceguard(wav_path)
                vg_genuine, vg_spoof = predict_voiceguard(audio_tensor)
                logger.info(f"VoiceGuard scores - Genuine: {vg_genuine:.1f}%, Spoof: {vg_spoof:.1f}%")
                
                # Determine VoiceGuard result for internal validation
                vg_result = "Human" if vg_genuine > vg_spoof else "AI"
                vg_confidence = max(vg_genuine, vg_spoof) / 100.0
                logger.info(f"VoiceGuard result: {vg_result} with confidence {vg_confidence:.3f}")
                
                # Log agreement/disagreement between models with detailed scores
                if vg_result == hf_result:
                    logger.info(f"✅ Models AGREE: Both predict {hf_result}")
                else:
                    logger.warning(f"⚠️ Models DISAGREE: HF={hf_result} ({hf_confidence:.3f}), VG={vg_result} ({vg_confidence:.3f})")
                
                # Log raw scores from both models for better diagnostics
                logger.info(f"Raw scores - HF: {hf_result}={hf_confidence:.3f}, VG: Human={vg_genuine/100:.3f}, AI={vg_spoof/100:.3f}")
            except Exception as e:
                logger.error(f"VoiceGuard prediction failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue without VoiceGuard
        
        # Create simplified response - only result and confidence from Hugging Face
        # This keeps the API contract simple while preserving internal dual-path scoring
        response = {
            "result": hf_result,
            "confidence": hf_confidence
        }
        
        # Compute combined score
        if vg_genuine is not None and vg_spoof is not None:
            # If we're using the second model and have valid results, give it more weight (60%)
            if use_second_model and second_result is not None and second_confidence is not None:
                # 20% weight to Hugging Face, 20% weight to VoiceGuard, 60% to WavLM model
                hf_genuine = (hf_confidence if hf_result == "Human" else 1 - hf_confidence) * 100
                second_genuine = (second_confidence if second_result == "Human" else 1 - second_confidence) * 100
                combined_genuine = hf_genuine * 0.2 + vg_genuine * 0.2 + second_genuine * 0.6
                logger.info(f"Using three-model weighted score: HF={hf_genuine:.1f}% (20%), VG={vg_genuine:.1f}% (20%), WavLM={second_genuine:.1f}% (60%)")
                logger.info(f"Final combined score: {combined_genuine:.1f}% genuine, {100-combined_genuine:.1f}% AI")
            else:
                # Original 80/20 weighting if second model isn't available or didn't detect AI
                combined_genuine = (hf_confidence if hf_result == "Human" else 1 - hf_confidence) * 100 * 0.8 + vg_genuine * 0.2
                
            combined_spoof = 100.0 - combined_genuine
            final_result = "Human" if combined_genuine > combined_spoof else "AI"
            
            logger.info(f"✅ Scores - Genuine: {combined_genuine:.1f}%, Spoof: {combined_spoof:.1f}%")

            # Add to response
            response["combined_score"] = {
                "genuine_score": round(combined_genuine / 100, 4),
                "spoof_score": round(combined_spoof / 100, 4),
                "result": final_result
            }
        
        # Log the response
        print("Returning API response:", response)
        
        # Store in recent analyses for debugging (includes both models when available)
        recent_analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "huggingface": {
                "result": hf_result,
                "confidence": hf_confidence
            }
        }
        
        # Add VoiceGuard results if available
        if vg_genuine is not None and vg_spoof is not None:
            recent_analysis["voiceguard"] = {
                "genuine": float(vg_genuine) / 100.0,  # Convert to 0-1 scale
                "spoof": float(vg_spoof) / 100.0,
                "result": vg_result,
                "confidence": vg_confidence
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

def prepare_audio_for_voiceguard(audio_path: str) -> torch.Tensor:
    """Process audio for VoiceGuard model - returns MFCC features."""
    try:
        import librosa
        
        # Load and resample audio - use 10 seconds for better detection
        max_duration = 10  # seconds
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
        
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

if __name__ == "__main__":
    # Find an available port if 8009 is in use
    port = 8009
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', 0))
            port = s.getsockname()[1]
        logger.info(f"Port 8009 is in use, using fallback port {port}")

    import uvicorn
    logger.info(f"🚀 Starting VoiceGuard server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
