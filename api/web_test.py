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
from fastapi import FastAPI, UploadFile, HTTPException, File
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

# Import routers
from api.analytics_handler import router as analytics_router

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

# Include routers
app.include_router(analytics_router, prefix="/api/analytics")

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
            
            # Calculate additional features that might help with AI detection
            # 1. Spectral flatness - AI voices often have more uniform spectral distribution
            spectral_range = hidden_max - hidden_min
            spectral_flatness = hidden_std / (spectral_range + 1e-6)
            
            # 2. Pattern regularity - AI voices often have more regular patterns
            # Look at the autocorrelation of the hidden states
            hidden_flat = hidden_states.flatten()
            hidden_mean = hidden_flat.mean().item()
            hidden_centered = hidden_flat - hidden_mean
            
            # Calculate a pattern regularity score
            pattern_score = 0.0
            if hidden_centered.shape[0] > 1:
                # Use the standard deviation as a simple measure of regularity
                # More regular patterns (AI) have lower std dev relative to their range
                pattern_score = 1.0 - min(1.0, (hidden_std / (spectral_range + 1e-6) * 5.0))
            
            # Normalize entropy estimate to a reasonable range (0-1)
            # For WavLM, entropy estimates can be very high for human speech
            normalized_entropy = min(1.0, entropy_estimate / 100.0)
            
            # Calculate additional features specifically for robo-call detection
            
            # 1. Prosody variation - human speech has more natural prosody variation
            # Calculate variance across different dimensions of hidden states
            dim_variances = torch.var(hidden_states, dim=1).mean().item()
            normalized_dim_variance = min(1.0, dim_variances * 10.0)  # Normalize to 0-1 range
            
            # 2. Temporal consistency - AI speech often has more consistent timing
            # Calculate the variance of differences between adjacent time steps
            if hidden_states.shape[1] > 1:
                temporal_diffs = torch.diff(hidden_states, dim=1)
                temporal_variance = torch.var(temporal_diffs).item()
                normalized_temporal_variance = min(1.0, temporal_variance * 10.0)
            else:
                normalized_temporal_variance = 0.5  # Default if not enough time steps
            
            # 3. Formant analysis - AI speech often has less natural formant transitions
            # Use the pattern score as a proxy for formant naturalness
            formant_naturalness = pattern_score
            
            # Combine multiple factors for AI detection with weights optimized for robo-calls:
            # - Lower entropy suggests AI (weight: 30%)
            # - Higher pattern regularity suggests AI (weight: 20%)
            # - Lower prosody variation suggests AI (weight: 25%)
            # - Lower temporal variance suggests AI (weight: 25%)
            entropy_factor = 1.0 - normalized_entropy
            pattern_factor = pattern_score
            prosody_factor = 1.0 - normalized_dim_variance
            temporal_factor = 1.0 - normalized_temporal_variance
            
            # Calculate final scores with a more balanced approach
            # Normalize factors to prevent overweighting any single feature
            entropy_factor = min(entropy_factor, 0.9)  # Cap entropy factor to prevent overweighting
            combined_factors = (entropy_factor + pattern_factor + prosody_factor + temporal_factor) / 4
            
            # More balanced scoring formula with reduced bias
            human_score = 0.5 - (0.4 * combined_factors)  # Reduced from 0.5 to 0.4 multiplier
            ai_score = 1.0 - human_score
            
            # Apply a very small bias only for extremely regular patterns (clear AI indicator)
            if pattern_factor > 0.8 and temporal_factor > 0.7:
                logger.warning("Detected highly regular patterns - possible robo-call")
                ai_bias = 0.03  # Reduced bias from 0.05 to 0.03
                human_score = max(0.0, human_score - ai_bias)
                ai_score = min(1.0, ai_score + ai_bias)
            
            logger.info(f"✅ WavLM analysis - mean: {hidden_mean:.3f}, std: {hidden_std:.3f}, range: {spectral_range:.3f}")
            logger.info(f"WavLM entropy: {entropy_estimate:.3f}, pattern: {pattern_score:.3f}, prosody: {normalized_dim_variance:.3f}, temporal: {normalized_temporal_variance:.3f}")
            logger.info(f"WavLM scores with robo-call bias - Human: {human_score:.3f}, AI: {ai_score:.3f}")
            
            # Very low threshold to be extremely sensitive to robo-calls
            # We'd rather have false positives than miss AI-generated voices
            if ai_score > 0.25:  # Very aggressive threshold
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

@app.get("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Run deepfake detection on an audio file."""
    try:
        # Save the file to a temporary location
        temp_file = os.path.join(TEMP_DIR, file.filename)
        with open(temp_file, 'wb') as f:
            f.write(file.file.read())
        
        # Convert to WAV format if necessary
        if not file.filename.endswith('.wav'):
            wav_file = os.path.join(TEMP_DIR, 'audio_16k.wav')
            if not convert_to_wav(temp_file, wav_file):
                raise Exception("Failed to convert audio to WAV format")
            temp_file = wav_file
        
        # Run local inference
        hf_result, hf_confidence = run_local_hf_inference(temp_file)
        
        # Run second model if available
        second_result = None
        second_confidence = None
        if use_second_model:
            second_result, second_confidence = run_second_model_inference(temp_file)
            if second_result and second_confidence is not None:
                logger.info(f"Second model result: {second_result} with confidence {second_confidence:.3f}")
            
            # If both models agree it's AI, increase confidence
            if hf_result == "AI" and second_result == "AI":
                logger.info("Both models agree this is AI-generated audio")
                hf_confidence = max(hf_confidence, 0.95)  # Increase confidence when both agree it's AI
            # If models disagree and second model says it's AI with high confidence
            elif hf_result == "Human" and second_result == "AI" and second_confidence > 0.8:
                logger.info("Second model detected AI with high confidence, overriding primary model")
                hf_result = "AI"
                hf_confidence = second_confidence
        
        # Create response with all model results
        response = {
            "result": hf_result,
            "confidence": hf_confidence,
            "models": {
                "huggingface": {
                    "result": hf_result,
                    "confidence": hf_confidence
                }
            }
        }
        
        # Add VoiceGuard results if available
        if vg_genuine is not None and vg_spoof is not None:
            response["models"]["voiceguard"] = {
                "result": vg_result_label,
                "confidence": vg_confidence,
                "genuine": vg_genuine / 100.0,
                "spoof": vg_spoof / 100.0
            }
            
        # Add WavLM results if available
        if second_result and second_confidence is not None:
            response["models"]["wavlm"] = {
                "result": second_result,
                "confidence": second_confidence
            }
        
        # Compute combined score
        if vg_genuine is not None and vg_spoof is not None:
            try:
                # Calculate human confidence scores for each model (0-100 scale)
                hf_human_score = (hf_confidence if hf_result == "Human" else 1 - hf_confidence) * 100
                vg_human_score = vg_genuine  # Already on 0-100 scale
                
                # If WavLM model results are available, use the three-model weighted approach
                if second_result is not None and second_confidence is not None:
                    # Convert WavLM confidence to human score (0-100 scale)
                    wavlm_human_score = (second_confidence if second_result == "Human" else 1 - second_confidence) * 100
                    
                    # Apply weights: 20% HF, 20% VG, 60% WavLM
                    combined_genuine = (hf_human_score * 0.2) + (vg_human_score * 0.2) + (wavlm_human_score * 0.6)
                    
                    logger.info(f"Using three-model weighted score: HF={hf_human_score:.1f}% (20%), "
                               f"VG={vg_human_score:.1f}% (20%), WavLM={wavlm_human_score:.1f}% (60%)")
                    logger.info(f"Final combined score: {combined_genuine:.1f}% genuine, {100-combined_genuine:.1f}% AI")
                    
                    # Only apply WavLM bias if both WavLM and HF models agree on AI
                    # This helps prevent false positives on human voices
                    if wavlm_human_score < 40 and hf_human_score < 40 and combined_genuine > 50:
                        logger.warning("Both WavLM and HF suggest AI - applying careful bias adjustment")
                        # Apply a very conservative adjustment to avoid misclassifying humans
                        adjustment_strength = 0.15
                        combined_genuine = (combined_genuine * (1 - adjustment_strength)) + (wavlm_human_score * adjustment_strength)
                        logger.info(f"Adjusted score with conservative bias: {combined_genuine:.1f}% genuine")
                else:
                    # Original 80/20 weighting if WavLM model isn't available
                    combined_genuine = (hf_human_score * 0.8) + (vg_human_score * 0.2)
                    logger.info(f"Using two-model weighted score: HF={hf_human_score:.1f}% (80%), VG={vg_human_score:.1f}% (20%)")
            except Exception as e:
                logger.error(f"Error in combined score calculation: {str(e)}")
                logger.error(traceback.format_exc())
                # Fallback to simple average if there's an error
                hf_human_score = (hf_confidence if hf_result == "Human" else 1 - hf_confidence) * 100
                combined_genuine = (hf_human_score * 0.5) + (vg_genuine * 0.5)
                logger.info(f"Using fallback score calculation due to error: {combined_genuine:.1f}%")
                
            combined_spoof = 100.0 - combined_genuine
            
            # Make final decision with special handling for robo-calls
            # Only adjust threshold if BOTH WavLM and HF models agree on AI
            if second_result == "AI" and second_confidence > 0.5 and hf_result == "AI" and hf_confidence > 0.5:
                # Use a modest threshold adjustment when both models agree on AI
                threshold = 48.0  # Slightly favor AI detection when both models agree
                final_result = "Human" if combined_genuine > threshold else "AI"
                logger.info(f"Using adjusted threshold ({threshold}%) - both WavLM and HF models indicate AI")
            else:
                # More conservative threshold (52%) to avoid false positives on human voices
                threshold = 52.0
                final_result = "Human" if combined_genuine > threshold else "AI"
                logger.info(f"Using conservative threshold ({threshold}%) to avoid false positives")
            
            logger.info(f"✅ Scores - Genuine: {combined_genuine:.1f}%, Spoof: {combined_spoof:.1f}%")

            # Update the main response with the combined result
            response["result"] = final_result
            response["confidence"] = round(max(combined_genuine, combined_spoof) / 100, 4)
            
            # Add detailed combined scores to response
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

