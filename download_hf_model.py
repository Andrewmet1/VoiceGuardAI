import os
import logging
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_model_cache")
MODEL_ID = "Heem2/Deepfake-audio-detection"

# Hugging Face token (same as in web_test.py)
HF_TOKEN = "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"

# Log in to Hugging Face
login(token=HF_TOKEN)
logger.info("Logged in to Hugging Face with token")


# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Created cache directory at {CACHE_DIR}")

# Download and save the model using the recommended approach
try:
    # Download feature extractor (processor)
    logger.info(f"Downloading feature extractor from {MODEL_ID}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    logger.info("Feature extractor downloaded successfully")
    
    # Download model
    logger.info(f"Downloading model from {MODEL_ID}...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
    logger.info("Model downloaded successfully")
    
    # Save to our cache directory
    logger.info(f"Saving feature extractor to {CACHE_DIR}...")
    processor.save_pretrained(CACHE_DIR)
    
    logger.info(f"Saving model to {CACHE_DIR}...")
    model.save_pretrained(CACHE_DIR)
    
    logger.info("✅ Model and feature extractor saved successfully!")
    logger.info(f"Model files are now available at: {CACHE_DIR}")
    
    # Test loading from the saved directory to verify it works
    logger.info("Testing loading from saved directory...")
    try:
        test_processor = Wav2Vec2FeatureExtractor.from_pretrained(CACHE_DIR)
        test_model = AutoModelForAudioClassification.from_pretrained(CACHE_DIR)
        logger.info("✅ Successfully loaded model from cache directory!")
    except Exception as e:
        logger.warning(f"Could not load from cache for verification: {str(e)}")
    
except Exception as e:
    logger.error(f"❌ Error downloading model: {str(e)}")
    raise

print("\nModel download complete! You can now use the local model in your API.")
