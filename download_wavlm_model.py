import os
import logging
from transformers import WavLMModel, AutoFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define the model ID and cache directory
MODEL_ID = "microsoft/wavlm-large"
CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_model_cache_2")

def main():
    # No login required for public models
    logger.info("Proceeding with downloading public model without authentication...")
    # Set environment variable to disable warnings about symlinks
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download and cache the model
    logger.info(f"Downloading WavLM model {MODEL_ID}...")
    try:
        # Download the feature extractor
        logger.info("Downloading feature extractor...")
        AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        logger.info("✅ Successfully downloaded feature extractor")
        
        # Download the model
        logger.info("Downloading model...")
        WavLMModel.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        logger.info("✅ Successfully downloaded model")
        
        logger.info(f"✅ Successfully downloaded and cached WavLM model to {CACHE_DIR}")
    except Exception as e:
        logger.error(f"❌ Failed to download WavLM model: {str(e)}")
        raise

if __name__ == "__main__":
    main()
