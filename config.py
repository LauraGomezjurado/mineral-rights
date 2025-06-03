"""Configuration settings for the classification system"""

# API Configuration
ANTHROPIC_API_KEY="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"

# Classification Parameters
DEFAULT_MAX_SAMPLES = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_TEMPERATURE_RANGE = (0.5, 1.0)

# Feature Weights (for confidence scoring)
FEATURE_WEIGHTS = {
    'sentence_count': 0.15,
    'trigger_word_presence': 0.20,
    'lexical_consistency': 0.15,
    'format_validity': 0.20,
    'answer_certainty': 0.20,
    'past_agreement': 0.10
}

# Output Configuration
RESULTS_DIR = "classification_results"
BATCH_RESULTS_DIR = "batch_results"
