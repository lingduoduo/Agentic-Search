"""Serving-side model layer: intent routing and LLM serving.

The training-side generation manager and its tensor helper moved to
``src/training/grpo/``. This package no longer depends on ``src.training`` at
all -- it is what runs when a request comes in, not what runs to train.
"""

from .intent import IntentExample as IntentExample
from .intent import load_intent_examples as load_intent_examples
from .intent.metrics import IntentPredictionRecord as IntentPredictionRecord
