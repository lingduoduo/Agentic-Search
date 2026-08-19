"""Models: how they are produced, and how they are served.

Two halves, split by when the training happens:

* ``pre_training`` -- models built offline, before any request. Today that is
  ``intents``, the nearest-canonical-example router, whose index is built by
  ``python -m src.model.pre_training.intents.cli build`` and then loaded
  read-only on the request path.
* ``post_training`` -- the methods that adapt a language model after it exists:
  ``sft``, ``dpo``, the ``grpo`` stack and the ``ppo`` base layer it is built on,
  plus the shared ``data``/``reward`` modules, the ``qlearning`` demo and the
  ``eval`` harnesses. None of it runs on the request path.

``serving.py`` is neither: it is the serving backend that loads and runs a model,
whatever produced it.
"""

from .pre_training.intents import IntentExample as IntentExample
from .pre_training.intents import load_intent_examples as load_intent_examples
from .pre_training.intents.metrics import (
    IntentPredictionRecord as IntentPredictionRecord,
)
