"""Direct Preference Optimization: train on preference pairs, no reward model.

Two modules, mirroring how ``sft`` and ``rl`` are shaped:

* ``data``    — ``PreferenceExample`` and the JSONL loader. Standard library only,
                so a dataset can be validated without torch installed.
* ``trainer`` — ``DPOConfig`` and ``DPOTrainer``. Imports torch.

Only ``data`` is re-exported here. ``trainer`` is reached by module path so that
importing a preference dataset does not drag torch in — the same package-
``__init__``-as-import-gate trap that broke CI in #536.
"""

from .data import PreferenceExample as PreferenceExample
from .data import load_preference_pairs as load_preference_pairs
