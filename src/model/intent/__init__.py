"""Intent routing: the taxonomy, the encoder, the index, and its offline tools.

Five modules, layered so that each one only imports from those above it:

* ``model``      — taxonomy, encoder, and ``IntentIndex``. The serving path.
* ``data``       — validated loaders, and the build that turns them into an index.
* ``metrics``    — scoring over prediction records.
* ``evaluation`` — the tuning/test split and the offline harness.
* ``cli``        — ``seed`` / ``build`` / ``evaluate`` entry points.

Only ``model`` and ``data`` are re-exported here. ``metrics`` and ``evaluation``
are import-only-on-demand on purpose: ``metrics`` pulls in scikit-learn, and
nothing on the request path should pay for that. Import them by module::

    from src.model.intent.metrics import IntentPredictionRecord
    from src.model.intent.evaluation import run_index_evaluation
"""

from __future__ import annotations

from .data import (
    IntentEvalQuery as IntentEvalQuery,
)
from .data import (
    IntentExample as IntentExample,
)
from .data import (
    build_index as build_index,
)
from .data import (
    fingerprint as fingerprint,
)
from .data import (
    load_canonical_examples as load_canonical_examples,
)
from .data import (
    load_intent_eval_queries as load_intent_eval_queries,
)
from .data import (
    load_intent_examples as load_intent_examples,
)
from .data import (
    load_out_of_scope_probes as load_out_of_scope_probes,
)
from .model import (
    ACTION_MODULES as ACTION_MODULES,
)
from .model import (
    DEFAULT_ENCODER as DEFAULT_ENCODER,
)
from .model import (
    INDEX_FILENAME as INDEX_FILENAME,
)
from .model import (
    INTENT_LABELS as INTENT_LABELS,
)
from .model import (
    MIN_MODULE_SUPPORT as MIN_MODULE_SUPPORT,
)
from .model import (
    SEMANTIC_MODULES as SEMANTIC_MODULES,
)
from .model import (
    TOP_K as TOP_K,
)
from .model import (
    CanonicalExample as CanonicalExample,
)
from .model import (
    IntentIndex as IntentIndex,
)
from .model import (
    KnnDecision as KnnDecision,
)
from .model import (
    ModuleSpec as ModuleSpec,
)
from .model import (
    encode_texts as encode_texts,
)
from .model import (
    modules_for_route as modules_for_route,
)
from .model import (
    prefix_for as prefix_for,
)
from .model import (
    route_of_module as route_of_module,
)
from .model import (
    validate_modules as validate_modules,
)
