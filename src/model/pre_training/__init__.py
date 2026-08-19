"""Models trained offline, before any request is served.

Currently one: ``intents``, the nearest-canonical-example router that decides
whether a request goes to chat, search or tools. Its index is built ahead of time
by ``python -m src.model.pre_training.intents.cli build`` and then loaded
read-only on the request path.

The counterpart is ``post_training``, which holds the methods that adapt the
*language model* after it exists -- SFT, DPO and the GRPO stack.
"""
