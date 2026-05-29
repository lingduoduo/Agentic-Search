"""Sampled test — requires ee.onyx.* modules not present in this repo. Skipped."""

import pytest

pytestmark = pytest.mark.skip(reason="requires ee.onyx external package")
