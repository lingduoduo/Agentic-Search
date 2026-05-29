"""Billing proxy tests.

The sampled Onyx version had a tenants/proxy.py module for cloud data-plane
forwarding.  This deployment is single-tenant self-hosted; the proxy layer is
not implemented.  Billing endpoints proxy directly via the service layer which
is tested in test_billing_service.py.

This file is a placeholder to document the coverage gap and serves as a
reminder that cloud-specific proxy tests belong here when multi-tenancy is added.
"""
