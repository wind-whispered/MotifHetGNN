"""Feature-provenance constants for the Information Sciences experiments."""

PRIMARY_FULL_MATCH_FEATURE_REVISION = "no_assist_pass_intensity_v2"
ASSIST_PROXY_FEATURE_REVISION = "assist_proxy_pass_intensity_v2"


def full_match_feature_revision(use_assist_flags: bool) -> str:
    return (ASSIST_PROXY_FEATURE_REVISION if use_assist_flags
            else PRIMARY_FULL_MATCH_FEATURE_REVISION)
