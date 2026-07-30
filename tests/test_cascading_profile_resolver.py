from pathlib import Path

from RAG_Agent.infrastructure.ingestion.cascading_profile_resolver import CascadingProfileResolver


def test_cascading_resolver_oran():
    profile = CascadingProfileResolver().resolve(
        Path("data/O-RAN.WG1.TS.Use-Cases-Detailed-Specification-R005-v19.00.pdf")
    )
    assert profile.rules.profile_id == "oran_default"
    assert profile.identity.metadata["group"] == "WG1"


def test_cascading_resolver_sufg():
    profile = CascadingProfileResolver().resolve(Path("data/O-RAN.SuFG.CE-v01.00.pdf"))
    assert profile.rules.profile_id == "oran_default"
    assert profile.identity.metadata["group"] == "SuFG"


def test_cascading_resolver_default():
    profile = CascadingProfileResolver().resolve(Path("data/some-other-doc.pdf"))
    assert profile.rules.profile_id == "default"
