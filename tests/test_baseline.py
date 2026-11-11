from hashlib import sha256
from pathlib import Path


def test_baseline_matches_published_yaml_byte_for_byte():
    path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    assert sha256(path.read_bytes()).hexdigest() == (
        "a335ac5d4fda88ef53b4c04788b58966e2a72c7540abb5465a22d4edac2e6f9b"
    )
