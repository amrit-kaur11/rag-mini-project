from scripts.prepare_data import chunk_text

def test_determinism():
    text = "A" * 4000
    c1 = chunk_text(text)
    c2 = chunk_text(text)
    assert c1 == c2