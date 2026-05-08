from flickr_captioning.text import END_TOKEN, START_TOKEN, Vocabulary, clean_caption, prepare_caption


def test_clean_caption_normalizes_text() -> None:
    assert clean_caption("A Dog, jumping!") == ["a", "dog", "jumping"]


def test_prepare_caption_adds_boundaries() -> None:
    assert prepare_caption("A dog.") == [START_TOKEN, "a", "dog", END_TOKEN]


def test_vocabulary_encodes_and_decodes() -> None:
    vocab = Vocabulary(min_freq=1)
    vocab.fit(["A dog jumps", "A dog runs"])

    encoded = vocab.encode("A dog jumps")
    assert encoded[0] == vocab.start_idx
    assert encoded[-1] == vocab.end_idx
    assert vocab.decode(encoded) == ["a", "dog", "jumps"]
