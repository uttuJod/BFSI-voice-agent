from voice.language_detect import LanguageDetector, majority_word_language


def test_devanagari_is_hindi():
    d = LanguageDetector().detect("मेरा बकाया कितना है?")
    assert d.output_language == "hindi"
    assert d.reason == "devanagari_script"


def test_hinglish_resolves_to_hindi_output():
    d = LanguageDetector().detect("Order ORD124 cancel karna hai")
    assert d.detected == "hinglish"
    assert d.output_language == "hindi"


def test_plain_english_stays_english():
    for text in [
        "What is the refund policy?",
        "Call the manager",
        "Give me the main account statement",
        "Do you deliver on Sunday?",
        "I paid yesterday, please check.",
        "Transfer me to a human agent.",
    ]:
        assert LanguageDetector().detect(text).output_language == "english", text


def test_single_hit_in_long_english_sentence_is_not_hindi():
    # "se" alone should not flip a long English sentence.
    d = LanguageDetector().detect("I booked the ticket from Delhi se and it worked fine overall")
    assert d.output_language == "english"


def test_short_utterance_needs_one_hit():
    assert LanguageDetector().detect("Balance batao").output_language == "hindi"
    assert LanguageDetector().detect("Yes please").output_language == "english"


def test_stt_hint_breaks_tie_only_with_weak_evidence():
    det = LanguageDetector()
    # No markers at all: hint is ignored.
    assert det.detect("Check my order status now", stt_hint="hi").output_language == "english"
    # One weak marker plus a Hindi hint: accept Hindi.
    assert det.detect("Check my order status abhi please", stt_hint="hi").output_language == "hindi"


def test_empty_text_defaults_to_hint_or_english():
    det = LanguageDetector()
    assert det.detect("").output_language == "english"
    assert det.detect("", stt_hint="hi").output_language == "hindi"


def test_majority_word_language_prefers_hindi_on_tie():
    words = [{"language": "en"}, {"language": "hi"}]
    assert majority_word_language(words) == "hindi"
    assert majority_word_language([]) is None
    assert majority_word_language([{"language": "en"}, {"language": "en"}]) == "english"
