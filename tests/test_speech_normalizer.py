from voice.speech_normalizer import (
    SpeechTextNormalizer,
)


def test_english_inr_uses_indian_units():
    n = SpeechTextNormalizer()

    assert (
        n.normalize(
            "Your balance is 100000 INR.",
            "english",
        )
        == "Your balance is one lakh rupees."
    )

    assert (
        n.normalize(
            "Your balance is 12500000 INR.",
            "english",
        )
        == "Your balance is one crore twenty five lakh rupees."
    )


def test_hindi_inr_uses_indian_units():
    n = SpeechTextNormalizer()

    assert (
        n.normalize(
            "आपकी राशि 100000 INR है।",
            "hindi",
        )
        == "आपकी राशि एक लाख रुपये है।"
    )


def test_hindi_first_person_is_feminine():
    n = SpeechTextNormalizer()

    assert (
        n.normalize(
            "मैं समझता हूँ और मैं यह नहीं कर सकता।",
            "hindi",
        )
        == "मैं समझती हूँ और मैं यह नहीं कर सकती।"
    )
