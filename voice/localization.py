from __future__ import annotations

import logging
import re
import threading

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)


class ResponseLocalizer:
    """
    Enforces OUTPUT language only.

    Input language is irrelevant here.

    English:
        return grounded response unchanged.

    Hindi:
        convert grounded response to Hindi AFTER router / guards / RAG /
        tools have finished.

    Safety-critical common responses are translated deterministically.
    Longer grounded RAG responses are translated sentence-by-sentence
    to reduce truncation and improve readability.
    """

    def __init__(
        self,
        *,
        hindi_model_name: str = (
            "Helsinki-NLP/opus-mt-en-hi"
        ),
        device: str = "cpu",
        eager_load: bool = True,
    ) -> None:
        self._hindi_model_name = (
            hindi_model_name
        )

        self._device = torch.device(
            device
        )

        self._tokenizer = None
        self._model = None

        self._load_lock = (
            threading.Lock()
        )

        self._translate_lock = (
            threading.Lock()
        )

        if eager_load:
            self.preload()

    def preload(self) -> None:
        """
        Load translator at app startup.

        First install/run may download the model once.
        Later runs use the Hugging Face cache.
        """
        self._ensure_hindi_model()

    def _ensure_hindi_model(
        self,
    ) -> None:
        if (
            self._tokenizer is not None
            and self._model is not None
        ):
            return

        with self._load_lock:
            if (
                self._tokenizer is not None
                and self._model is not None
            ):
                return

            logger.info(
                "Loading EN→HI response translator | %s | device=%s",
                self._hindi_model_name,
                self._device,
            )

            tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    self._hindi_model_name
                )
            )

            model = (
                AutoModelForSeq2SeqLM
                .from_pretrained(
                    self._hindi_model_name
                )
            )

            model.to(
                self._device
            )

            model.eval()

            self._tokenizer = (
                tokenizer
            )

            self._model = (
                model
            )

            logger.info(
                "EN→HI response translator ready."
            )

    def localize(
        self,
        text: str,
        language: str,
    ) -> str:
        text = str(
            text
            or ""
        ).strip()

        if not text:
            return ""

        language = str(
            language
            or "english"
        ).strip().lower()

        if language == "english":
            return text

        if language != "hindi":
            raise ValueError(
                "Output language must be english or hindi."
            )

        deterministic = (
            self._deterministic_hindi(
                text
            )
        )

        if deterministic is not None:
            return deterministic

        return (
            self._translate_paragraph_to_hindi(
                text
            )
        )

    def _deterministic_hindi(
        self,
        text: str,
    ) -> str | None:
        """
        Known response patterns where exact, natural Hindi is safer
        than generic machine translation.
        """

        normalized = " ".join(
            text.split()
        ).strip()

        exact = {
            (
                "I don't have enough verified policy "
                "information to answer that safely."
            ): (
                "मेरे पास इसका सुरक्षित उत्तर देने के लिए "
                "पर्याप्त सत्यापित नीति जानकारी उपलब्ध नहीं है।"
            ),

            (
                "That request is outside this "
                "customer-support role."
            ): (
                "यह अनुरोध इस ग्राहक-सहायता सेवा के दायरे से बाहर है।"
            ),

            (
                "Please confirm the exact payment date "
                "before a response is given."
            ): (
                "कृपया सटीक भुगतान तिथि की पुष्टि करें।"
            ),

            (
                "The account status is active."
            ): (
                "आपके खाते की स्थिति सक्रिय है।"
            ),

            (
                "Please confirm the exact payment date "
                "before a commitment is recorded."
            ): (
                "कृपया भुगतान का वादा दर्ज करने से पहले "
                "सटीक भुगतान तिथि की पुष्टि करें।"
            ),

            (
                "Please provide an exact calendar date "
                "for the promise to pay."
            ): (
                "कृपया भुगतान के वादे के लिए सटीक कैलेंडर तिथि बताइए।"
            ),

            (
                "I recorded your reported payment for verification. "
                "It has not been treated as confirmed yet."
            ): (
                "मैंने आपके बताए गए भुगतान को सत्यापन के लिए दर्ज कर दिया है। "
                "अभी इसे पुष्टि किया हुआ भुगतान नहीं माना गया है।"
            ),

            (
                "I cannot disclose another customer's "
                "private account information."
            ): (
                "मैं किसी अन्य ग्राहक की निजी खाता जानकारी साझा नहीं कर सकती।"
            ),

            (
                "I can't disclose another customer's "
                "private account information."
            ): (
                "मैं किसी अन्य ग्राहक की निजी खाता जानकारी साझा नहीं कर सकती।"
            ),

            (
                "That action was already recorded in this session, "
                "so I did not create a duplicate."
            ): (
                "यह कार्रवाई इस सत्र में पहले ही दर्ज की जा चुकी है, "
                "इसलिए मैंने इसकी दूसरी प्रविष्टि नहीं बनाई।"
            ),
        }

        if normalized in exact:
            return exact[
                normalized
            ]

        balance = re.fullmatch(
            r"Your current outstanding balance is "
            r"([0-9]+(?:\.[0-9]+)?) "
            r"([A-Z]{3})\.",
            normalized,
        )

        if balance:
            amount = balance.group(1)
            currency = balance.group(2)

            return (
                "आपकी वर्तमान बकाया राशि "
                f"{amount} {currency} है।"
            )

        # Current collections-policy answer.
        grace_days = re.search(
            r"The grace period is "
            r"(\d+) days after the scheduled due date\.",
            normalized,
        )

        if (
            "Collections Policy v2"
            in normalized
            and grace_days
        ):
            days = grace_days.group(1)

            parts = [
                "उपलब्ध नीति के अनुसार,",
                (
                    "निर्धारित देय तिथि के बाद "
                    f"{days} दिनों का ग्रेस पीरियड है।"
                ),
            ]

            if (
                "supersedes the previous five-day "
                "grace-period rule"
                in normalized
            ):
                parts.append(
                    "यह नियम पहले के पाँच दिन वाले "
                    "ग्रेस पीरियड नियम की जगह लेता है।"
                )

            if (
                "standard collection escalation "
                "should not begin"
                in normalized
            ):
                parts.append(
                    "ग्रेस पीरियड के दौरान सामान्य "
                    "वसूली एस्केलेशन शुरू नहीं किया जाना चाहिए।"
                )

            promised = re.search(
                r"The promised date must be within "
                r"(\d+) calendar days",
                normalized,
            )

            if promised:
                limit = promised.group(1)

                parts.append(
                    "भुगतान का वादा किया गया दिन सामान्यतः "
                    f"{limit} कैलेंडर दिनों के भीतर होना चाहिए, "
                    "जब तक कि स्वीकृत वित्तीय-कठिनाई व्यवस्था लागू न हो।"
                )

            return " ".join(
                parts
            )

        lower = normalized.lower()

        if (
            ("safety" in lower or "privacy" in lower)
            and (
                "ignore" in lower
                or "override" in lower
                or "bypass" in lower
                or "cannot" in lower
                or "can't" in lower
            )
        ):
            return (
                "मैं सुरक्षा या गोपनीयता नियमों को अनदेखा या ओवरराइड "
                "नहीं कर सकती, और निजी ग्राहक जानकारी साझा नहीं कर सकती।"
            )

        if (
            "human support" in lower
            and (
                "review" in lower
                or "specialist" in lower
                or "requested" in lower
            )
        ):
            return (
                "मैंने मानव सहायता विशेषज्ञ से समीक्षा का अनुरोध कर दिया है।"
            )

        if (
            "will not assume" in lower
            and "payment" in lower
        ):
            return (
                "मैं समझती हूँ। मैं यह मानकर नहीं चलूँगी कि भुगतान हो चुका है।"
            )

        if (
            lower.startswith("please confirm")
            and "information" in lower
        ):
            return (
                "कृपया बताइए कि आपको कौन-सी सटीक जानकारी चाहिए।"
            )

        return None

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        text = " ".join(
            text.split()
        ).strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence
            in sentences
            if sentence.strip()
        ]

    def _translate_paragraph_to_hindi(
        self,
        text: str,
    ) -> str:
        sentences = (
            self._split_sentences(
                text
            )
        )

        if not sentences:
            return ""

        translated = []

        for sentence in sentences:
            translated.append(
                self._translate_sentence_to_hindi(
                    sentence
                )
            )

        result = " ".join(
            part
            for part
            in translated
            if part
        ).strip()

        if not result:
            raise RuntimeError(
                "Hindi translator returned an empty response."
            )

        return result

    def _translate_sentence_to_hindi(
        self,
        text: str,
    ) -> str:
        self._ensure_hindi_model()

        tokenizer = (
            self._tokenizer
        )

        model = (
            self._model
        )

        if (
            tokenizer is None
            or model is None
        ):
            raise RuntimeError(
                "Hindi translator failed to initialize."
            )

        # Make a few finance phrases easier for the general translator
        # without changing their factual meaning.
        prepared = (
            text
            .replace(
                "financial-hardship",
                "financial hardship",
            )
            .replace(
                "collection escalation",
                "debt collection escalation",
            )
        )

        with self._translate_lock:
            encoded = tokenizer(
                prepared,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            encoded = {
                key: value.to(
                    self._device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                )

            result = tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
            ).strip()

        return result
