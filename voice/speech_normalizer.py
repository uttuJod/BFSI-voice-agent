from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


class SpeechTextNormalizer:
    """
    Converts display text into natural text for TTS only.

    - UI / grounded response text is not changed.
    - INR amounts use Indian number units.
    - Hindi first-person assistant grammar is normalized to feminine form.
    """

    _MONEY = re.compile(
        r"(?<![\w.])"
        r"(?:₹\s*)?"
        r"(\d[\d,]*(?:\.\d{1,2})?)"
        r"\s*(INR|Rs\.?|rupees?)"
        r"(?!\w)",
        re.IGNORECASE,
    )

    _EN_ONES = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    _EN_TENS = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    _HI_ONES = [
        "शून्य",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "छह",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "ग्यारह",
        "बारह",
        "तेरह",
        "चौदह",
        "पंद्रह",
        "सोलह",
        "सत्रह",
        "अठारह",
        "उन्नीस",
        "बीस",
        "इक्कीस",
        "बाईस",
        "तेईस",
        "चौबीस",
        "पच्चीस",
        "छब्बीस",
        "सत्ताईस",
        "अट्ठाईस",
        "उनतीस",
        "तीस",
        "इकतीस",
        "बत्तीस",
        "तैंतीस",
        "चौंतीस",
        "पैंतीस",
        "छत्तीस",
        "सैंतीस",
        "अड़तीस",
        "उनतालीस",
        "चालीस",
        "इकतालीस",
        "बयालीस",
        "तैंतालीस",
        "चवालीस",
        "पैंतालीस",
        "छियालीस",
        "सैंतालीस",
        "अड़तालीस",
        "उनचास",
        "पचास",
        "इक्यावन",
        "बावन",
        "तिरपन",
        "चौवन",
        "पचपन",
        "छप्पन",
        "सत्तावन",
        "अट्ठावन",
        "उनसठ",
        "साठ",
        "इकसठ",
        "बासठ",
        "तिरसठ",
        "चौंसठ",
        "पैंसठ",
        "छियासठ",
        "सड़सठ",
        "अड़सठ",
        "उनहत्तर",
        "सत्तर",
        "इकहत्तर",
        "बहत्तर",
        "तिहत्तर",
        "चौहत्तर",
        "पचहत्तर",
        "छिहत्तर",
        "सतहत्तर",
        "अठहत्तर",
        "उनासी",
        "अस्सी",
        "इक्यासी",
        "बयासी",
        "तिरासी",
        "चौरासी",
        "पचासी",
        "छियासी",
        "सत्तासी",
        "अट्ठासी",
        "नवासी",
        "नब्बे",
        "इक्यानवे",
        "बानवे",
        "तिरानवे",
        "चौरानवे",
        "पचानवे",
        "छियानवे",
        "सत्तानवे",
        "अट्ठानवे",
        "निन्यानवे",
    ]

    def normalize(
        self,
        text: str,
        language: str,
    ) -> str:
        text = str(text or "").strip()
        language = str(
            language or "english"
        ).strip().lower()

        if not text:
            return ""

        text = self._normalize_money(
            text,
            language,
        )

        if language == "hindi":
            text = (
                self._feminize_hindi_first_person(
                    text
                )
            )

        return text

    def _normalize_money(
        self,
        text: str,
        language: str,
    ) -> str:
        def replace(
            match: re.Match,
        ) -> str:
            raw_number = match.group(1)

            try:
                value = Decimal(
                    raw_number.replace(
                        ",",
                        "",
                    )
                )
            except InvalidOperation:
                return match.group(0)

            whole = int(value)

            paise = int(
                (value - whole)
                * 100
            )

            if language == "hindi":
                words = (
                    self._hi_indian_number(
                        whole
                    )
                )

                if paise:
                    return (
                        f"{words} रुपये और "
                        f"{self._hi_indian_number(paise)} पैसे"
                    )

                return (
                    f"{words} रुपये"
                )

            words = (
                self._en_indian_number(
                    whole
                )
            )

            if paise:
                return (
                    f"{words} rupees and "
                    f"{self._en_indian_number(paise)} paise"
                )

            return (
                f"{words} rupees"
            )

        return self._MONEY.sub(
            replace,
            text,
        )

    def _en_below_thousand(
        self,
        n: int,
    ) -> str:
        if n < 20:
            return self._EN_ONES[n]

        if n < 100:
            tens, ones = divmod(
                n,
                10,
            )

            return " ".join(
                part
                for part in (
                    self._EN_TENS[tens],
                    self._EN_ONES[ones],
                )
                if part
            )

        hundreds, rest = divmod(
            n,
            100,
        )

        parts = [
            self._EN_ONES[
                hundreds
            ],
            "hundred",
        ]

        if rest:
            parts.append(
                self._en_below_thousand(
                    rest
                )
            )

        return " ".join(
            parts
        )

    def _en_indian_number(
        self,
        n: int,
    ) -> str:
        if n == 0:
            return "zero"

        if n < 0:
            return (
                "minus "
                + self._en_indian_number(
                    -n
                )
            )

        parts: list[str] = []

        crore, n = divmod(
            n,
            10_000_000,
        )

        lakh, n = divmod(
            n,
            100_000,
        )

        thousand, n = divmod(
            n,
            1_000,
        )

        if crore:
            parts.extend(
                [
                    self._en_indian_number(
                        crore
                    ),
                    "crore",
                ]
            )

        if lakh:
            parts.extend(
                [
                    self._en_below_thousand(
                        lakh
                    ),
                    "lakh",
                ]
            )

        if thousand:
            parts.extend(
                [
                    self._en_below_thousand(
                        thousand
                    ),
                    "thousand",
                ]
            )

        if n:
            parts.append(
                self._en_below_thousand(
                    n
                )
            )

        return " ".join(
            parts
        )

    def _hi_below_thousand(
        self,
        n: int,
    ) -> str:
        if n < 100:
            return self._HI_ONES[n]

        hundreds, rest = divmod(
            n,
            100,
        )

        parts = [
            self._HI_ONES[
                hundreds
            ],
            "सौ",
        ]

        if rest:
            parts.append(
                self._HI_ONES[
                    rest
                ]
            )

        return " ".join(
            parts
        )

    def _hi_indian_number(
        self,
        n: int,
    ) -> str:
        if n == 0:
            return "शून्य"

        if n < 0:
            return (
                "माइनस "
                + self._hi_indian_number(
                    -n
                )
            )

        parts: list[str] = []

        crore, n = divmod(
            n,
            10_000_000,
        )

        lakh, n = divmod(
            n,
            100_000,
        )

        thousand, n = divmod(
            n,
            1_000,
        )

        if crore:
            parts.extend(
                [
                    self._hi_indian_number(
                        crore
                    ),
                    "करोड़",
                ]
            )

        if lakh:
            parts.extend(
                [
                    self._hi_below_thousand(
                        lakh
                    ),
                    "लाख",
                ]
            )

        if thousand:
            parts.extend(
                [
                    self._hi_below_thousand(
                        thousand
                    ),
                    "हज़ार",
                ]
            )

        if n:
            parts.append(
                self._hi_below_thousand(
                    n
                )
            )

        return " ".join(
            parts
        )

    def _feminize_hindi_first_person(
        self,
        text: str,
    ) -> str:
        """
        Normalize only first-person assistant grammar.

        Unlike the old exact-string replacement, these regexes allow
        words between "मैं" and the verb, e.g.:

            मैं यह नहीं कर सकता
                -> मैं यह नहीं कर सकती

            मैं आपकी मदद कर सकता हूँ
                -> मैं आपकी मदद कर सकती हूँ
        """

        # समझता -> समझती
        text = re.sub(
            r"मैं(\s+)समझता(?=\s|हूँ|हूं|[।.!?]|$)",
            r"मैं\1समझती",
            text,
        )

        # General "मैं ... नहीं ... सकता" forms.
        text = re.sub(
            r"(मैं[^।.!?]{0,100}?\bनहीं\s+"
            r"(?:कर|बता|साझा|पुष्टि|मदद)\s+)"
            r"सकता"
            r"(?=\s|हूँ|हूं|[।.!?]|$)",
            r"\1सकती",
            text,
        )

        # General positive "मैं ... सकता" forms.
        text = re.sub(
            r"(मैं[^।.!?]{0,100}?"
            r"(?:कर|बता|साझा|पुष्टि|मदद)\s+)"
            r"सकता"
            r"(?=\s|हूँ|हूं|[।.!?]|$)",
            r"\1सकती",
            text,
        )

        # Clean up common malformed MT forms.
        text = text.replace(
            "मैं नहीं कर सकते",
            "मैं नहीं कर सकती",
        )

        text = text.replace(
            "मैं कर सकते",
            "मैं कर सकती हूँ",
        )

        return text
