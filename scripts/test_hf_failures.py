from integration.llm_router import QwenBFSIRouter

router = QwenBFSIRouter(
    adapter_path="models/qwen35_customer_support_FINAL_FROZEN"
)

tests = [
    "I could probably pay something sometime after the 14th.",
    "Around ₹2700 might be possible once salary arrives.",
    "Maybe I can make the payment during the first week next month.",
    "I might manage ₹1900 sometime next week.",
    "Month end ke aas paas ₹3200 arrange ho sakta hai.",
    "Salary मिलने के बाद ₹2500 के आसपास दे सकता हूँ।",
]

for text in tests:
    print("\nUSER:", text)
    result = router.route(text)
    print(result.model_dump())