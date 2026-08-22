from integration.vllm_router import SupportDecision


def test_schema_accepts_valid_router_decision():
    item = SupportDecision.model_validate(
        {
            "intent": "account_status",
            "confidence": 0.98,
            "requires_rag": False,
            "requires_tool": True,
            "tool": "get_outstanding_balance",
            "arguments": {},
            "response_style": "neutral",
            "needs_clarification": False,
            "response": "I can retrieve the current outstanding balance.",
        }
    )
    assert item.intent == "account_status"


def test_schema_forbids_unknown_intent():
    try:
        SupportDecision.model_validate(
            {
                "intent": "made_up_intent",
                "confidence": 0.98,
                "requires_rag": False,
                "requires_tool": False,
                "tool": None,
                "arguments": {},
                "response_style": "neutral",
                "needs_clarification": False,
                "response": "x",
            }
        )
    except Exception:
        return
    raise AssertionError("Unknown intent should fail validation.")
