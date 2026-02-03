# -----------------------------
# Conversational / feedback
# -----------------------------
def is_conversational_feedback(query: str) -> bool:
    q = query.lower().strip()
    conversational = {
        "thanks", "thank you", "good", "okay", "ok",
        "cool", "nice", "great", "awesome", "alright"
    }
    return q in conversational


# -----------------------------
# Retry / improve instructions
# -----------------------------
def is_retry_instruction(query: str) -> bool:
    q = query.lower()
    return any(
        phrase in q
        for phrase in [
            "try again",
            "improve",
            "do better",
            "rewrite",
            "regenerate",
            "explain better"
        ]
    )


# -----------------------------
# Ambiguous queries ✅ FIX
# -----------------------------
def is_ambiguous_query(query: str) -> bool:
    """
    Very short queries with no clear intent
    """
    tokens = query.strip().split()
    return len(tokens) <= 2


# -----------------------------
# Document-level queries
# -----------------------------
def is_document_level_query(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in [
            "summary",
            "summarize",
            "overview",
            "about the document",
            "what is this document",
            "summarize this pdf",
            "summarize this document",
        ]
    )

# -----------------------------
# Policy / rules queries
# -----------------------------
def is_policy_query(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in [
            "rules",
            "policies",
            "guidelines",
            "house rules",
            "do's",
            "dont's",
            "terms",
            "restrictions"
        ]
    )
