from typing import Literal

# Out-of-scope keywords
OUT_OF_SCOPE_KEYWORDS = [
    "bitcoin", "crypto", "crytpocurrency", 
    "joke", "ανέκδοτο",
    "iphone", "samsung", "τεχνολογία",
    "πολιτική", "κυβέρνηση",
    "επένδυση", "επενδύσω",
    "μετοχές", "stock", "forex", "trading"
]

# Sensitive / unsafe content
UNSAFE_KEYWORDS = [
    "pin μου", "κωδικός μου", "password", 
    "πώς να κλέψω", "hack", "χακάρω"
]

# Personal advice (regulatory)
PERSONAL_ADVICE_KEYWORDS = [
    "να πάρω", "να επιλέξω", "τι με συμφέρει",
    "συμβουλή", "επενδυτική συμβουλή", "προτείνεις"
]


def is_out_of_scope(query: str) -> bool:
    query_l = query.lower()
    return any(word in query_l for word in OUT_OF_SCOPE_KEYWORDS)


def is_unsafe(query: str) -> bool:
    query_l = query.lower()
    return any(word in query_l for word in UNSAFE_KEYWORDS)


def is_personal_advice(query: str) -> bool:
    query_l = query.lower()
    return any(word in query_l for word in PERSONAL_ADVICE_KEYWORDS)


def classify_product(query: str) -> Literal["cards", "deposits", "mortgages", "unknown"]:
    q = query.lower()
    if any(x in q for x in ["κάρτα", "card", "visa", "mastercard", "πιστωτ"]):
        return "cards"
    if any(x in q for x in ["κατάθεση", "deposit", "λογαριασμό"]):
        return "deposits"
    if any(x in q for x in ["στεγαστικό", "mortgage", "δάνειο"]):
        return "mortgages"
    return "unknown"
