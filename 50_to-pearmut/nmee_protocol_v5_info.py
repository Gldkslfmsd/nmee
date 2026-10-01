info = {
    "assignment": "task-based",
    "protocol": "ESA",
    "sliders": [],
    "mqm_severities": [],
    "word_level": True,
    "show_alignment": False,
    "special_tokens": ["[no harm]", "[undecidable]"]
}
with open("nmee_protocol_v5.html", "r") as f:
    info["instructions"] = f.read()