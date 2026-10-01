# the info part of custom pearmut protocol: 
info = {
    "assignment": "task-based",
    "protocol": "ESA",
    "sliders": [],
    "mqm_severities": [],
    "word_level": True,
    "show_alignment": False,
    "special_tokens": ["[no harm]", "[undecidable]"]
}
with open("nmee_protocol_v6.html", "r") as f:
    info["instructions"] = f.read()