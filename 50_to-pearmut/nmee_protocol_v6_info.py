# the info part of custom pearmut protocol: 
info = {
    "assignment": "task-based",
    "protocol": "ESA",
    "sliders": [],
    "mqm_severities": [],
    "word_level": True,
    "show_alignment": False,
    "special_tokens": ["[no harm]", "[undecidable]"],
    "require_full_attention": True,
    "require_span": True,
    "exclusive_special_tokens": True,
}
with open("nmee_protocol_v6.html", "r") as f:
    info["instructions"] = f.read()
