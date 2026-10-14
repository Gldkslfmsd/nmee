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

import os
_PATH = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_PATH, "nmee_protocol_v6.html")
if not os.path.exists(_PATH):
    raise FileNotFoundError("nmee_protocol_v6.html not found in the same directory as nmee_protocol_v6_info.py")

with open(_PATH, "r") as f:
    info["instructions"] = f.read()
