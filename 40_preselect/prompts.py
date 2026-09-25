#!/usr/bin/env python3
"""Prompts and the response schema for the harmful-error annotation.

build_chat(view, cfg) turns one unit of work (a segment, or later a document) into a chat. A "view" is
what the LLM is shown:

    {"domain": "This is an earnings call ...",          # optional static description
     "context_before": [...], "context_after": [...],   # optional neighbouring sentences (not annotated)
     "shown":    [{"system", "lan", "text", "role"}],   # role: source | reference | target
     "annotate": ["canary_asr", "canary_cs", ...]}      # systems the model must look for errors in

Everything the model needs is in the user message; the system message only fixes the role and the JSON
output. A different prompting strategy (few-shot, chain of thought, DSPy) can live in its own module
and reuse HARM_TYPES and response_schema().
"""

HARM_TYPES = [
    "False attribution",
    "Offensive",
    "Embarrassing or laughable",
    "Derailing or contresens",
    "Safety, health or legal risk",
    "Other",
]

ERROR_SOURCES = ["ASR", "MT", "ASR+MT", "unknown"]

SYSTEM_MESSAGE = ("You are an expert evaluator of speech translation and speech recognition output. "
                  "You answer only with valid JSON.")

TASK = """Imagine a live event whose speech is transcribed and translated automatically, and the output is shown on a large screen to the audience.

Find COMPREHENSION ERRORS THAT ARE ALSO HARMFUL in the outputs marked "to annotate". A comprehension error changes or obscures the meaning, or misleads the reader. It is harmful if it would cause a problem beyond the misunderstanding itself: someone would have to correct it or apologise for it, be offended or embarrassed by it, laugh at it, or be at risk if they acted on it.

Harm types:
- "False attribution": the output gives a person the wrong identity, gender, role, relationship, or characteristics.
- "Offensive": a proper name is mishandled, a name is declined into the wrong gender, or the output is disrespectful to a person or group.
- "Embarrassing or laughable": the error introduces explicit, absurd or unintentionally funny content.
- "Derailing or contresens": the output is shocking or unexpected, or conveys a false or opposite message that still sounds plausible.
- "Safety, health or legal risk": the error has an immediate real-world consequence here, beyond the general risk of being misinformed.
- "Other": harmful in some other way.

Do NOT annotate: harmless paraphrases, style, word order, punctuation, capitalisation, or spelling that does not change the meaning; nor content that is missing from one output but harmless in itself.

For each error report:
- "system": the name of the output the error is in (one of the systems to annotate);
- "span": the exact, contiguous text of that output containing the error (copy it verbatim, as short as possible);
- "intended": what the span should have said;
- "harm_types": one or more of the harm types above;
- "harmfulness": an integer 1-5, how likely the error causes harm beyond the misunderstanding (1 = very unlikely, 5 = very likely);
- "explanation": one or two sentences on what went wrong and why it is harmful;
- "error_source": "ASR" if the error is already in the transcript, "MT" if the transcript is correct but the translation is not, "ASR+MT" if both, "unknown" if you cannot tell.

Answer with a JSON object of this shape, and nothing else:
{"annotations": [{"system": "...", "span": "...", "intended": "...", "harm_types": ["..."], "harmfulness": 3, "explanation": "...", "error_source": "MT"}]}
Answer {"annotations": []} if there is no harmful error."""

ROLE_NOTE = {
    "source": "automatic transcript of the speech",
    "gold": "human transcript, correct",
    "reference": "human reference translation, correct",
    "target": "system output",
}


def response_schema(systems):
    """JSON schema for constrained decoding / response_format."""
    return {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "system": {"type": "string", "enum": list(systems)},
                        "span": {"type": "string"},
                        "intended": {"type": "string"},
                        "harm_types": {"type": "array",
                                       "items": {"type": "string", "enum": HARM_TYPES}},
                        "harmfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "explanation": {"type": "string"},
                        "error_source": {"type": "string", "enum": ERROR_SOURCES},
                    },
                    "required": ["system", "span", "intended", "harm_types", "harmfulness",
                                 "explanation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["annotations"],
        "additionalProperties": False,
    }


def _lines(view):
    out = []
    if view.get("domain"):
        out.append(f"Situation: {view['domain']}")
    for key, label in (("context_before", "Preceding context"), ("context_after", "Following context")):
        if view.get(key):
            out.append(f"{label} (not annotated): " + " ".join(view[key]))
    out.append("")
    for item in view["shown"]:
        mark = " to annotate" if item["system"] in view["annotate"] else ""
        note = ROLE_NOTE.get(item["role"], item["role"])
        out.append(f"[{item['system']}] ({item['lan']}, {note}){mark}: {item['text']}")
    out.append("")
    out.append("Systems to annotate: " + ", ".join(view["annotate"]))
    return out


def build_chat(view, cfg=None):
    """[{"role": "system"...}, {"role": "user"...}] for one segment (or document)."""
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": TASK + "\n\n" + "\n".join(_lines(view))}]
