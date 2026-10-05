#!/usr/bin/env python3
"""Prompts and the response schema for the harmful-error annotation.

build_chat(view, cfg) turns one unit of work (a segment) into a chat. A "view" is what the LLM is shown:

    {"domain": "This is an earnings call ...",          # optional static description
     "context_before": [...], "context_after": [...],   # optional neighbouring sentences (not annotated)
     "shown":    [{"system", "lan", "text", "role"}],   # role: source | reference | target
     "annotate": ["canary_asr", "canary_cs", ...]}      # systems the model must look for errors in

build_group_chat(views) puts several views (segments of one document) into a single request; the
model then labels every annotation with the number of the segment it belongs to.

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

You are preselecting errors for human annotation. False positives are costly: two independent annotators must later agree on every error you flag, and they will waste time on unclear cases. Only flag errors that are UNAMBIGUOUS and CLEARLY HARMFUL.

Find COMPREHENSION ERRORS THAT ARE ALSO HARMFUL in the outputs marked "to annotate". A comprehension error changes or obscures the meaning, or misleads the reader. It is harmful if it would cause a problem beyond the misunderstanding itself: someone would have to correct it or apologise for it, be offended or embarrassed by it, laugh at it, or be at risk if they acted on it.

Harm types:
- "False attribution": the output gives a person the wrong identity, gender, role, relationship, or characteristics.
- "Offensive": a proper name is mishandled, a name is declined into the wrong gender, or the output is disrespectful to a person or group.
- "Embarrassing or laughable": the error introduces explicit, absurd or unintentionally funny content.
- "Derailing or contresens": the output is shocking or unexpected, or conveys a false or opposite message that still sounds plausible.
- "Safety, health or legal risk": the error has an immediate real-world consequence here, beyond the general risk of being misinformed.
- "Other": harmful in some other way.

Do NOT annotate:
- Borderline cases, minor errors, or anything two independent annotators might reasonably disagree on.
- Harmless paraphrases, style, word order, punctuation, capitalisation, or spelling that does not change the meaning.
- Content that is missing from one output but harmless in itself.
- Errors of degree or subjective interpretation.

When you are uncertain, do not flag the error. Precision is more important than recall.

For each error report:
- "system": the name of the output the error is in (one of the systems to annotate);
- "span": the exact, contiguous text of that output containing the error (copy it verbatim, as short as possible);
- "intended": what the span should have said;
- "harm_types": one or more of the harm types above;
- "harmfulness": an integer 1-5, how likely the error causes harm beyond the misunderstanding (1 = very unlikely, 5 = very likely); only flag errors with harmfulness 3 or above;
- "explanation": one or two sentences on what went wrong and why it is harmful;
- "error_source": "ASR" if the error is already in the transcript, "MT" if the transcript is correct but the translation is not, "ASR+MT" if both, "unknown" if you cannot tell.

Answer with a JSON object of this shape, and nothing else:
{"annotations": [{"system": "...", "span": "...", "intended": "...", "harm_types": ["..."], "harmfulness": 3, "explanation": "...", "error_source": "MT"}]}
Answer {"annotations": []} if there is no unambiguous harmful error."""

# the same task for a request with several numbered segments
GROUP_TASK = (
    TASK
    .replace("Find COMPREHENSION",
             "The input below contains several numbered segments, in order, taken from one speech "
             "(some segments of it may be left out). Judge every segment on its own: the outputs of one "
             "segment are never evidence for another, but the neighbouring segments help you understand "
             "the context.\n\nFind COMPREHENSION", 1)
    .replace("For each error report:\n",
             "For each error report:\n- \"segment\": the number of the segment the error is in;\n", 1)
    .replace('{"annotations": [{"system"', '{"annotations": [{"segment": 1, "system"', 1)
    .replace("if there is no unambiguous harmful error.", "if there is no unambiguous harmful error in any segment.", 1)
)

ROLE_NOTE = {
    "source": "automatic transcript of the speech",
    "gold": "human transcript, correct",
    "reference": "human reference translation, correct",
    "target": "system output",
}


def response_schema(systems, grouped=False):
    """JSON schema for constrained decoding / response_format.
    grouped: every annotation also carries the number of its segment."""
    item = {
        "system": {"type": "string", "enum": list(systems)},
        "span": {"type": "string"},
        "intended": {"type": "string"},
        "harm_types": {"type": "array", "items": {"type": "string", "enum": HARM_TYPES}},
        "harmfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
        "error_source": {"type": "string", "enum": ERROR_SOURCES},
    }
    required = ["system", "span", "intended", "harm_types", "harmfulness", "explanation"]
    if grouped:
        item = {"segment": {"type": "integer", "minimum": 1}, **item}
        required = ["segment"] + required
    return {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "items": {"type": "object", "properties": item, "required": required,
                          "additionalProperties": False},
            },
        },
        "required": ["annotations"],
        "additionalProperties": False,
    }


def _header(view):
    """Situation and neighbouring context."""
    out = []
    if view.get("domain"):
        out.append(f"Situation: {view['domain']}")
    for key, label in (("context_before", "Preceding context"), ("context_after", "Following context")):
        if view.get(key):
            out.append(f"{label} (not annotated): " + " ".join(view[key]))
    return out


def _body(view, with_systems=True):
    """The outputs of one segment and (optionally) the list of systems to annotate."""
    out = []
    for item in view["shown"]:
        mark = " to annotate" if item["system"] in view["annotate"] else ""
        note = ROLE_NOTE.get(item["role"], item["role"])
        out.append(f"[{item['system']}] ({item['lan']}, {note}){mark}: {item['text']}")
    if with_systems:
        out.append("")
        out.append("Systems to annotate: " + ", ".join(view["annotate"]))
    return out


def _lines(view):
    return _header(view) + [""] + _body(view)


def build_chat(view, cfg=None):
    """[{"role": "system"...}, {"role": "user"...}] for one segment."""
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": TASK + "\n\n" + "\n".join(_lines(view))}]


def build_group_chat(views, cfg=None):
    """One chat for several segments of the same document, numbered from 1.
    The situation is stated once; the context before the first and after the last segment is shown.
    If every segment has the same systems to annotate, the list is given once in the header."""
    first, last = views[0], views[-1]
    header = _header({"domain": first.get("domain"),
                      "context_before": first.get("context_before"),
                      "context_after": last.get("context_after")})
    common = all(v["annotate"] == first["annotate"] for v in views)
    if common:
        header.append("Systems to annotate in every segment: " + ", ".join(first["annotate"]))
    lines = header + [""]
    for k, view in enumerate(views, 1):
        lines.append(f"=== Segment {k} ===")
        lines += _body(view, with_systems=not common)
        lines.append("")
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": GROUP_TASK + "\n\n" + "\n".join(lines).rstrip()}]