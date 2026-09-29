"""Render the flagged spans of one segment as an HTML table for the item instructions.
One column per target system, rows: the suggested harmful error span, the intended
translation, the harm type, the likely error source, and the explanation (with the ASR
transcript appended)."""
from collections import defaultdict
import html


def esc(s):
    return html.escape(str(s), quote=False)


def instruction_html(rec, targets):
    """Table of the flagged spans: one column per target system, rows for the
    suggested span, the intended translation, the harm type, the likely error
    source, and the explanation (with the ASR transcript)."""
    if not targets:
        return ""
    # group the flagged spans by system, in the order they were flagged
    by_system = defaultdict(list)
    for t in targets:
        by_system[t["system"]].append(t)
    systems = list(by_system)

    def row(label, cells):
        return f'<tr><th align="left">{label}</th>' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def cell(system, render):
        return "<br>".join(render(t) for t in by_system[system])

    def explanation_cell(system):
        out = "<br>".join(esc(t.get("explanation", "")) for t in by_system[system])
        if rec.get("asr"):
            out += f"<br>ASR: {esc(rec['asr'])}"
        return out

    parts = ['<table border="1" style="border-collapse: collapse;">']
    # row 1: the system labels, e.g. canary_cs (CS)
    parts.append("<tr><th></th>" + "".join(
        f"<th>{esc(s)} ({esc(by_system[s][0].get('tgt_lan', '').upper())})</th>"
        for s in systems) + "</tr>")

    parts.append(row("<b>Suggested harmful error span</b>", [
        cell(s, lambda t: f"{esc(t['span'])}" if t.get("span")
             else f"<i>missing: {esc(t.get('intended', ''))}</i>")
        for s in systems]))
    parts.append(row("<b>Intended translation</b>", [
        cell(s, lambda t: f"{esc(t.get('intended', ''))}")
        for s in systems]))
    parts.append(row("<b>Harm type</b>", [
        cell(s, lambda t: f"<i>{esc(', '.join(t.get('harm_types', [])))}</i>")
        for s in systems]))
    parts.append(row("<b>Likely source</b>", [
        cell(s, lambda t: f"<i>{esc(t.get('error_source', 'unknown'))}</i>")
        for s in systems]))
    parts.append(row("<b>Explanation</b>", [explanation_cell(s) for s in systems]))
    parts.append("</table>")
    return "".join(parts)