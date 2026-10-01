"""Render the flagged spans of one segment as an HTML table for the item instructions.
One column per target system to be annotated (the ASR column included); systems without
any flagged span get empty cells. Rows: the suggested harmful error span, the intended
translation, the harm type, the likely error source, and the explanation (with the ASR
transcript appended)."""
from collections import defaultdict
import html


def esc(s):
    return html.escape(str(s), quote=False)

def instruction_html(rec, targets, tgt):
    if not tgt:
        return ""
    by_system = defaultdict(list)
    for t in targets:
        by_system[t["system"]].append(t)
    systems = list(tgt)

    HEAD = 'style="color: #555;"'

    def row(label, cells):
        return (f'<tr><th align="left" {HEAD}>{label}</th>'
                + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    def cell(system, render):
        return "<br>".join(render(t) for t in by_system[system]) if by_system[system] else ""

    def explanation_cell(system):
        if not by_system[system]:
            return ""
        out = "<br>".join(esc(t.get("explanation", "")) for t in by_system[system])
        if rec.get("asr"):
            out += f"<br>ASR: {esc(rec['asr'])}"
        return out

    def column_language(system):
        if by_system[system]:
            return by_system[system][0].get("tgt_lan", "")
        return rec.get("src_language", "")

    parts = ['<table border="1" style="border-collapse: collapse; '
             'font-style: italic; font-size: 0.85em;">']
    parts.append("<tr><th></th>" + "".join(
        f'<th {HEAD}>{esc(s)} ({esc(column_language(s).upper())})</th>'
        for s in systems) + "</tr>")

    parts.append(row("<b>Suggested harmful error span</b>", [
        cell(s, lambda t: f"{esc(t['span'])}" if t.get("span")
             else f"<i>missing: {esc(t.get('intended', ''))}</i>")
        for s in systems]))
    parts.append(row("<b>Intended translation</b>", [
        cell(s, lambda t: f"{esc(t.get('intended', ''))}")
        for s in systems]))
    parts.append(row("<b>Harm type</b>", [
        cell(s, lambda t: esc(', '.join(t.get('harm_types', []))))
        for s in systems]))
    parts.append(row("<b>Likely source</b>", [
        cell(s, lambda t: esc(t.get('error_source', 'unknown')))
        for s in systems]))
    parts.append(row("<b>Explanation</b>", [explanation_cell(s) for s in systems]))
    parts.append("</table>")
    return "".join(parts)