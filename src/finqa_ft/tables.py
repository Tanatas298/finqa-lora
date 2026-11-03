def to_markdown(table: list[list[str]]) -> str:
    """Render a FinQA table (first row = header) as a Markdown table."""
    if not table:
        return ""
    width = max(len(r) for r in table)
    rows = [[c.strip().replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
