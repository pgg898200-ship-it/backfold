"""Audit exact answer availability on public Backfold documents. Python 3.9+."""
import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

BASELINE = "d3f3ad5cc8f2b74cd0a67acb1c47ffe513989ef0"
REPO = "pgg898200-ship-it/backfold"
SITE = "https://pgg898200-ship-it.github.io/backfold"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BASELINE}"
BUNDLE = "https://github.com/" + REPO + "/releases/download/v0.3.0/backfold-0.3.0.mcpb"
REGISTRY = "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.pgg898200-ship-it%2Fbackfold/versions/0.3.0"
EXAMPLE = "integrations/mcpb/README.md"

# Match literal answers or distinctive destination paths, independent of document position.
MCP_PATHS = {
    "Download the desktop MCP bundle": (BUNDLE,),
    "Find the official registry entry": (REGISTRY,),
    "Find the four-row MCP example": (EXAMPLE,),
}
TASKS = {
    "Install the Python library": ("python -m pip install backfold",),
    "Read the forked example output": ("18.0 7.0",),
    "Remove a reading": ("retract",),
    **MCP_PATHS,
    "Name the two CSV tools": ("backfold_digest_csv", "backfold_drill_csv"),
    "Recover the reported edit benchmark": ("6.06 µs", "100,000"),
}


class Article(HTMLParser):
    """Read visible article text, optionally keeping link destinations."""
    def __init__(self, keep_links):
        super().__init__()
        self.keep_links = keep_links
        self.in_article = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        elif self.in_article and tag == "a" and self.keep_links:
            self.parts.append(dict(attrs).get("href", ""))

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False

    def handle_data(self, data):
        if self.in_article:
            self.parts.append(data)


def fetch(url, accept=None):
    headers = {"User-Agent": "Backfold-doc-delivery-replay", "Accept-Encoding": "identity"}
    if accept:
        headers["Accept"] = accept
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return response.read()


def hits(facts, text):
    text = " ".join(text.split())
    return [task for task, fragments in facts.items() if all(value in text for value in fragments)]


def run(guide_path):
    markdown = fetch(f"{RAW}/README.md")
    html = fetch(f"https://api.github.com/repos/{REPO}/readme?ref={BASELINE}",
                 "application/vnd.github.html+json")
    if b"<article" not in html or not markdown.startswith(b"# Backfold"):
        raise ValueError("Unexpected GitHub response for the pinned README")
    baseline_guide = fetch(f"{RAW}/llms.txt")
    current_guide = guide_path.read_bytes() if guide_path else fetch(f"{SITE}/llms.txt")

    formats = {"Markdown": markdown, "HTML with hrefs": html, "HTML text only": html}
    trials = {}
    for name, document in formats.items():
        trials[name] = {}
        for budget in (1024, 2048, 4096, 8192, None):
            prefix = document[:budget].decode("utf-8", errors="ignore")
            if name != "Markdown":
                reader = Article(keep_links=name == "HTML with hrefs")
                reader.feed(prefix)
                prefix = " ".join(reader.parts)
            trials[name][str(budget or "full")] = hits(TASKS, prefix)
    return {
        "readme_commit": BASELINE,
        "body_bytes": {name: len(body) for name, body in formats.items()},
        "body_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in formats.items()},
        "tasks": list(TASKS),
        "readme_answers_available": trials,
        "direct_mcp_links": {
            "baseline": hits(MCP_PATHS, baseline_guide.decode()),
            "candidate" if guide_path else "live": hits(MCP_PATHS, current_guide.decode()),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", type=Path, help="use a local candidate in place of the live agent guide")
    print(json.dumps(run(parser.parse_args().guide), indent=2, ensure_ascii=False))
