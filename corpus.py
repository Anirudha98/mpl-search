"""Build the search corpus: one document per public matplotlib function or method name.

Methods with the same name on different classes (plt.legend, Axes.legend, Figure.legend)
are merged into one document, because Stack Overflow answers call them interchangeably
and a search result is useful if it names the right function, whichever object it hangs on.

Usage:  .venv/bin/python corpus.py          -> writes data/corpus.jsonl
        .venv/bin/python corpus.py --check
"""
import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import axes, axis, colorbar, colors, dates, figure, legend, lines, patches, text, ticker  # noqa: E402

OUT = Path(__file__).parent / "data" / "corpus.jsonl"
DOC_CHARS = 1500  # summary + the start of the parameter list; the rest is mostly noise for retrieval

# Where people's questions actually land. Modules give functions and classes, classes give methods.
SOURCES = {
    "pyplot": plt,
    "Axes": axes.Axes,
    "Figure": figure.Figure,
    "Axis": axis.Axis,
    "Legend": legend.Legend,
    "Text": text.Text,
    "Line2D": lines.Line2D,
    "Patch": patches.Patch,
    "Colorbar": colorbar.Colorbar,
    "colors": colors,
    "ticker": ticker,
    "dates": dates,
}


def public_callables(owner_name, obj):
    for name, member in inspect.getmembers(obj):
        if name.startswith("_") or not callable(member):
            continue
        if inspect.ismodule(obj) and getattr(member, "__module__", "") and not member.__module__.startswith("matplotlib"):
            continue  # numpy and friends re-exported by pyplot
        doc = inspect.getdoc(member)
        if doc:
            yield name, f"{owner_name}.{name}", doc


def build():
    docs = {}
    for owner_name, obj in SOURCES.items():
        for name, qualname, doc in public_callables(owner_name, obj):
            d = docs.setdefault(name, {"name": name, "owners": [], "text": ""})
            d["owners"].append(qualname)
            if len(doc) > len(d["text"]):  # keep the fullest docstring among the same-named ones
                d["text"] = doc
    for d in docs.values():
        d["text"] = f"{d['name']}\n{d['text'][:DOC_CHARS]}"
    return sorted(docs.values(), key=lambda d: d["name"])


def check():
    docs = build()
    names = {d["name"] for d in docs}
    assert {"legend", "set_xlim", "tight_layout", "savefig", "set_major_formatter"} <= names
    leg = next(d for d in docs if d["name"] == "legend")
    assert {"pyplot.legend", "Axes.legend", "Figure.legend"} <= set(leg["owners"]), leg["owners"]
    assert "array" not in names, "numpy re-exports leaked in"
    print(f"ok: {len(docs)} documents")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        docs = build()
        OUT.parent.mkdir(exist_ok=True)
        with OUT.open("w") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        print(f"wrote {len(docs)} documents to {OUT}")
