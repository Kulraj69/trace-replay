"""Deterministic agent trace recording and replay.

You cannot debug a 14-step agent workflow if you can't replay the exact
sequence that led to the failure. A Tracer records spans (llm hops, tool
calls, state mutations) as a tree with explicit parent links, writes them
to JSONL, and replays them: reconstruct the ordered event stream or the
accumulated state at any step. Diffing two traces shows exactly where tool
paths diverge. All offline, no model calls.
"""

import json
import time
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str                 # e.g. 'llm.call', 'tool.read_file', 'state.set'
    attrs: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    parent: str = None
    span_id: str = ""
    start: float = 0.0
    end: float = 0.0

    def to_dict(self):
        return {"span_id": self.span_id, "parent": self.parent, "name": self.name,
                "attrs": self.attrs, "events": self.events, "start": self.start, "end": self.end}

    @staticmethod
    def from_dict(d):
        return Span(d["name"], d.get("attrs", {}), d.get("events", []),
                    d.get("parent"), d["span_id"], d.get("start", 0.0), d.get("end", 0.0))


class Tracer:
    """Records spans into an in-memory tree; flush to JSONL."""

    def __init__(self, now=None):
        self._now = now or time.time
        self._spans = []
        self._open = []
        self._counter = 0

    def start_span(self, name, **attrs):
        self._counter += 1
        span = Span(name, attrs, [], self._open[-1].span_id if self._open else None,
                    "%06d" % self._counter, self._now(), 0.0)
        self._open.append(span)
        self._spans.append(span)
        return span

    def event(self, name, **attrs):
        if not self._open:
            raise RuntimeError("event outside span")
        self._open[-1].events.append({"name": name, "attrs": attrs, "t": self._now()})

    def end_span(self):
        if not self._open:
            raise RuntimeError("no open span")
        self._open.pop().end = self._now()

    def span(self, name, **attrs):
        """Context manager helper."""
        tracer = self
        class _Ctx:
            def __enter__(self):
                return tracer.start_span(name, **attrs)
            def __exit__(self, *exc):
                tracer.end_span()
        return _Ctx()

    def spans(self):
        return list(self._spans)

    def dump_jsonl(self):
        return "\n".join(json.dumps(s.to_dict(), sort_keys=True) for s in self._spans)

    @staticmethod
    def load_jsonl(text):
        t = Tracer()
        t._spans = [Span.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]
        if t._spans:
            t._counter = max(int(s.span_id) for s in t._spans)
        return t


def replay(tracer):
    """Flatten the span tree into the exact ordered event stream.

    Returns a list of steps: (index, path, name, attrs). Path shows the
    parent chain (root/child/...) so nested tool calls stay attributable.
    """
    children = {}
    for s in tracer.spans():
        children.setdefault(s.parent, []).append(s)

    steps = []
    def walk(span, path, counter):
        label = "%s/%s" % (path, span.name) if path else span.name
        steps.append({"index": counter[0], "path": label, "attrs": span.attrs})
        counter[0] += 1
        for child in children.get(span.span_id, []):
            walk(child, label, counter)

    counter = [0]
    for root in children.get(None, []):
        walk(root, "", counter)
    return steps


def diff(trace_a, trace_b):
    """Ordered step paths where two runs diverge. Deterministic."""
    a = [(s["path"]) for s in replay(trace_a)]
    b = [(s["path"]) for s in replay(trace_b)]
    out = []
    for i in range(max(len(a), len(b))):
        sa = a[i] if i < len(a) else None
        sb = b[i] if i < len(b) else None
        if sa != sb:
            out.append({"step": i, "a": sa, "b": sb})
    return out
