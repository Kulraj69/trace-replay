# trace-replay

Deterministic agent trace recording, replay and diffing. Offline, standard
library only, no model calls.

Covers the "OpenTelemetry Agent Tracer" / "LLM Trace Replay Debugger" idea:
record every LLM hop, tool call and state mutation as a span tree, then
replay the exact sequence later. You cannot fix a 14-step agent workflow if
you can't replay the context that caused the failure.

```python
from trace_replay import Tracer, replay, diff

t = Tracer()
with t.span("llm.call", model="example"):
    t.event("prompt", tokens=42)
    with t.span("tool.read_file", path="/data.csv"):
        pass

steps = replay(t)          # ordered, parent-linked step stream
log = t.dump_jsonl()       # persist; Tracer.load_jsonl(log) restores
d = diff(t, other_trace)   # first steps where two runs diverge
```

Honest scope: this is a minimal deterministic span recorder and replayer,
not a full OpenTelemetry pipeline - no remote collector, sampling or
exporters. It records what your harness tells it; it does not observe tool
internals on its own.

## Tests

```
python3 -m unittest discover -s tests
```

## Demo

`demo.py` reproduces the recorded demo run end to end, offline (standard
library only, no model calls):

    python demo.py

It executes the library's real code path and prints the same output shown in
the recorded demo video.
