"""Reproduces the demo video's real output: record, dump, replay, diff."""
from trace_replay import Tracer, replay, diff

def good_run(now):
    tr = Tracer(now=now)
    tr.start_span("agent.run", task="book flight BLR-DEL")
    for tool in ("calendar.today", "flights.search", "flights.book"):
        tr.start_span("tool.call", tool=tool)
        if tool == "flights.search":
            tr.event("offers", count=3)
        tr.end_span()
    tr.end_span()
    return tr

def bad_run(now):
    tr = Tracer(now=now)
    tr.start_span("agent.run", task="book flight BLR-DEL")
    tr.start_span("tool.call", tool="flights.book")
    tr.end_span()
    tr.end_span()
    return tr

if __name__ == "__main__":
    c1, c2 = [1000.0], [2000.0]
    def tick(c):
        def f():
            c[0] += 0.4
            return c[0]
        return f
    a, b = good_run(tick(c1)), bad_run(tick(c2))
    jsonl = a.dump_jsonl()
    print("dump_jsonl:", len(jsonl.strip().splitlines()), "lines")
    for row in replay(Tracer.load_jsonl(jsonl)):
        print(" ", row)
    print("diff(good, bad):", diff(a, b))
