import unittest

from trace_replay import Tracer, replay, diff


def make(now):
    return Tracer(now=lambda: now[0])


class TestTracer(unittest.TestCase):
    def test_record_and_order(self):
        now = [0.0]
        t = make(now)
        with t.span("llm.call", model="test"):
            t.event("prompt", tokens=10)
            now[0] += 1.0
            with t.span("tool.read_file", path="/x"):
                pass
        spans = t.spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[1].parent, spans[0].span_id)
        self.assertEqual(spans[0].events[0]["name"], "prompt")

    def test_jsonl_roundtrip(self):
        t = make([0.0])
        with t.span("a"):
            with t.span("b", k=1):
                pass
        t2 = Tracer.load_jsonl(t.dump_jsonl())
        self.assertEqual([s.name for s in t2.spans()], ["a", "b"])
        self.assertEqual(t2.spans()[1].attrs["k"], 1)

    def test_replay_paths(self):
        t = make([0.0])
        with t.span("llm.call"):
            with t.span("tool.search"):
                pass
        with t.span("tool.write"):
            pass
        steps = replay(t)
        self.assertEqual([s["path"] for s in steps],
                         ["llm.call", "llm.call/tool.search", "tool.write"])

    def test_diff_detects_divergence(self):
        t1 = make([0.0])
        with t1.span("tool.a"):
            pass
        with t1.span("tool.b"):
            pass
        t2 = make([0.0])
        with t2.span("tool.a"):
            pass
        with t2.span("tool.c"):
            pass
        d = diff(t1, t2)
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["a"], "tool.b")
        self.assertEqual(d[0]["b"], "tool.c")

    def test_diff_identical_traces(self):
        t1 = make([0.0])
        t2 = make([0.0])
        for t in (t1, t2):
            with t.span("x"):
                pass
        self.assertEqual(diff(t1, t2), [])

    def test_event_outside_span_raises(self):
        t = make([0.0])
        with self.assertRaises(RuntimeError):
            t.event("orphan")

    def test_deterministic_replay(self):
        def build():
            t = make([0.0])
            with t.span("root"):
                with t.span("c1"):
                    pass
                with t.span("c2"):
                    pass
            return replay(t)
        self.assertEqual(build(), build())


if __name__ == "__main__":
    unittest.main()
