"""CPU-only panel callback tests (no Gradio or worker needed)."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

PANEL = Path(__file__).with_name("qwen_web_app_minimal.py")


class PanelTest(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        self.active = Path(sandbox.name) / "active.json"
        # Run the real callbacks without launching Gradio or sleeping between polls.
        tree = ast.parse(PANEL.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.With, ast.If))
                     and not (isinstance(n, ast.Import) and n.names[0].name == "gradio")
                     and not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                              and isinstance(n.value.func, ast.Name) and n.value.func.id == "_ts_ip")
                     and not isinstance(n, ast.Expr)]
        self.ns = {"gr": SimpleNamespace(Progress=lambda: lambda *a, **kw: None, Error=RuntimeError)}
        exec(compile(tree, str(PANEL), "exec"), self.ns)
        self.ns.update(ACTIVE=str(self.active), OUTDIR=sandbox.name)
        self.ns["time"] = SimpleNamespace(time=lambda: 10.0, monotonic=lambda: 0.0, sleep=lambda _: None)

    def test_displays_warmup_even_when_not_loaded(self):
        self.ns["qdit_state"] = lambda: {"loaded": False, "busy": True, "phase": "load"}
        self.assertIn("预热", self.ns["status_text"]())
        self.assertNotIn("未驻留", self.ns["status_text"]())

    def test_unknown_job_ends_poll_and_clears_only_its_record(self):
        self.active.write_text(json.dumps({"jid": "new"}))
        replies = iter([{"status": "unknown"}, {"status": "done"}])
        self.ns["_get"] = lambda *a, **kw: next(replies)
        with self.assertRaisesRegex(RuntimeError, "失效"):
            self.ns["_poll"]("old", lambda *a, **kw: None, 0, 1)
        self.assertEqual(json.loads(self.active.read_text()), {"jid": "new"})

    def test_stalled_job_stops_polling_but_preserves_reconnect_record(self):
        self.active.write_text(json.dumps({"jid": "slow"}))
        times = iter([0.0, 1.0, 901.0])
        replies = iter([{"status": "running"}, {"status": "done"}])
        self.ns["time"].monotonic = lambda: next(times)
        self.ns["_get"] = lambda *a, **kw: next(replies)
        with self.assertRaisesRegex(RuntimeError, "任务没有被取消"):
            self.ns["_poll"]("slow", lambda *a, **kw: None, 0, 1)
        self.assertEqual(json.loads(self.active.read_text()), {"jid": "slow"})

    def test_cold_generation_submits_one_job_and_refreshes_history(self):
        self.ns["qdit_state"] = lambda: {"loaded": False}
        calls = []

        def post(path, req):
            calls.append(path)
            return {"job": "generate-1"}

        self.ns["_post"] = post
        self.ns["_poll"] = lambda *a: {"image": "test.png"}
        self.ns["refresh_gallery"] = lambda: ["test.png"]
        result = self.ns["generate"]("a cat")
        self.assertEqual(calls, ["/generate"])
        self.assertEqual((result[0], result[2]), ("test.png", ["test.png"]))


if __name__ == "__main__":
    unittest.main()
