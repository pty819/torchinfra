"""CPU-only lifecycle tests: run with python -m pytest tests/test_qdit_lifecycle.py."""
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))




class WorkerTest(unittest.TestCase):
    def setUp(self):
        # Only the model/runtime boundary is faked; use the real HTTP server and worker.
        torch = ModuleType("torch")
        torch.no_grad = lambda: lambda fn: fn
        torch.cuda = SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)
        dynamo = ModuleType("torch._dynamo")
        dynamo.config = SimpleNamespace()
        with patch.dict(sys.modules, {"torch": torch, "torch._dynamo": dynamo}), patch.object(sys, "argv", ["serve.py"]):
            import qwen21dit.serve as serve
            self.serve = serve
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.residency = []

        def load_models(prompts):
            self.serve.S.model_c = object()  # same order as real load + warmup
            self.residency.append(("load", threading.get_ident()))
            self.started.set()
            if self.block:
                self.release.wait(3)
            self.serve.S.loaded_at = time.time()
            self.serve.S.last_used = time.time()
            return {"ok": True}

        def generate(jid, req):
            if self.block:
                self.started.set()
                self.release.wait(3)
            self.serve.S.jobs[jid]["image"] = "test.png"

        torch.cuda.empty_cache = lambda: self.residency.append(("unload", threading.get_ident()))
        self.serve.do_load = load_models
        self.serve.do_generate = generate
        self.serve.IDLE_S = 0.0
        self.server = self.serve.ThreadingHTTPServer(("127.0.0.1", 0), self.serve.H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.worker = threading.Thread(target=self.serve.worker, daemon=True)
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.release.set)

    def request(self, path, body=None):
        url = f"http://127.0.0.1:{self.server.server_port}{path}"
        req = Request(url, data=None if body is None else json.dumps(body).encode())
        with urlopen(req, timeout=2) as r:
            return json.load(r)

    def submit(self, kind, **body):
        return self.request(f"/{kind}", body)["job"]

    def until(self, fn, timeout=2):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            result = fn()
            if result:
                return result
            time.sleep(0.01)
        self.fail("condition not reached")

    def finish(self, jid):
        def done():
            job = self.request(f"/job/{jid}")
            return job if job["status"] in ("done", "error") else None
        return self.until(done)

    def test_warmup_is_busy_not_loaded_and_rejects_unload(self):
        self.block = True
        jid = self.submit("load")
        self.assertTrue(self.started.wait(2))
        st = self.request("/state")
        self.assertEqual((st["loaded"], st["busy"], st["idle_s"]), (False, True, None))
        self.assertEqual(self.request(f"/job/{jid}")["status"], "running")
        with self.assertRaises(HTTPError) as cm:
            self.submit("unload")
        with cm.exception as error:
            self.assertEqual(error.code, 409)
        self.release.set()
        self.assertEqual(self.finish(jid)["status"], "done")

    def test_generation_atomically_reloads_after_manual_unload(self):
        self.assertEqual(self.finish(self.submit("load"))["status"], "done")
        self.assertEqual(self.finish(self.submit("unload"))["reason"], "manual")
        self.assertFalse(self.request("/state")["loaded"])
        job = self.finish(self.submit("generate", prompt="a cat"))
        self.assertEqual((job["status"], job.get("image")), ("done", "test.png"))

    def test_idle_unload_runs_on_worker_only_after_generation(self):
        self.serve.IDLE_S = 0.25
        self.assertEqual(self.finish(self.submit("load"))["status"], "done")
        self.block = True
        self.started.clear()
        jid = self.submit("generate", prompt="a cat")
        self.assertTrue(self.started.wait(2))
        time.sleep(0.3)
        self.assertTrue(self.request("/state")["busy"])
        self.assertIsNone(self.request("/state")["idle_s"])
        self.assertEqual(self.residency, [("load", self.worker.ident)])
        self.release.set()
        self.assertEqual(self.finish(jid)["status"], "done")
        self.until(lambda: not self.request("/state")["loaded"] and not self.request("/state")["busy"])
        self.assertEqual(self.residency, [("load", self.worker.ident), ("unload", self.worker.ident)])

    def test_generate_arriving_during_idle_unload_reloads_on_worker(self):
        self.serve.IDLE_S = 0.15
        unloading = threading.Event()
        empty_cache = self.serve.torch.cuda.empty_cache

        def pause_unload():
            unloading.set()
            self.release.wait(3)
            empty_cache()

        self.serve.torch.cuda.empty_cache = pause_unload
        self.assertEqual(self.finish(self.submit("load"))["status"], "done")
        self.assertTrue(unloading.wait(2))
        jid = self.submit("generate", prompt="a cat")
        self.assertEqual(self.request(f"/job/{jid}")["status"], "queued")
        self.assertEqual(self.request("/state")["phase"], "unload")
        self.release.set()
        job = self.finish(jid)
        self.assertEqual((job["status"], job.get("image")), ("done", "test.png"))
        self.assertEqual(self.residency[:3], [(k, self.worker.ident) for k in ("load", "unload", "load")])

    def test_failed_load_leaves_worker_unloaded_and_can_retry(self):
        load_models = self.serve.do_load

        def fail(prompts):
            self.serve.S.model_c = object()
            raise RuntimeError("warmup failed")

        self.serve.do_load = fail
        self.assertEqual(self.finish(self.submit("load"))["status"], "error")
        self.assertFalse(self.request("/state")["loaded"])
        self.assertIsNone(self.serve.S.model_c)
        self.serve.do_load = load_models
        self.assertEqual(self.finish(self.submit("load"))["status"], "done")

    def test_disabled_idle_stays_resident(self):
        self.assertEqual(self.finish(self.submit("load"))["status"], "done")
        time.sleep(0.3)
        st = self.request("/state")
        self.assertTrue(st["loaded"])
        self.assertIsNone(st["idle_limit_s"])


if __name__ == "__main__":
    unittest.main()
