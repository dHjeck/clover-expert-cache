"""OpenAI-compatible server for Clover-35B-A3B-Preview on a 16 GB consumer GPU.

Endpoints
    GET  /v1/models
    POST /v1/chat/completions      (stream=true supported, SSE)

Why the loader is hand-rolled
    transformers 5.19 + accelerate 1.15 leave every CPU-side weight of this
    custom-code model on the `meta` device when device_map="auto" is used, so
    the first forward dies with "Tensor on device meta". We load real BF16
    weights on the CPU, move the embedding / final norm / lm_head and the first
    N decoder layers to the GPU, and hop activations between devices per layer.

Run
    python clover_server.py --model /path/to/Clover-35B-A3B-Preview --port 8321

Environment
    CLOVER_MODEL     model directory when --model is omitted
    CLOVER_VRAM_GB   budget for decoder layers on the GPU (default 13.0)
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = os.environ.get("CLOVER_MODEL", "")
VRAM_GB = float(os.environ.get("CLOVER_VRAM_GB", "13.0"))
MODEL_ID = "clover-35b-a3b-preview"

state = {"tok": None, "model": None, "lock": threading.Lock(), "gpu_layers": 0, "load_s": None}


def build():
    if not MODEL_PATH:
        raise RuntimeError("Set CLOVER_MODEL or pass --model with a local Clover model directory")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()

    model.model.embed_tokens.to("cuda:0")
    model.lm_head.to("cuda:0")
    model.model.norm.to("cuda:0")

    layers = model.model.layers
    per_layer = sum(p.numel() * p.element_size() for p in layers[0].parameters()) / 1e9
    base = torch.cuda.memory_allocated() / 1e9
    n_gpu = 0
    for layer in layers:
        if base + per_layer * (n_gpu + 1) > VRAM_GB:
            break
        layer.to("cuda:0")
        n_gpu += 1

    LayerCls = type(layers[0])
    orig = LayerCls.forward

    def forward(self, x, cache, valid, prefix_valid, positions, preview):
        dev = self.input_layernorm.weight.device
        src = x.device
        if dev != src:
            x = x.to(dev)
            valid = valid.to(dev)
            prefix_valid = prefix_valid.to(dev)
            positions = positions.to(dev)
        out, new_cache = orig(self, x, cache, valid, prefix_valid, positions, preview)
        if dev != src:
            out = out.to(src)
        return out, new_cache

    LayerCls.forward = forward
    torch.cuda.synchronize()

    state.update(tok=tok, model=model, gpu_layers=n_gpu, load_s=round(time.time() - t0, 1))
    print(f"[clover] ready in {state['load_s']}s, {n_gpu} layers on GPU, "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB VRAM", flush=True)


def encode(messages, enable_thinking):
    prompt = state["tok"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    ids = state["tok"](prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids.to("cuda:0")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p in ("/v1/models", "/models"):
            self._json({"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": int(time.time()),
                "owned_by": "local", "root": MODEL_PATH}]})
        elif p in ("/health", ""):
            self._json({"status": "ok" if state["model"] is not None else "loading",
                        "gpu_layers": state["gpu_layers"], "load_s": state["load_s"]})
        else:
            self._json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._json({"error": {"message": "not found"}}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json({"error": {"message": f"bad json: {e}"}}, 400)

        messages = body.get("messages") or []
        if not messages:
            return self._json({"error": {"message": "messages required"}}, 400)
        if state["model"] is None:
            return self._json({"error": {"message": "model still loading"}}, 503)

        stream = bool(body.get("stream"))
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
        temp = body.get("temperature")
        thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        block = int(body.get("block_size") or 32)
        thresh = float(body.get("threshold") or 0.95)

        ids = encode(messages, thinking)
        kw = {"max_new_tokens": max_new, "block_size": block, "threshold": thresh}
        if temp is not None and float(temp) > 0:
            kw["do_sample"] = True
            kw["temperature"] = float(temp)
            if body.get("top_p") is not None:
                kw["top_p"] = float(body["top_p"])
            if body.get("top_k") is not None:
                kw["top_k"] = int(body["top_k"])

        cid = "chatcmpl-" + uuid.uuid4().hex[:16]
        created = int(time.time())

        if not stream:
            t0 = time.time()
            with state["lock"], torch.inference_mode():
                out = state["model"].generate(input_ids=ids, **kw)
            dt = max(time.time() - t0, 1e-6)
            new = out[0, ids.shape[1]:]
            text = state["tok"].decode(new, skip_special_tokens=True)
            return self._json({
                "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(new.numel()),
                          "total_tokens": int(ids.shape[1]) + int(new.numel())},
                "timings": {"gen_s": round(dt, 2), "tok_per_s": round(int(new.numel()) / dt, 3)},
            })

        streamer = TextIteratorStreamer(state["tok"], skip_prompt=True,
                                        skip_special_tokens=True, timeout=1800)
        err = {}

        def run():
            try:
                with state["lock"], torch.inference_mode():
                    state["model"].generate(input_ids=ids, streamer=streamer, **kw)
            except Exception as e:
                err["e"] = repr(e)
                streamer.end()

        threading.Thread(target=run, daemon=True).start()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def frame(obj):
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        frame({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
               "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        try:
            for piece in streamer:
                if piece:
                    frame({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": MODEL_ID,
                           "choices": [{"index": 0, "delta": {"content": piece},
                                        "finish_reason": None}]})
        except Exception as e:
            frame({"error": {"message": repr(e)}})
        if err:
            frame({"error": {"message": err["e"]}})
        frame({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global MODEL_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CLOVER_MODEL"),
                    help="Local Clover model directory, or set CLOVER_MODEL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--load-only", action="store_true")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model is required (or set CLOVER_MODEL)")
    MODEL_PATH = a.model
    os.environ["CLOVER_MODEL"] = a.model
    build()
    if a.load_only:
        return
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"[clover] listening on http://{a.host}:{a.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
