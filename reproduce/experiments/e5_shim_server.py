"""Minimal OpenAI-compatible chat-completions server backed by transformers (paper E5, Table tab:paraphrase-model).

Used for the open-weight paraphraser run when vLLM is unavailable: the paper's
Llama-3.1-8B-Instruct corpus was built through this server (bf16, one A100-40GB,
about 5.4 h). Generation is single-stream behind a lock, so concurrent requests
queue. It implements just what ``build_corpora_dirbucket.py`` needs:
POST ``/v1/chat/completions`` (user messages are joined; ``temperature`` and
``max_tokens`` honoured; sampling uses top_p=0.95) and GET ``/health``. Usage
counts in the response are zero.

Point the builder at it with ``OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`` and
any non-empty ``OPENAI_API_KEY`` (see ``scripts/33_e5_open_weight_paraphraser.sh``).
An OpenAI-compatible vLLM server (``vllm serve <model> --port <port>``) is a
faster drop-in alternative with its own sampler.

Usage:
  python experiments/e5_shim_server.py \
      --model NousResearch/Meta-Llama-3.1-8B-Instruct --port 8123
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--max_new_tokens_cap", type=int, default=900)
    ap.add_argument("--device", type=str, default=None,
                    help="torch device (default: cuda:0 if available, else cpu).")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    help="torch dtype name of the model weights (paper: bfloat16).")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[shim] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model = model.to(device)
    model.eval()
    lock = threading.Lock()
    print("[shim] model ready", flush=True)

    def generate(prompt: str, temperature: float, max_tokens: int) -> str:
        msgs = [{"role": "user", "content": prompt}]
        # return_dict=True is version-robust: transformers 4.x/5.x variously
        # return tensors or BatchEncoding from apply_chat_template.
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=min(int(max_tokens or 700), args.max_new_tokens_cap),
            pad_token_id=tok.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with lock, torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        return tok.decode(out[0][n_in:], skip_special_tokens=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = "\n".join(m["content"] for m in req["messages"]
                                   if m.get("role") == "user")
                text = generate(prompt, float(req.get("temperature", 0.8)),
                                int(req.get("max_tokens", 700)))
                self._send(200, {
                    "id": "shim-1", "object": "chat.completion",
                    "created": int(time.time()), "model": args.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                              "total_tokens": 0},
                })
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[shim] request error: {e}", flush=True)
                traceback.print_exc()
                self._send(500, {"error": {"message": str(e)}})

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[shim] serving {args.model} on {device} at http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
