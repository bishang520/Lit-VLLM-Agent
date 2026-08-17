"""FastAPI 服务：OpenAI 兼容接口 + RAG 检索 + 引导解码 + SSE 流式。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import PROJECT_ROOT, load_config
from src.rag.retriever import RetrievalResult
from src.serving.guided_decode import extract_json, load_response_schema, validate_json

RAG_SYSTEM_PROMPT = (
    "你是严谨的学术文献助手。必须严格基于下方“检索片段”作答，禁止使用片段之外的知识编造。"
    "输出必须是 JSON，结构如下："
    "{\"answer\": 最终回答, \"reasoning\": 逐步推理, "
    "\"citations\": [{\"chunk_id\": 片段编号, \"confidence\": 0-1}], "
    "\"confidence\": 0-1, \"is_grounded\": true/false}。"
    "若片段不足，将 is_grounded 置为 false 并明确说明。"
)


class ChatBody(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class RetrieveBody(BaseModel):
    query: str
    top_k: int = 5
    method: str = "hybrid"  # dense / sparse / hybrid


def format_context(results: list[Any]) -> str:
    parts = []
    for r in results:
        c = r.chunk
        parts.append(
            f"[{c.chunk_id}] 章节：{c.section_path or '无'}，页码：{c.page}\n{c.text}"
        )
    return "\n\n".join(parts)


def mock_answer(query: str, results: list[Any], schema: dict) -> dict:
    """无模型时的模板回答：使用检索片段拼接，保证 JSON 合法。"""
    top = results[0].chunk if results else None
    payload = {
        "answer": (
            f"检索到 {len(results)} 个相关片段。"
            + (f"最相关片段：{top.text[:180]}..." if top else "但未检索到相关内容。")
        ),
        "reasoning": "1. 对问题做混合检索；2. 用 RRF 融合稠密/稀疏结果；3. 取重排后的片段作为答案依据。",
        "citations": [
            {"chunk_id": r.chunk.chunk_id, "confidence": round(float(r.score), 3)}
            for r in results[:3]
        ],
        "confidence": round(float(results[0].score), 3) if results else 0.0,
        "is_grounded": bool(results),
    }
    ok, err = validate_json(payload, schema)
    if not ok:
        payload["answer"] = f"{payload['answer']}（mock 校验提示：{err}）"
    return payload


def _openai_response(payload: dict) -> dict:
    content = json.dumps(payload, ensure_ascii=False)
    return {
        "id": "chatcmpl-academic",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }


def sse_stream(backend, messages, schema, temperature, max_tokens, mock_payload=None):
    if mock_payload is not None:
        content = json.dumps(mock_payload, ensure_ascii=False)
        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': content}}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return
    yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}}]}, ensure_ascii=False)}\n\n"
    for token in backend.stream(messages, schema, temperature, max_tokens):
        yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(retriever=None, backend=None, schema=None, mock: bool = False):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    schema = schema or load_response_schema()
    app = FastAPI(title="Academic Document Agent", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "mock": mock, "retriever": retriever is not None}

    @app.post("/v1/retrieve")
    def retrieve(body: RetrieveBody):
        if retriever is None:
            raise HTTPException(status_code=400, detail="未配置检索器")
        if body.method == "dense":
            qv = retriever.embedder.encode([body.query])[0]
            hits = retriever.dense_store.search(qv, top_k=body.top_k)
            results = [
                RetrievalResult(chunk=c, score=s, rank=i + 1, sources=["dense"])
                for i, (c, s) in enumerate(hits)
            ]
        elif body.method == "sparse":
            hits = retriever.bm25.search(body.query, top_k=body.top_k)
            results = [
                RetrievalResult(chunk=c, score=s, rank=i + 1, sources=["sparse"])
                for i, (c, s) in enumerate(hits)
            ]
        else:
            results = retriever.retrieve(body.query, top_k=body.top_k)
        return {
            "query": body.query,
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "text": r.chunk.text[:500],
                    "section": r.chunk.section_path,
                    "page": r.chunk.page,
                    "score": round(float(r.score), 4),
                    "sources": r.sources,
                }
                for r in results
            ],
        }

    @app.post("/v1/chat/completions")
    def chat(body: ChatBody):
        query = ""
        for m in reversed(body.messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        results = retriever.retrieve(query, top_k=5) if retriever else []
        context = format_context(results)
        user_content = f"检索片段：\n{context}\n\n问题：{query}" if context else f"问题：{query}"
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        temperature = body.temperature if body.temperature is not None else 0.2
        max_tokens = body.max_tokens or 1024

        if backend is None:
            payload = mock_answer(query, results, schema)
            if body.stream:
                return StreamingResponse(
                    sse_stream(None, messages, schema, temperature, max_tokens, mock_payload=payload),
                    media_type="text/event-stream",
                )
            return _openai_response(payload)

        if body.stream:
            return StreamingResponse(
                sse_stream(backend, messages, schema, temperature, max_tokens),
                media_type="text/event-stream",
            )
        text = backend.generate(messages, schema, temperature, max_tokens)
        payload = extract_json(text)
        if payload is None:
            return {
                **_openai_response(
                    {
                        "answer": text,
                        "reasoning": "",
                        "citations": [],
                        "confidence": 0.0,
                        "is_grounded": False,
                    }
                ),
                "warning": "模型输出不是合法 JSON，已原样返回",
            }
        return _openai_response(payload)

    return app


class VLLMBackend:
    """vLLM 推理后端：引导解码 + 流式。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._apply_transformers_v5_compat()
        self.llm, self.tokenizer = self._build()

    @staticmethod
    def _apply_transformers_v5_compat() -> None:
        """transformers 5.x 移除了 all_special_tokens_extended，vLLM 0.8.x 仍在使用。"""
        try:
            import transformers
            from transformers import Qwen2Tokenizer, Qwen2TokenizerFast

            for cls in (Qwen2TokenizerFast, Qwen2Tokenizer):
                if not hasattr(cls, "all_special_tokens_extended"):
                    cls.all_special_tokens_extended = property(
                        lambda self: tuple(self.all_special_tokens)
                    )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] transformers 兼容垫片未生效: {e}")

    def _build(self):
        from transformers import AutoTokenizer
        from vllm import LLM

        kwargs = {
            "model": self.cfg["model"],
            "task": self.cfg.get("task", "generate"),
            "tensor_parallel_size": self.cfg.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": self.cfg.get("gpu_memory_utilization", 0.9),
            "max_model_len": self.cfg.get("max_model_len", 8192),
            "dtype": self.cfg.get("dtype", "auto"),
            "trust_remote_code": True,
        }
        if self.cfg.get("quantization"):
            kwargs["quantization"] = self.cfg["quantization"]
        if self.cfg.get("guided_decoding_backend"):
            kwargs["guided_decoding_backend"] = self.cfg["guided_decoding_backend"]
        if self.cfg.get("enforce_eager"):
            kwargs["enforce_eager"] = True
        tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"], trust_remote_code=True)
        return LLM(**kwargs), tokenizer

    def _prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling(self, schema, temperature, max_tokens):
        from vllm import SamplingParams

        base = dict(temperature=temperature, max_tokens=max_tokens)
        try:  # vLLM >= 0.8：GuidedDecodingParams
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(
                **base,
                guided_decoding=GuidedDecodingParams(
                    json=schema,
                    backend=self.cfg.get("guided_decoding_backend", "xgrammar"),
                ),
            )
        except ImportError:
            pass
        try:  # vLLM <= 0.7：guided_json 直接参数
            return SamplingParams(**base, guided_json=schema)
        except TypeError:
            raise RuntimeError(
                "当前 vLLM 版本不支持引导解码，请升级 vLLM 或检查引导解码后端"
            )

    def generate(self, messages, schema, temperature, max_tokens) -> str:
        sp = self._sampling(schema, temperature, max_tokens)
        out = self.llm.generate([self._prompt(messages)], sp)
        return out[0].outputs[0].text

    def stream(self, messages, schema, temperature, max_tokens):
        """流式生成；vLLM 离线接口不支持 stream 时一次性返回完整结果。"""
        try:
            sp = self._sampling(schema, temperature, max_tokens)
            last = 0
            for res in self.llm.generate([self._prompt(messages)], sp, stream=True):
                text = res.outputs[0].text
                yield text[last:]
                last = len(text)
        except TypeError:
            # vLLM < 0.8 用 stream=True；0.8.x 离线接口不支持，退回整段返回
            yield self.generate(messages, schema, temperature, max_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学术文档 Agent API 服务")
    parser.add_argument("--config", default="configs/serving/vllm.yaml")
    parser.add_argument("--mock", action="store_true", help="不加载模型，返回模板答案（本机联调）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--retriever-config", default="configs/rag/retriever.yaml")
    parser.add_argument("--no-retriever", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    server_cfg = cfg.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8000)

    schema = load_response_schema(
        server_cfg.get("response_schema", "configs/serving/answer_schema.json")
    )

    retriever = None
    if not args.no_retriever:
        try:
            from src.rag.factory import build_retriever

            retriever = build_retriever(args.retriever_config)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 检索器加载失败（API 将不执行检索）: {e}")

    backend = None
    if not args.mock:
        backend = VLLMBackend(cfg)

    app = create_app(retriever=retriever, backend=backend, schema=schema, mock=args.mock)

    import uvicorn

    print(f"启动 API：http://{host}:{port}（mock={args.mock}）")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    sys.exit(main())
