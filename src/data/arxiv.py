"""ArXiv 论文采集与文本提取。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.data.layout import LayoutDocument, parse_pdf, to_sections


def _out_dir(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _download_pdf(url: str, out_dir: Path, name: str, retries: int = 4) -> Path | None:
    """下载 PDF 到 out_dir，429/5xx 时指数退避重试。"""
    headers = {"User-Agent": "AcademicDocAgent/0.1 (mailto:dev@example.com)"}
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, timeout=60, stream=True) as r:
                if r.status_code == 200:
                    target = out_dir / f"{name}.pdf"
                    with target.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk:
                                f.write(chunk)
                    return target
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = 5 * (2**attempt)
                    print(f"[retry] HTTP {r.status_code}，{wait}s 后重试 {name}")
                    time.sleep(wait)
                    continue
                print(f"[skip] {name}: HTTP {r.status_code}")
                return None
        except requests.RequestException as e:
            wait = 3 * (2**attempt)
            print(f"[retry] 连接失败（{e.__class__.__name__}），{wait}s 后重试 {name}")
            time.sleep(wait)
    print(f"[skip] {name}: 重试 {retries} 次仍失败")
    return None


def download_by_ids(arxiv_ids: Iterable[str], download_dir: str | Path) -> list[Path]:
    """按 arXiv ID 批量下载 PDF。"""
    try:
        import arxiv
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 arxiv 包，请安装 requirements-base.txt") from e

    out = _out_dir(download_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    saved: list[Path] = []
    for aid in tqdm(list(arxiv_ids), desc="下载 ArXiv PDF"):
        try:
            try:
                res = next(client.results(arxiv.Search(id_list=[aid])))
                pdf_url = res.pdf_url
            except Exception:  # noqa: BLE001  接口限流时直连 PDF
                pdf_url = f"https://arxiv.org/pdf/{aid}"
            path = _download_pdf(pdf_url, out, aid.replace("/", "_"))
            if path:
                saved.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {aid}: {e}")
        time.sleep(1.5)
    return saved


def search_and_download(
    query: str,
    max_results: int = 100,
    categories: list[str] | None = None,
    download_dir: str | Path = "work/data/arxiv_pdfs",
) -> list[Path]:
    """按查询词（可带分类过滤）搜索并下载。"""
    try:
        import arxiv
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 arxiv 包，请安装 requirements-base.txt") from e

    out = _out_dir(download_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    saved: list[Path] = []
    for res in tqdm(client.results(search), total=max_results, desc="搜索并下载"):
        cats = {str(c) for c in res.categories}
        if categories and not cats.intersection(categories):
            continue
        try:
            aid = res.entry_id.rstrip("/").split("/")[-1]
            path = _download_pdf(res.pdf_url, out, aid)
            if path:
                saved.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {res.entry_id}: {e}")
        time.sleep(3)
    return saved


def extract_pdfs(pdf_dir: str | Path, extract_dir: str | Path) -> list[Path]:
    """把 PDF 解析为结构化文本文件（每篇一个 .txt，含章节标记）。"""
    src = _out_dir(pdf_dir)
    dst = _out_dir(extract_dir)
    dst.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdfs = sorted(src.glob("*.pdf"))
    for pdf in tqdm(pdfs, desc="版面解析"):
        try:
            doc = parse_pdf(pdf)
            txt = _doc_to_text(doc)
            out = dst / (pdf.stem + ".txt")
            out.write_text(txt, encoding="utf-8")
            written.append(out)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {pdf.name}: {e}")
    return written


def _doc_to_text(doc: LayoutDocument) -> str:
    """把版面解析结果渲染为带结构的纯文本。"""
    lines = [f"# TITLE: {doc.title}"]
    if doc.abstract:
        lines += ["", "# ABSTRACT", doc.abstract]

    def walk(sections, level: int) -> None:
        for sec in sections:
            lines.append("")
            lines.append("#" * (level + 1) + f" {sec.title}")
            if sec.text:
                lines.append(sec.text)
            walk(sec.subsections, level + 1)

    walk(to_sections(doc), 1)
    return "\n".join(lines)


def load_extracted(extract_dir: str | Path) -> list[dict]:
    """读取提取后的文本目录，返回 [{path, title, text}]。"""
    d = _out_dir(extract_dir)
    records = []
    for f in sorted(d.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].removeprefix("# TITLE: ") if text else f.stem
        records.append({"path": str(f), "title": title, "text": text})
    return records


def export_manifest(records: list[dict], out: str | Path) -> Path:
    out_path = _out_dir(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path
