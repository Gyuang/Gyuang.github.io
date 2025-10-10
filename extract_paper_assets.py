"""
Extract figures and tables from PDFs via Adobe PDF Services, pick key assets
(main architecture figure and main results table), and optionally inject into
matching Jekyll posts under _posts/Paper.

Requirements:
  pip install adobe-pdfservices-sdk

Credentials:
  Provide via --creds <JSON>. Supports both Adobe's OAuth Server-to-Server JSON
  with keys like CLIENT_ID/CLIENT_SECRETS and the newer client_id/client_secret
  schema. You may also set env vars PDF_SERVICES_CLIENT_ID/SECRET.

Usage:
  # Extract + select key assets across a directory of PDFs (dry-run)
  python3 extract_paper_assets.py papers_collection_clean --creds \
      "./papers_collection_clean/733LinenClownfish-3854154-OAuth Server-to-Server.json"

  # Inject selected images into matching posts (append section at bottom)
  python3 extract_paper_assets.py papers_collection_clean --creds <json> \
      --inject-posts

Outputs per PDF (under assets/images/paper/<pdf_base>/):
  - figures.json, tables.json (metadata including caption/page/bbox)
  - fig_##.png, table_##.png (renditions)
  - key_assets.json { architecture_image, results_image, markdown }
"""

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_client_credentials(creds_path: Optional[str]) -> Dict[str, str]:
  env_id = os.getenv("PDF_SERVICES_CLIENT_ID")
  env_secret = os.getenv("PDF_SERVICES_CLIENT_SECRET")

  def parse_json(p: Path) -> Optional[Dict[str, str]]:
    try:
      data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
      return None
    # Support multiple layouts
    cid = (
      data.get("client_id")
      or (data.get("client_credentials") or {}).get("client_id")
      or data.get("CLIENT_ID")
    )
    csec = (
      data.get("client_secret")
      or (data.get("client_credentials") or {}).get("client_secret")
      or (data.get("CLIENT_SECRETS") or [None])[0]
    )
    if cid and csec:
      return {"client_id": cid, "client_secret": csec}
    return None

  if creds_path:
    p = Path(creds_path)
    if not p.exists():
      raise FileNotFoundError(f"Credentials file not found: {p}")
    creds = parse_json(p)
    if not creds:
      raise ValueError("client_id/secret not found in credentials JSON")
    return creds

  # Try any *OAuth Server-to-Server.json in CWD
  for cand in Path.cwd().glob("*OAuth Server-to-Server.json"):
    creds = parse_json(cand)
    if creds:
      return creds

  if env_id and env_secret:
    return {"client_id": env_id, "client_secret": env_secret}

  raise RuntimeError("Adobe PDF credentials missing. Use --creds or set env vars.")


def _require_adobe_sdk():
  try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.execution_context import ExecutionContext
    from adobe.pdfservices.operation.io.file_ref import FileRef
    from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType
  except ImportError:
    print("Missing adobe-pdfservices-sdk. pip install pdfservices-sdk", file=sys.stderr)
    raise
  return {
    "ServicePrincipalCredentials": ServicePrincipalCredentials,
    "ExecutionContext": ExecutionContext,
    "FileRef": FileRef,
    "ExtractPDFOperation": ExtractPDFOperation,
    "ExtractPDFOptions": ExtractPDFOptions,
    "ExtractElementType": ExtractElementType,
    "ExtractRenditionsElementType": ExtractRenditionsElementType,
  }


def _iter_pdfs(path: Path) -> List[Path]:
  if path.is_file() and path.suffix.lower() == ".pdf":
    return [path]
  if path.is_dir():
    return [p for p in path.rglob("*.pdf")]
  return []


def _ensure_dir(p: Path):
  p.mkdir(parents=True, exist_ok=True)


def _copy_rendition(src_root: Path, file_names, dest_dir: Path, prefix: str, index: int, prefer_exts=(".png", ".jpg", ".jpeg")) -> Optional[Path]:
  if not file_names:
    return None
  if isinstance(file_names, str):
    file_names = [file_names]
  # Reorder by preferred extensions first
  ordered = sorted(file_names, key=lambda p: (0 if Path(p).suffix.lower() in prefer_exts else 1, p))
  src = None
  for name in ordered:
    fname = Path(name).name
    candidates = [src_root / name, src_root / "renditions" / fname, src_root / "images" / fname, src_root / "figures" / fname, src_root / "tables" / fname]
    for c in candidates:
      if c.exists():
        if prefer_exts and c.suffix.lower() not in prefer_exts:
          continue
        src = c
        break
    if src:
      break
  if not src:
    # fallback: glob by base name
    hits = []
    for name in ordered:
      fname = Path(name).name
      hits.extend([h for h in src_root.rglob(fname) if (not prefer_exts or h.suffix.lower() in prefer_exts)])
    if hits:
      src = hits[0]
  if not src:
    return None
  _ensure_dir(dest_dir)
  dest = dest_dir / f"{prefix}_{index:02d}{src.suffix.lower()}"
  dest.write_bytes(src.read_bytes())
  return dest


def _normalize_text(s: str) -> str:
  return (s or "").strip().lower()


ARCH_KWS = [
  "architecture", "framework", "pipeline", "overview", "system", "method", "network", "model", "workflow", "approach"
]
RESULT_KWS = [
  "results", "comparison", "performance", "evaluation", "benchmark", "sota", "state-of-the-art", "table", "metrics"
]


def _score_caption(caption: str, kws: List[str]) -> int:
  c = _normalize_text(caption)
  return sum(1 for k in kws if k in c)


def _area(bbox) -> float:
  try:
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0) * (y1 - y0))
  except Exception:
    return -1.0


def _file_index_score(path_str: Optional[str]) -> float:
  """Smaller index gets slightly higher score (prefer early figures/tables).
  Accepts names like fig_04.png, table_239.png or fileoutpart239.png.
  """
  if not path_str:
    return 0.0
  name = Path(path_str).name.lower()
  import re
  m = re.search(r"(?:fig|table|fileoutpart)[_-]?([0-9]{1,4})", name)
  if not m:
    return 0.0
  try:
    idx = int(m.group(1))
  except Exception:
    return 0.0
  # Map smaller index to larger score in a bounded way
  return max(0.0, 10.0 - min(999, idx) * 0.01)


def _load_overrides() -> Dict[str, Dict[str, object]]:
  p = Path("_data/main_assets_overrides.yml")
  if not p.exists():
    return {}
  try:
    import yaml  # type: ignore
  except Exception:
    # Lazy YAML parser: the file will be simple key: [..] value pairs; fallback to JSON if user prefers
    try:
      return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
      return {}
  try:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
  except Exception:
    return {}


def extract_one(pdf: Path, creds: Dict[str, str], sdk, out_base: Path) -> Dict[str, object]:
  ServicePrincipalCredentials = sdk["ServicePrincipalCredentials"]
  ExecutionContext = sdk["ExecutionContext"]
  FileRef = sdk["FileRef"]
  ExtractPDFOperation = sdk["ExtractPDFOperation"]
  ExtractPDFOptions = sdk["ExtractPDFOptions"]
  ExtractElementType = sdk["ExtractElementType"]
  ExtractRenditionsElementType = sdk["ExtractRenditionsElementType"]

  creds_obj = ServicePrincipalCredentials(creds["client_id"], creds["client_secret"])
  ctx = ExecutionContext.create(creds_obj)
  input_ref = FileRef.create_from_local_file(str(pdf))
  op = ExtractPDFOperation.create_new()
  op.set_input(input_ref)
  options = ExtractPDFOptions.builder() \
    .with_elements_to_extract([ExtractElementType.TEXT, ExtractElementType.TABLES]) \
    .with_elements_to_extract_renditions([ExtractRenditionsElementType.FIGURES, ExtractRenditionsElementType.TABLES]) \
    .build()
  op.set_options(options)
  result = op.execute(ctx)

  tmp_root = Path(".pdf_extract") / pdf.stem
  _ensure_dir(tmp_root)
  zip_path = tmp_root / "extract.zip"
  if zip_path.exists():
    try:
      zip_path.unlink()
    except Exception:
      pass
  result.save_as(str(zip_path))
  with zipfile.ZipFile(zip_path) as z:
    z.extractall(tmp_root)

  sd_path = tmp_root / "structuredData.json"
  data = json.loads(sd_path.read_text(encoding="utf-8"))
  # Build page→text elements for simple caption lookup
  page_texts: Dict[int, List[Dict[str, object]]] = {}
  for el in data.get("elements", []):
    if "Text" in el and el.get("Bounds") is not None:
      page_texts.setdefault(int(el.get("Page", -1)), []).append(el)

  out_dir = out_base / pdf.stem
  _ensure_dir(out_dir)
  figures, tables = [], []

  for el in data.get("elements", []):
    path = (el.get("Path") or "").lower()
    etype = (el.get("Type") or el.get("type") or "").lower()
    is_fig = "figure" in path or etype == "figure"
    is_tbl = "table" in path or etype == "table"
    if not (is_fig or is_tbl):
      continue
    bbox = el.get("Bounds") or el.get("BoundingBox") or el.get("bounds")
    page = int(el.get("Page", -1) or -1)
    # Gather rendition file from filePaths if present
    fps = el.get("filePaths") or []
    file_name = fps if fps else (el.get("FileName") or el.get("fileName") or "")
    # Heuristic caption: nearest text line below the element that horizontally overlaps
    caption = ""
    if bbox and page in page_texts:
      x0, y0, x1, y1 = bbox
      best = None
      for t in page_texts[page]:
        tb = t.get("Bounds")
        if not tb:
          continue
        tx0, ty0, tx1, ty1 = tb
        # Horizontal overlap
        if tx1 < x0 or tx0 > x1:
          continue
        # Prefer text just below the figure (ty1 close to y0)
        dy_below = abs(ty1 - y0)
        dy_above = abs(ty0 - y1)
        # below within 80pt or above within 40pt
        score = None
        if ty1 <= y0 and dy_below <= 80:
          score = (0, dy_below)
        elif ty0 >= y1 and dy_above <= 40:
          score = (1, dy_above)
        if score is not None:
          if best is None or score < best[0]:
            best = (score, t)
      if best:
        caption = (best[1].get("Text") or "").strip()
    item = {
      "page": page,
      "bbox": bbox,
      "caption": caption,
      "file_name": file_name,
    }
    if is_fig:
      figures.append(item)
    if is_tbl:
      tables.append(item)

  copied_figs, copied_tabs = [], []
  for idx, f in enumerate(figures, 1):
    p = _copy_rendition(tmp_root, f.get("file_name", ""), out_dir, "fig", idx, prefer_exts=(".png", ".jpg", ".jpeg"))
    if p:
      f["output_file"] = str(p)
      copied_figs.append(p)
  for idx, t in enumerate(tables, 1):
    p = _copy_rendition(tmp_root, t.get("file_name", ""), out_dir, "table", idx, prefer_exts=(".png", ".jpg", ".jpeg"))
    if p:
      t["output_file"] = str(p)
      copied_tabs.append(p)

  (out_dir / "figures.json").write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
  (out_dir / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")

  # Overrides per PDF (by stem)
  overrides = _load_overrides().get(pdf.stem, {})

  def _pick_by_override(candidate_list: List[Dict[str, object]], override_val) -> Optional[Dict[str, object]]:
    if not override_val:
      return None
    names = override_val
    if isinstance(names, str):
      names = [names]
    names = [n.lower() for n in names]
    for n in names:
      for c in candidate_list:
        out = (c.get("output_file") or "").lower()
        base = Path(out).name.lower()
        if n in out or n in base:
          return c
    return None

  # Select key assets with improved scoring
  def arch_key(x):
    return (
      _score_caption(x.get("caption", ""), ARCH_KWS) + _file_index_score(x.get("output_file")) * 0.5,
      _area(x.get("bbox")),
    )
  arch = _pick_by_override(figures, overrides.get("arch")) or (sorted(figures, key=arch_key, reverse=True)[0] if figures else None)

  def res_key(x):
    return (
      _score_caption(x.get("caption", ""), RESULT_KWS) + _file_index_score(x.get("output_file")) * 0.2,
      _area(x.get("bbox")),
    )
  res_primary = _pick_by_override(tables or figures, overrides.get("results"))
  if not res_primary:
    res_primary = (sorted((tables or figures), key=res_key, reverse=True)[0] if (tables or figures) else None)
  # Optionally a second results asset
  res_secondary = None
  res_list = overrides.get("results")
  if isinstance(res_list, list) and len(res_list) > 1:
    # Try to pick the next match in order
    for name in res_list:
      cand = _pick_by_override(tables or figures, name)
      if cand and cand is not res_primary:
        res_secondary = cand
        break

  md_lines = []
  if arch and arch.get("output_file"):
    rel = "/" + str(Path(arch["output_file"]).as_posix())
    md_lines.append("### Main Architecture")
    md_lines.append(f"![Architecture]({rel})")
  if res_primary and res_primary.get("output_file"):
    rel = "/" + str(Path(res_primary["output_file"]).as_posix())
    md_lines.append("")
    md_lines.append("### Main Results Table")
    md_lines.append(f"![Results]({rel})")
  if res_secondary and res_secondary.get("output_file"):
    rel2 = "/" + str(Path(res_secondary["output_file"]).as_posix())
    md_lines.append("")
    md_lines.append("### Main Results Table (2)")
    md_lines.append(f"![Results-2]({rel2})")
  markdown = "\n".join(md_lines)
  key = {
    "architecture": arch,
    "results": res_primary,
    "results_2": res_secondary,
    "markdown": markdown,
  }
  (out_dir / "key_assets.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

  return {
    "pdf": str(pdf),
    "out_dir": str(out_dir),
    "architecture_image": arch.get("output_file") if arch else None,
    "results_image": res_primary.get("output_file") if res_primary else None,
    "markdown": markdown,
  }


def _slugify(s: str) -> str:
  s = s.lower()
  s = re.sub(r"[^a-z0-9]+", "-", s)
  return re.sub(r"-+", "-", s).strip("-")


def _find_post_for_pdf(pdf: Path) -> Optional[Path]:
  base = pdf.stem
  slug = _slugify(base)
  posts = list(Path("_posts/Paper").glob("*.md"))
  # match any post where slug (or its words) appear in filename
  best = None
  for p in posts:
    name = p.name.lower()
    score = 0
    for part in slug.split("-"):
      if part and part in name:
        score += 1
    if score > 2 and (best is None or score > best[0]):
      best = (score, p)
  return best[1] if best else None


def _inject_into_post(post_path: Path, markdown: str) -> bool:
  txt = post_path.read_text(encoding="utf-8")
  # Replace existing section if present
  start_tag = "\n## 주요 도식/표"
  idx = txt.find(start_tag)
  block = "\n\n## 주요 도식/표\n\n" + markdown.strip() + "\n"
  if idx >= 0:
    # Find next section header or end
    next_idx = txt.find("\n## ", idx + 1)
    if next_idx < 0:
      next_idx = len(txt)
    new_txt = txt[:idx] + block + txt[next_idx:]
    changed = new_txt != txt
    if changed:
      post_path.write_text(new_txt, encoding="utf-8")
    return changed
  # Else append
  if markdown.strip() in txt:
    return False
  post_path.write_text(txt + block, encoding="utf-8")
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("input", help="PDF file or directory")
  ap.add_argument("--creds", help="Adobe OAuth Server-to-Server JSON", default=None)
  ap.add_argument("--out", default=str(Path("assets") / "images" / "paper"))
  ap.add_argument("--inject-posts", action="store_true", help="Append selected images to matching posts")
  args = ap.parse_args()

  creds = _load_client_credentials(args.creds)
  sdk = _require_adobe_sdk()
  pdfs = _iter_pdfs(Path(args.input))
  if not pdfs:
    print("No PDFs found.")
    return 1
  out_base = Path(args.out)
  out_base.mkdir(parents=True, exist_ok=True)

  print(f"Processing {len(pdfs)} PDF(s)...")
  injected = 0
  for i, pdf in enumerate(sorted(pdfs), 1):
    try:
      print(f"[{i}/{len(pdfs)}] {pdf}")
      info = extract_one(pdf, creds, sdk, out_base)
      print(f"  arch={info['architecture_image']}  results={info['results_image']}")
      if args.inject_posts and info["markdown"].strip():
        post = _find_post_for_pdf(pdf)
        if post:
          if _inject_into_post(post, info["markdown"]):
            injected += 1
            print(f"  injected → {post}")
        else:
          print("  no matching post found")
    except Exception as e:
      print(f"  ! Failed: {pdf} — {e}", file=sys.stderr)
  if args.inject_posts:
    print(f"Injected into {injected} post(s)")
  print("Done.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
