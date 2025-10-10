"""
Extract only figure renditions (PNG) and captions from PDFs using Adobe PDF Services.

Usage examples:
  # Single PDF
  python3 extract_figures_only.py TRIPLEX_CVPR2024.pdf \
    --creds "./733LinenClownfish-3854154-OAuth Server-to-Server.json"

  # Process all PDFs in a directory
  python3 extract_figures_only.py ./papers_dir \
    --creds ~/.adobe/pdfservices_oauth.json

Outputs:
  - Intermediate: .pdf_extract/<pdf_base>/ (unzipped job output, structuredData.json, renditions/..)
  - Final figures: assets/images/paper/<pdf_base>/fig_##.png
  - Summary: assets/images/paper/<pdf_base>/figures.json (page, bbox, caption, filename)

Notes:
  - Credentials are NOT committed. Point --creds to your local JSON file downloaded from
    Adobe "OAuth Server-to-Server". This script will read client_id/client_secret from it.
  - You can also set env vars PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET instead.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


def _load_client_credentials(creds_path: Optional[str]) -> Dict[str, str]:
  """Resolve client_id and client_secret from JSON or environment.

  Priority:
    1) --creds JSON (OAuth Server-to-Server)
    2) Env PDF_SERVICES_CLIENT_ID / PDF_SERVICES_CLIENT_SECRET
    3) Default basename in CWD that matches *OAuth Server-to-Server.json
  """
  # 2) Environment
  env_id = os.getenv("PDF_SERVICES_CLIENT_ID")
  env_secret = os.getenv("PDF_SERVICES_CLIENT_SECRET")

  # 1) Explicit creds JSON
  if creds_path:
    path = Path(creds_path)
    if not path.exists():
      raise FileNotFoundError(f"Credentials file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
    client_id = data.get("client_id") or (data.get("client_credentials") or {}).get("client_id")
    client_secret = data.get("client_secret") or (data.get("client_credentials") or {}).get("client_secret")
    if not client_id or not client_secret:
      raise ValueError("client_id/client_secret not found in credentials JSON")
    return {"client_id": client_id, "client_secret": client_secret}

  # 3) Try to find a default creds JSON in CWD
  for cand in Path.cwd().glob("*OAuth Server-to-Server.json"):
    try:
      with cand.open("r", encoding="utf-8") as f:
        data = json.load(f)
      client_id = data.get("client_id") or (data.get("client_credentials") or {}).get("client_id")
      client_secret = data.get("client_secret") or (data.get("client_credentials") or {}).get("client_secret")
      if client_id and client_secret:
        return {"client_id": client_id, "client_secret": client_secret}
    except Exception:
      continue

  # Env fallback
  if env_id and env_secret:
    return {"client_id": env_id, "client_secret": env_secret}

  raise RuntimeError(
    "Adobe PDF Services credentials not provided. Use --creds <json> or set PDF_SERVICES_CLIENT_ID/SECRET."
  )


def _require_adobe_sdk():
  """Support the public adobe.pdfservices.operation v2.x API."""
  try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.execution_context import ExecutionContext
    from adobe.pdfservices.operation.io.file_ref import FileRef
    from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType
  except ImportError:
    print("Missing adobe-pdfservices-sdk. Install: pip install pdfservices-sdk", file=sys.stderr)
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


def _iter_pdfs(input_path: Path) -> List[Path]:
  if input_path.is_file() and input_path.suffix.lower() == ".pdf":
    return [input_path]
  if input_path.is_dir():
    return [p for p in input_path.rglob("*.pdf")]
  raise FileNotFoundError(f"No PDF found at {input_path}")


def _ensure_dir(p: Path):
  p.mkdir(parents=True, exist_ok=True)


def _copy_rendition(src_dir: Path, file_name: str, dest_dir: Path, index: int, pdf_base: str) -> Optional[Path]:
  # Adobe SDK typically places renditions under renditions/; try common locations
  candidates = [src_dir / file_name, src_dir / "renditions" / file_name, src_dir / "images" / file_name, src_dir / "figures" / file_name]
  src = None
  for c in candidates:
    if c.exists():
      src = c
      break
  if not src:
    # try basename-only search
    for c in src_dir.rglob(Path(file_name).name):
      if c.is_file():
        src = c
        break
  if not src:
    return None
  _ensure_dir(dest_dir)
  dest = dest_dir / f"{pdf_base}_fig_{index:02d}{src.suffix.lower()}"
  dest.write_bytes(src.read_bytes())
  return dest


def process_pdf(pdf_path: Path, creds: Dict[str, str], sdk_mods: Dict[str, object], out_base: Path) -> Dict[str, object]:
  ServicePrincipalCredentials = sdk_mods["ServicePrincipalCredentials"]
  ExecutionContext = sdk_mods["ExecutionContext"]
  FileRef = sdk_mods["FileRef"]
  ExtractPDFOperation = sdk_mods["ExtractPDFOperation"]
  ExtractPDFOptions = sdk_mods["ExtractPDFOptions"]
  ExtractElementType = sdk_mods["ExtractElementType"]
  ExtractRenditionsElementType = sdk_mods["ExtractRenditionsElementType"]

  creds_obj = ServicePrincipalCredentials(creds["client_id"], creds["client_secret"])
  ctx = ExecutionContext.create(creds_obj)

  input_ref = FileRef.create_from_local_file(str(pdf_path))
  op = ExtractPDFOperation.create_new()
  op.set_input(input_ref)

  options = ExtractPDFOptions.builder() \
    .with_elements_to_extract([ExtractElementType.FIGURES]) \
    .with_elements_to_extract_renditions([ExtractRenditionsElementType.FIGURES]) \
    .build()
  op.set_options(options)

  result = op.execute(ctx)

  tmp_root = Path(".pdf_extract") / pdf_path.stem
  _ensure_dir(tmp_root)
  zip_path = tmp_root / "extract.zip"
  result.save_as(str(zip_path))

  with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(tmp_root)

  sd_path = tmp_root / "structuredData.json"
  if not sd_path.exists():
    raise FileNotFoundError(f"structuredData.json not found under {tmp_root}")
  data = json.loads(sd_path.read_text(encoding="utf-8"))

  figures: List[Dict[str, object]] = []
  for el in data.get("elements", []):
    if (el.get("Type") or el.get("type") or "").lower() == "figure":
      file_name = el.get("FileName") or el.get("fileName") or ""
      caption = el.get("Caption") or el.get("caption") or ""
      page = el.get("Page") or el.get("page")
      bbox = el.get("Bounds") or el.get("BoundingBox") or el.get("bounds")
      figures.append({
        "page": page,
        "bbox": bbox,
        "caption": (caption or "").strip(),
        "file_name": file_name,
      })

  # Copy renditions to assets/images/paper/<pdf_base>/
  dest_dir = out_base / pdf_path.stem
  copied = []
  for idx, fig in enumerate(figures, 1):
    if not fig.get("file_name"):
      continue
    copied_path = _copy_rendition(tmp_root, str(fig["file_name"]), dest_dir, idx, pdf_path.stem)
    if copied_path:
      fig["output_file"] = str(copied_path)
      copied.append(copied_path)

  # Write summary JSON alongside images
  summary_path = dest_dir / "figures.json"
  _ensure_dir(dest_dir)
  summary_path.write_text(json.dumps({
    "pdf": str(pdf_path),
    "count": len(figures),
    "figures": figures,
  }, ensure_ascii=False, indent=2), encoding="utf-8")

  return {
    "pdf": str(pdf_path),
    "summary": str(summary_path),
    "images": [str(p) for p in copied],
    "tmp": str(tmp_root),
  }


def main():
  parser = argparse.ArgumentParser(description="Extract figures from PDFs via Adobe PDF Services")
  parser.add_argument("input", help="PDF file or directory containing PDFs")
  parser.add_argument("--creds", help="Path to Adobe OAuth Server-to-Server JSON", default=None)
  parser.add_argument(
    "--out",
    help="Destination base directory for images",
    default=str(Path("assets") / "images" / "paper"),
  )
  args = parser.parse_args()

  input_path = Path(args.input)
  out_base = Path(args.out)

  creds = _load_client_credentials(args.creds)
  sdk_mods = _require_adobe_sdk()

  pdfs = _iter_pdfs(input_path)
  if not pdfs:
    print("No PDFs found.")
    return 1

  print(f"Processing {len(pdfs)} PDF(s) → {out_base}")
  results = []
  for i, pdf in enumerate(sorted(pdfs), 1):
    try:
      print(f"[{i}/{len(pdfs)}] {pdf}")
      res = process_pdf(pdf, creds, sdk_mods, out_base)
      print(f"  → {res['summary']}  (+{len(res['images'])} images)")
      results.append(res)
    except Exception as e:
      print(f"  ! Failed: {pdf} — {e}", file=sys.stderr)

  print("Done.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
