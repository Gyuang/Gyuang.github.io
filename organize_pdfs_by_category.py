"""
Organize (copy) PDFs in papers_collection_clean into folders by post category.

For each post in _posts/Paper, we try to find the best-matching PDF under
papers_collection_clean and copy it to papers_by_category/<category>/.

Notes
- We use the second category in front matter (e.g., categories: [paper, medical-ai]).
- Non-destructive: copies files; original PDFs remain.
- Outputs a summary and a mapping JSON for reference.

Usage
  python3 organize_pdfs_by_category.py
  python3 organize_pdfs_by_category.py --dest papers_by_category

This script is intended for local organization; add the destination to .gitignore
to avoid committing large binaries.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def slugify(s: str) -> str:
  s = s.lower()
  s = re.sub(r"[^a-z0-9]+", "-", s)
  return re.sub(r"-+", "-", s).strip("-")


def load_front_matter(path: Path) -> Dict[str, object]:
  text = path.read_text(encoding="utf-8")
  if not text.startswith("---"):
    return {}
  end = text.find("\n---", 3)
  if end == -1:
    return {}
  fm = text[3:end]
  data: Dict[str, object] = {}
  for line in fm.splitlines():
    if ":" not in line:
      continue
    k, v = line.split(":", 1)
    k = k.strip()
    v = v.strip()
    # arrays
    if v == "" and k in ("categories", "tags"):
      # next lines will include dash items; naive parse
      continue
    if k in ("categories", "tags"):
      # accumulate dash items from subsequent lines
      arr: List[str] = []
      # re-scan lines following the current one
      # fallback: if inline YAML, parse simple cases like [a, b]
    data[k] = v
  # naive parse for categories
  cats: List[str] = []
  for line in fm.splitlines():
    if line.strip().startswith("categories:"):
      continue
    if line.strip().startswith("tags:"):
      continue
    if line.strip().startswith("-"):
      item = line.strip()[1:].strip()
      # strip quotes
      item = item.strip('"\'')
      if item:
        cats.append(item)
  if cats:
    data["categories_list"] = cats
  return data


def best_pdf_for_post(post: Path, pdfs: List[Path]) -> Optional[Path]:
  # Build a score by word overlap against post filename slug without date
  name = post.name
  # Remove leading date
  try:
    _, rest = name.split("-", 1)
    rest = rest.split(".", 1)[0]
  except Exception:
    rest = name.rsplit(".", 1)[0]
  post_slug = slugify(rest)
  post_parts = [p for p in post_slug.split("-") if p]
  best: Tuple[int, Optional[Path]] = (0, None)
  for pdf in pdfs:
    pdf_slug = slugify(pdf.stem)
    score = 0
    for part in post_parts:
      if part and part in pdf_slug:
        score += 1
    if score > best[0]:
      best = (score, pdf)
  # Require a minimal overlap to avoid false positives
  return best[1] if best[0] >= 3 else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", default="papers_collection_clean", help="Source PDF root")
  ap.add_argument("--dest", default="papers_by_category", help="Destination root")
  args = ap.parse_args()

  src = Path(args.src)
  dest = Path(args.dest)
  dest.mkdir(parents=True, exist_ok=True)

  pdfs = [p for p in src.rglob("*.pdf")]
  posts = list(Path("_posts/Paper").glob("*.md"))

  mapping = []
  copied = 0
  missed = []
  for post in posts:
    fm = load_front_matter(post)
    cats = fm.get("categories_list") or []
    if isinstance(cats, list) and len(cats) >= 2:
      cat = slugify(str(cats[1]))
    else:
      cat = "uncategorized"
    pdf = best_pdf_for_post(post, pdfs)
    if not pdf:
      missed.append(str(post))
      continue
    target_dir = dest / cat
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / pdf.name
    if not target_path.exists():
      shutil.copy2(pdf, target_path)
      copied += 1
    mapping.append({
      "post": str(post),
      "category": cat,
      "pdf_src": str(pdf),
      "pdf_dest": str(target_path),
    })

  (dest / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

  print(f"Posts: {len(posts)}  PDFs found: {len(pdfs)}  Copied: {copied}")
  if missed:
    print("No PDF match for posts:")
    for m in missed:
      print(" -", m)
  print(f"Destination: {dest}")


if __name__ == "__main__":
  raise SystemExit(main())

