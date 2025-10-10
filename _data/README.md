Data and Helper Scripts
=======================

This site sometimes embeds assets extracted from research PDFs (figures, captions).
Below is the lightweight pipeline used during authoring. Credentials and PDFs are
not tracked by Git.

Figure extraction (Adobe PDF Services)
- Script: `extract_figures_only.py` (repo root)
- Requires: `pip install adobe-pdfservices-sdk`
- Credentials: Use your local Adobe “OAuth Server-to-Server” JSON. Do NOT commit it.

Example usage
- Single PDF:
  - `python3 extract_figures_only.py TRIPLEX_CVPR2024.pdf --creds \
     "./733LinenClownfish-3854154-OAuth Server-to-Server.json"`
- Directory of PDFs:
  - `python3 extract_figures_only.py ./my_papers --creds ~/.adobe/pdfservices_oauth.json`

Outputs
- Intermediate (not committed): `.pdf_extract/<pdf_base>/...`
- Final images: `assets/images/paper/<pdf_base>/fig_##.png`
- Summary metadata: `assets/images/paper/<pdf_base>/figures.json`

Notes
- Credentials are discovered via `--creds` first, then environment variables
  `PDF_SERVICES_CLIENT_ID` / `PDF_SERVICES_CLIENT_SECRET`, and finally any file in
  the CWD matching `*OAuth Server-to-Server.json`.
- If you don’t want images in Git, add a local ignore rule or keep them outside
  the repo and reference absolute URLs in posts.

