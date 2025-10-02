# Repository Guidelines

## Project Structure & Module Organization
Primary site content lives in `_posts` (dated research write-ups) and `_pages` (static sections). Layout logic is split between `_layouts` templates and `_includes` partials, while `_sass` and `assets` contain theme styling and bundled media. Generated artifacts in `_site` are disposable; edit source files only. Data-driven features pull from `_data`, and helper scripts for paper processing sit in the repository root alongside `papers_collection*` data dumps.

## Build, Test, and Development Commands
- `bundle exec jekyll serve --livereload`: builds the site locally with incremental rebuilds for authoring.
- `bundle exec jekyll build`: produces the production `_site` output and surfaces Liquid/YAML errors before publishing.
- `npm install && npm run build:js`: minifies theme JavaScript and applies the banner via `banner.js`; run after editing scripts under `assets/js`.

## Coding Style & Naming Conventions
Use two-space indentation for Markdown front matter, YAML files, and Liquid templates. Posts must follow the `YYYY-MM-DD-title.md` pattern with descriptive, lower-kebab-case names. Favor Markdown content blocks over raw HTML, and wrap custom components inside includes for reusability. For Python utilities, keep PEP 8 spacing and document new entry points with concise docstrings.

## Testing Guidelines
Before committing, run `bundle exec jekyll build` to catch template or data issues. Spot-check any regenerated `_site` output in a browser when layout changes touch `_layouts` or `_includes`. When adjusting `assets/js`, execute `npm run build:js` and confirm the resulting `assets/js/main.min.js` loads without console errors. Large content imports should be validated with existing paper check scripts to avoid malformed metadata.

## Commit & Pull Request Guidelines
Follow the existing Conventional Commit style (`feat:`, `fix:`, `enhance:`) observed in `git log`. Keep messages imperative and scoped to a single change. Pull requests should summarize visible outcomes, link related issues, and include screenshots or GIFs for layout updates. Confirm that local builds succeed and note any configuration changes in the description to aid reviewers.

## Configuration & Security Tips
Avoid editing `minimal-mistakes-jekyll.gemspec` or `_site` unless upgrading the theme. Do not commit credentials; secrets belong in local `_config.yml` overrides ignored by Git. Document new data sources in `_data/README.md` (create it if absent) so downstream scripts remain reproducible.
