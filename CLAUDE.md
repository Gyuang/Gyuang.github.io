# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Jekyll-based personal blog built on the Minimal Mistakes theme (v4.19.2), focused on medical AI topics. The site is hosted on GitHub Pages and serves Korean content about machine learning research papers and technical tutorials.

## Commands

### Development
- `bundle install` - Install Ruby dependencies
- `bundle exec jekyll serve` - Start local development server (typically runs on http://localhost:4000)
- `bundle exec jekyll serve --drafts` - Include draft posts in development
- `bundle exec jekyll build` - Build static site for production

### Asset Management  
- `npm install` - Install Node.js dependencies for asset processing
- `npm run build:js` - Build and minify JavaScript assets
- `npm run watch:js` - Watch for JavaScript changes during development

## Architecture

### Content Structure
- `_posts/` - Blog posts organized by category (Blog/, Paper/, Study/)
- `_pages/` - Static pages including category archives and about page
- `_data/` - Site configuration data (navigation, UI text)
- `_includes/` - Reusable template components
- `_layouts/` - Page layout templates
- `_sass/` - Sass stylesheets with custom skin (`myskin`)

### Key Configuration
- `_config.yml` - Main Jekyll configuration with Korean locale, Disqus comments, Google Analytics
- `Gemfile` - Ruby gem dependencies for Jekyll and plugins
- `package.json` - Node.js dependencies for asset processing

### Content Categories
The site covers three main content types:
- **Blog**: Technical tutorials (git, Docker, Kubernetes, etc.)
- **Paper**: Medical AI research paper reviews and summaries
- **Study**: Technical deep-dives (Dempster-Shafer theory, etc.)

### Theme Customization
- Uses custom skin (`myskin`) defined in `_sass/minimal-mistakes/skins/`
- Custom includes for analytics, comments (Disqus), and author profile
- Image assets stored in `assets/images/` with paper-specific subdirectories

### Deployment
Site is configured for GitHub Pages deployment with proper SEO, analytics tracking, and social sharing integration.