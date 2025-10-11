# Gemini Context: Gyuang.github.io

This document provides a comprehensive overview of the `Gyuang.github.io` project, which is a personal blog focused on AI in medicine. The context is intended for the Gemini AI assistant to ensure efficient and accurate assistance.

## Project Overview

This is a static website project built using the **Jekyll** framework and hosted on GitHub Pages. It serves as a personal blog for "Gyuang" with a focus on "Medical AI".

- **Framework:** Jekyll
- **Theme:** [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) (used as a `remote_theme`) with a custom skin named "myskin".
- **Primary Content:** Blog posts written in Markdown, located in the `_posts` directory.
- **Core Technologies:**
    - **Ruby/Jekyll:** For static site generation. Dependencies are managed via `Gemfile`.
    - **HTML/SCSS/JavaScript:** For structure, styling, and client-side behavior.
    - **Node.js/npm:** Used for frontend asset management, specifically for uglifying JavaScript files as defined in `package.json`.

The main configuration is centralized in `_config.yml`, which controls the site title, author details, navigation, social links, comment system (Disqus), and analytics (Google).

## Building and Running

The project uses both Ruby's Bundler and Node.js's npm for dependency management and build tasks.

### Prerequisites

1.  **Ruby and Bundler:** Ensure Ruby and the Bundler gem are installed.
2.  **Node.js and npm:** Ensure Node.js and npm are installed.

### Key Commands

- **Install Dependencies:**
  - To install Ruby gems: `bundle install`
  - To install Node.js packages: `npm install`

- **Run the Local Development Server:**
  - The primary command to serve the Jekyll site locally is:
    ```bash
    bundle exec jekyll serve
    ```

- **Build Frontend Assets:**
  - To minify the JavaScript files as defined in `package.json`:
    ```bash
    npm run build:js
    ```

## Development Conventions

- **Content Creation:**
  - New blog posts should be created as Markdown files (`.md`) within the `_posts` directory.
  - The filename format for posts is `YYYY-MM-DD-post-title.md`.
  - Post metadata (layout, title, categories, tags, etc.) is managed via YAML front matter at the top of each Markdown file.

- **Directory Structure:**
  - `_posts/`: Contains all blog articles.
  - `_pages/`: Contains custom pages like "About" or archive pages.
  - `_data/`: Holds structured data files (e.g., `navigation.yml`, `ui-text.yml`) used to populate the site.
  - `_includes/`: Contains reusable HTML snippets for layouts.
  - `_layouts/`: Defines the main HTML structure for different types of content.
  - `assets/`: Contains images, CSS/Sass, and JavaScript files.

- **Configuration:**
  - All global site settings are in `_config.yml`. Changes to this file require restarting the Jekyll server.
  - Navigation links are managed in `_data/navigation.yml`.
  - Author information, which appears in the sidebar, is configured under the `author` section in `_config.yml`.
