# VLM Paper Post Automation Tools

This directory contains tools to automate the creation of VLM (Vision-Language Model) paper posts for your Jekyll blog.

## Tools Overview

### 1. Interactive Generator (`generate_vlm_post.py`)
Creates a single VLM paper post through interactive prompts.

**Usage:**
```bash
cd tools
python3 generate_vlm_post.py
```

**Features:**
- Interactive prompts for all paper details
- Automatic filename generation with date
- Creates proper Jekyll front matter
- Sets up VLM category and tags
- Creates image directory structure

### 2. Batch Generator (`batch_vlm_generator.py`) 
Creates multiple VLM paper posts from a JSON configuration file.

**Usage:**
```bash
cd tools
python3 batch_vlm_generator.py
```

**Features:**
- Processes multiple papers from JSON config
- Creates sample configuration on first run
- Batch processing for efficiency
- Tracks required image files

### 3. Template (`vlm_paper_template.md`)
The base template used by both generators with placeholders for:
- Paper metadata (title, tags, dates)
- Content sections (introduction, methods, experiments)
- Image handling
- Consistent formatting

## Quick Start

### For Single Papers
1. Run the interactive generator:
   ```bash
   python3 tools/generate_vlm_post.py
   ```
2. Follow the prompts to enter paper details
3. Add images to `/assets/images/paper/vlm/` if needed
4. Preview with `bundle exec jekyll serve`

### For Multiple Papers  
1. Run the batch generator to create sample config:
   ```bash
   python3 tools/batch_vlm_generator.py
   ```
2. Edit `tools/vlm_papers_config.json` with your paper details
3. Run the batch generator again to create all posts
4. Add required images to `/assets/images/paper/vlm/`

## Configuration Format

The batch generator uses JSON configuration:

```json
{
  "papers": [
    {
      "title": "Paper Title",
      "excerpt": "Brief description",
      "additional_tags": "Tag1, Tag2",
      "introduction": "Introduction text...",
      "related_work_vlm": "VLM related work...",
      "specific_domain": "Domain name",
      "related_work_domain": "Domain related work...",
      "architecture_description": "Architecture details...",
      "architecture_image": "image_filename.png",
      "key_components": "Key components...",
      "training_strategy": "Training details...",
      "datasets": "Datasets used...",
      "results": "Main results...",
      "ablation_studies": "Ablation studies...",
      "conclusion": "Conclusion...",
      "key_takeaways": "Key points..."
    }
  ]
}
```

## File Structure

After running the tools, your blog will have:

```
├── _posts/Paper/
│   └── YYYY-MM-DD-Paper-Title.md
├── _pages/categories/
│   └── category-vlm.md  
├── assets/images/paper/vlm/
│   └── [your paper images]
└── tools/
    ├── generate_vlm_post.py
    ├── batch_vlm_generator.py
    ├── vlm_paper_template.md
    ├── vlm_papers_config.json
    └── README.md
```

## Customization

### Template Customization
Edit `vlm_paper_template.md` to modify:
- Front matter structure
- Section headings
- Default content layout
- Image handling

### Adding New Fields
1. Add placeholder to template: `{{NEW_FIELD}}`
2. Update generator scripts to collect the field
3. Add replacement in `create_post_content()` function

## Tips

- Use descriptive filenames for images
- Keep excerpts concise but informative  
- Add relevant tags for better categorization
- Preview posts locally before publishing
- Images should be placed in `/assets/images/paper/vlm/`

## Troubleshooting

**Template not found:** Ensure you're running scripts from the blog root directory

**Permission denied:** Make scripts executable with `chmod +x script_name.py`

**JSON errors:** Validate your configuration file with a JSON validator

**Missing images:** Check that image files exist in the correct directory