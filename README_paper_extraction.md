# Paper Image Extraction Tool

## Overview

This tool automates the process of extracting images from ArXiv research papers and integrating them into Jekyll blog posts. It downloads PDFs, extracts figures and tables, organizes them in the proper directory structure, and automatically inserts them into your existing blog posts.

## Features

- **Automated PDF Download**: Downloads papers directly from ArXiv using paper IDs
- **Smart Image Extraction**: Uses PyMuPDF to extract figures, tables, and diagrams from PDFs
- **Intelligent Naming**: Generates meaningful filenames based on content analysis
- **Web Optimization**: Resizes and optimizes images for web performance
- **Auto-Integration**: Automatically inserts images into Jekyll posts at appropriate locations
- **Error Handling**: Graceful error handling with detailed logging

## Setup

### Install Dependencies
```bash
pip install pymupdf pillow requests beautifulsoup4 python-frontmatter
```

### Required Directory Structure
```
your-jekyll-site/
├── _posts/Paper/
├── assets/images/paper/
└── paper_image_extractor.py
```

## Usage

### For the Pre-configured Papers
Simply run:
```bash
python paper_image_extractor.py
```

This will process the 4 prompt tuning papers that are already configured.

### For New Papers

1. Edit the `papers` dictionary in `paper_image_extractor.py`:
```python
self.papers = {
    "Your Paper Name": {
        "arxiv_id": "YYMM.NNNNN",
        "post_file": "YYYY-MM-DD-Your-Post.md", 
        "folder_name": "your-paper-folder"
    }
}
```

2. Ensure your Jekyll post exists in `_posts/Paper/`

3. Run the script:
```bash
python paper_image_extractor.py
```

## Configuration Options

### Image Extraction Settings
- `min_image_size`: Minimum image dimensions (default: 100x100)
- `max_image_width`: Maximum width for web optimization (default: 1200px)
- `image_quality`: JPEG quality for optimization (default: 85)

### Integration Settings
- `auto_insert`: Automatically insert images into posts (default: True)
- `create_captions`: Generate captions for images (default: True)
- `add_additional_section`: Add "Additional Figures" section (default: True)

## Output Structure

The tool creates the following structure:
```
assets/images/paper/
├── paper-name-1/
│   ├── architecture_overview_0.png
│   ├── results_table_3_0.png
│   └── figure_5_1.png
└── paper-name-2/
    ├── method_diagram_2.png
    └── results_table_4_0.png
```

## Jekyll Integration

Images are automatically inserted into posts using:
```markdown
![Image Caption](/assets/images/paper/folder-name/image-name.png)
*Figure: Image Caption*
```

### Insertion Logic
1. **Architecture diagrams** → After method/architecture sections
2. **Results tables** → After experiment/results sections  
3. **Overview figures** → After introduction/overview sections
4. **Remaining images** → In "Additional Figures" section

## Error Handling

The script includes comprehensive error handling:
- Failed downloads are logged and skipped
- Corrupted images are filtered out
- Missing posts are reported
- Partial failures don't stop the entire process

## Logging

All operations are logged with timestamps:
```
2025-09-15 17:27:48 - INFO - Processing paper: Paper Name
2025-09-15 17:27:49 - INFO - Extracted 6 images from Paper Name
2025-09-15 17:27:49 - INFO - Updated post: paper-post.md
```

## Customization

### Custom Image Naming
Modify the `generate_image_name()` method to change naming conventions.

### Custom Insertion Points
Modify the `find_insertion_points()` method to change where images are inserted.

### Custom Optimization
Modify the `optimize_images()` method to change image optimization settings.

## Troubleshooting

### Common Issues

1. **"No images extracted"**
   - PDF might be image-based (scanned) rather than native PDF
   - Images might be embedded in vector graphics
   - Try manual image extraction

2. **"Post file not found"**
   - Check the post filename in the configuration
   - Ensure the post exists in `_posts/Paper/`

3. **"Failed to download PDF"**
   - Check internet connection
   - Verify ArXiv ID is correct
   - Try manual download first

### Debug Mode
Add verbose logging by changing the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for the script:
- OCR support for scanned PDFs
- Integration with other paper sources (ACL Anthology, etc.)
- Automatic paper metadata extraction
- Support for video and other media types
- Integration with citation management tools

## Files Created

After running the script, you'll find:
- `paper_extraction_report.md` - Detailed processing report
- `paper_config_template.json` - Template for future configurations
- Extracted images in `assets/images/paper/`
- Updated Jekyll posts with integrated images