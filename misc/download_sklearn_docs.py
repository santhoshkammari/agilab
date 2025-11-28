#!/usr/bin/env python3
"""
Download scikit-learn User Guide and convert to organized Markdown files.
"""

import os
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin
import time

BASE_URL = "https://scikit-learn.org/stable/"
OUTPUT_DIR = "scikit_learn_user_guide"

# Complete hierarchy extracted from the website
HIERARCHY = {
    "total_sections": 14,
    "sections": [
        {"number": "1", "title": "Supervised learning", "href": "supervised_learning.html", "folder": "01_supervised_learning"},
        {"number": "2", "title": "Unsupervised learning", "href": "unsupervised_learning.html", "folder": "02_unsupervised_learning"},
        {"number": "3", "title": "Model selection and evaluation", "href": "model_selection.html", "folder": "03_model_selection"},
        {"number": "4", "title": "Metadata Routing", "href": "metadata_routing.html", "folder": "04_metadata_routing"},
        {"number": "5", "title": "Inspection", "href": "inspection.html", "folder": "05_inspection"},
        {"number": "6", "title": "Visualizations", "href": "visualizations.html", "folder": "06_visualizations"},
        {"number": "7", "title": "Dataset transformations", "href": "data_transforms.html", "folder": "07_data_transforms"},
        {"number": "8", "title": "Dataset loading utilities", "href": "datasets.html", "folder": "08_datasets"},
        {"number": "9", "title": "Computing with scikit-learn", "href": "computing.html", "folder": "09_computing"},
        {"number": "10", "title": "Model persistence", "href": "model_persistence.html", "folder": "10_model_persistence"},
        {"number": "11", "title": "Common pitfalls", "href": "common_pitfalls.html", "folder": "11_common_pitfalls"},
        {"number": "12", "title": "Dispatching", "href": "dispatching.html", "folder": "12_dispatching"},
        {"number": "13", "title": "Choosing the right estimator", "href": "machine_learning_map.html", "folder": "13_estimator_selection"},
        {"number": "14", "title": "External Resources", "href": "presentations.html", "folder": "14_resources"}
    ]
}

# Module pages that belong to each section
MODULE_PAGES = {
    "01_supervised_learning": [
        "modules/linear_model.html",
        "modules/lda_qda.html",
        "modules/kernel_ridge.html",
        "modules/svm.html",
        "modules/sgd.html",
        "modules/neighbors.html",
        "modules/gaussian_process.html",
        "modules/cross_decomposition.html",
        "modules/naive_bayes.html",
        "modules/tree.html",
        "modules/ensemble.html",
        "modules/multiclass.html",
        "modules/feature_selection.html",
        "modules/semi_supervised.html",
        "modules/isotonic.html",
        "modules/calibration.html",
        "modules/neural_networks_supervised.html"
    ],
    "02_unsupervised_learning": [
        "modules/mixture.html",
        "modules/manifold.html",
        "modules/clustering.html",
        "modules/biclustering.html",
        "modules/decomposition.html",
        "modules/covariance.html",
        "modules/outlier_detection.html",
        "modules/density.html",
        "modules/neural_networks_unsupervised.html"
    ],
    "03_model_selection": [
        "modules/cross_validation.html",
        "modules/grid_search.html",
        "modules/classification_threshold.html",
        "modules/model_evaluation.html",
        "modules/learning_curve.html"
    ],
    "05_inspection": [
        "modules/partial_dependence.html",
        "modules/permutation_importance.html"
    ],
    "07_data_transforms": [
        "modules/compose.html",
        "modules/feature_extraction.html",
        "modules/preprocessing.html",
        "modules/impute.html",
        "modules/unsupervised_reduction.html",
        "modules/random_projection.html",
        "modules/kernel_approximation.html",
        "modules/metrics.html",
        "modules/preprocessing_targets.html"
    ],
    "08_datasets": [
        "datasets/toy_dataset.html",
        "datasets/real_world.html",
        "datasets/sample_generators.html",
        "datasets/loading_other_datasets.html"
    ],
    "09_computing": [
        "computing/scaling_strategies.html",
        "computing/computational_performance.html",
        "computing/parallelism.html"
    ],
    "12_dispatching": [
        "modules/array_api.html"
    ]
}


def setup_html2text():
    """Configure html2text for better markdown conversion."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    h.protect_links = True
    h.wrap_links = False
    return h


def extract_main_content(soup):
    """Extract the main article content from the page."""
    # Try to find the main article content
    main_content = soup.find('article')
    if not main_content:
        main_content = soup.find('main')
    if not main_content:
        main_content = soup.find('div', {'class': 'body'})

    return main_content if main_content else soup


def download_page(url, retries=3):
    """Download a page with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ✗ Failed to download {url}: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None


def html_to_markdown(html_content, base_url):
    """Convert HTML content to Markdown."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract main content
    main_content = extract_main_content(soup)

    # Remove navigation, footer, and other non-content elements
    for element in main_content.find_all(['nav', 'footer', 'script', 'style']):
        element.decompose()

    # Remove breadcrumbs
    for element in main_content.find_all(attrs={'class': ['breadcrumb', 'headerlink']}):
        element.decompose()

    # Convert to markdown
    h = setup_html2text()
    markdown = h.handle(str(main_content))

    return markdown


def sanitize_filename(filename):
    """Create a safe filename from a title."""
    # Remove or replace invalid characters
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = filename.replace(':', '_').replace('?', '')
    filename = filename.replace('*', '').replace('"', '').replace('<', '').replace('>', '')
    filename = filename.replace('|', '_').strip()
    return filename


def download_and_convert(url, output_path):
    """Download a page and convert it to Markdown."""
    html_content = download_page(url)
    if not html_content:
        return False

    markdown = html_to_markdown(html_content, BASE_URL)

    # Save markdown file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    return True


def main():
    """Main function to download and organize the documentation."""
    print("📚 Downloading scikit-learn User Guide Documentation")
    print("=" * 60)

    base_path = Path(OUTPUT_DIR)
    base_path.mkdir(exist_ok=True)

    # Download main user guide page
    print("\n📄 Downloading main user guide page...")
    main_url = urljoin(BASE_URL, "user_guide.html")
    if download_and_convert(main_url, base_path / "README.md"):
        print("  ✓ user_guide.html -> README.md")

    # Download each section
    for section in HIERARCHY["sections"]:
        section_num = section["number"]
        section_title = section["title"]
        section_href = section["href"]
        section_folder = section["folder"]

        print(f"\n📖 Section {section_num}: {section_title}")
        print("-" * 60)

        # Create section folder
        section_path = base_path / section_folder
        section_path.mkdir(exist_ok=True)

        # Download main section page
        section_url = urljoin(BASE_URL, section_href)
        section_file = section_path / "README.md"

        if download_and_convert(section_url, section_file):
            print(f"  ✓ {section_href} -> {section_folder}/README.md")

        # Download module pages for this section
        if section_folder in MODULE_PAGES:
            module_pages = MODULE_PAGES[section_folder]

            for module_href in module_pages:
                module_url = urljoin(BASE_URL, module_href)

                # Extract module name for filename
                module_name = Path(module_href).stem
                module_file = section_path / f"{module_name}.md"

                if download_and_convert(module_url, module_file):
                    print(f"  ✓ {module_href} -> {section_folder}/{module_name}.md")

                # Small delay to be respectful to the server
                time.sleep(0.5)

    # Create main index
    create_main_index(base_path)

    print("\n" + "=" * 60)
    print(f"✅ Download complete! Documentation saved to: {OUTPUT_DIR}/")
    print("=" * 60)


def create_main_index(base_path):
    """Create a main index/navigation file."""
    index_content = """# scikit-learn User Guide

This directory contains the scikit-learn user guide documentation converted to Markdown format.

## Table of Contents

"""

    for section in HIERARCHY["sections"]:
        section_num = section["number"]
        section_title = section["title"]
        section_folder = section["folder"]

        index_content += f"{section_num}. [{section_title}]({section_folder}/README.md)\n"

    index_content += """

## About

This documentation was downloaded from the official scikit-learn website:
https://scikit-learn.org/stable/user_guide.html

Version: 1.7.2 (stable)

## Organization

Each major section has its own folder containing:
- `README.md` - The main section overview
- Individual module documentation files

"""

    with open(base_path / "INDEX.md", 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"\n  ✓ Created INDEX.md")


if __name__ == "__main__":
    main()
