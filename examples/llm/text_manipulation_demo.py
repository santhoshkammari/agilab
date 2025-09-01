#!/usr/bin/env python3
"""
Text Content Manipulation Demo

This script demonstrates the powerful text manipulation capabilities of the flowgen
text_content.py tools. It shows how AI can perform precise, line-level operations
on text content with full visibility and control.

Run this script to see examples of:
1. Line-based extraction and copying
2. Content replacement and summarization  
3. Regex pattern matching and replacement
4. Structural analysis and manipulation planning
5. Safe content transformation with verification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgen.tools.text_content import *
import json


def demo_line_operations():
    """Demonstrate precise line-based content operations."""
    print("=" * 60)
    print("DEMO 1: LINE-BASED OPERATIONS")
    print("=" * 60)
    
    sample_code = """def calculate_factorial(n):
    \"\"\"Calculate factorial of a number.\"\"\"
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Test the function
print(calculate_factorial(5))
print(calculate_factorial(0))"""

    print("Original content:")
    for i, line in enumerate(sample_code.splitlines(), 1):
        print(f"{i:2}: {line}")
    
    print("\n--- Extracting function signature (line 1) ---")
    result = text_get_lines(sample_code, 1)
    print(f"Extracted: {result['extracted_content']}")
    
    print("\n--- Extracting function body (lines 3-10) ---") 
    result = text_get_lines(sample_code, 3, 10)
    print(f"Lines {result['line_range']}: {result['line_count']} lines, {result['word_count']} words")
    print(f"Content preview: {result['extracted_content'][:50]}...")
    
    print("\n--- Copying lines for reuse ---")
    copy_result = text_copy_lines(sample_code, 1, 2)
    print("Ready for paste:")
    print(copy_result['ready_for_paste'])


def demo_content_replacement():
    """Demonstrate content replacement and summarization."""
    print("\n" + "=" * 60)
    print("DEMO 2: CONTENT REPLACEMENT & SUMMARIZATION")
    print("=" * 60)
    
    verbose_doc = """# API Documentation

## Introduction

This comprehensive documentation provides detailed information about our RESTful API
endpoints, authentication mechanisms, rate limiting policies, error handling procedures,
and best practices for integration. The API follows REST principles and uses JSON
for data exchange. All endpoints support HTTPS and require proper authentication.

## Authentication

To use our API, you must first obtain an API key from your dashboard. The API key
should be included in every request using the Authorization header. We support
both basic authentication and JWT tokens for enhanced security.

## Rate Limiting

Our API implements rate limiting to ensure fair usage across all clients. The
current limits are 1000 requests per hour for authenticated users and 100 requests
per hour for unauthenticated requests. Rate limit headers are included in responses.
"""

    print("Original verbose content:")
    lines = verbose_doc.splitlines()
    for i, line in enumerate(lines, 1):
        print(f"{i:2}: {line}")
    
    print("\n--- Summarizing Introduction section (lines 3-7) ---")
    summary_result = text_summarize_section(verbose_doc, 3, 7, "brief")
    print("Original section:")
    print(summary_result['original_section'])
    print("\nBrief summary:")
    print(summary_result['summary_generated'])
    print(f"Compression: {summary_result['compression_analysis']['original_words']} → {summary_result['compression_analysis']['summary_words']} words")
    
    print("\n--- Replacing verbose section with summary ---")
    replace_result = text_replace_lines(verbose_doc, 3, 7, summary_result['summary_generated'])
    print("Modified document:")
    print(replace_result['modified_content'])


def demo_regex_operations():
    """Demonstrate regex pattern matching and replacement."""
    print("\n" + "=" * 60) 
    print("DEMO 3: REGEX OPERATIONS")
    print("=" * 60)
    
    code_sample = """function getUserName() {
    return user.name;
}

function getUserEmail() {
    return user.email;
}

function getUserAge() {
    return user.age;
}"""

    print("Original code:")
    for i, line in enumerate(code_sample.splitlines(), 1):
        print(f"{i:2}: {line}")
    
    print("\n--- Finding function patterns ---")
    pattern_result = text_find_patterns(code_sample, r'function\s+(\w+)\(\)', "")
    print(f"Found {len(pattern_result['line_matches'])} function definitions:")
    for match in pattern_result['line_matches']:
        print(f"  Line {match['line_number']}: {match['highlighted_line']}")
    
    print("\n--- Validating replacement pattern ---")
    validation = text_validate_regex(r'function\s+(\w+)\(\)', code_sample)
    print(f"Pattern valid: {validation['valid']}")
    print(f"Total matches: {validation['test_results']['total_matches']}")
    
    print("\n--- Converting function syntax ---")
    replacement_result = text_replace_pattern(
        code_sample, 
        r'function\s+(\w+)\(\)', 
        r'const \1 = () =>', 
        ""
    )
    print("Modified code:")
    print(replacement_result['modified_content'])
    print(f"Applied {replacement_result['replacement_summary']['replacements_made']} replacements")


def demo_content_analysis():
    """Demonstrate structural analysis and content insights."""
    print("\n" + "=" * 60)
    print("DEMO 4: CONTENT ANALYSIS & STRUCTURE")
    print("=" * 60)
    
    mixed_content = """# Project Overview

This project implements a text processing system.

    def main():
        print("Starting application")
        
    class TextProcessor:
        def __init__(self):
            self.data = []

## Features

- Real-time processing
- Multiple input formats
- Extensible architecture

## Configuration

The system can be configured through environment variables:

    PROCESSOR_MODE=advanced
    MAX_WORKERS=4

## Conclusion

This system provides comprehensive text processing capabilities."""

    print("Analyzing content structure...")
    structure = text_analyze_structure(mixed_content)
    
    print(f"\nStructural elements found:")
    print(f"- Headers: {len(structure['structural_elements']['headers'])}")
    print(f"- Code blocks: {len(structure['structural_elements']['code_blocks'])}")
    print(f"- List items: {len(structure['structural_elements']['list_items'])}")
    print(f"- Empty lines: {len(structure['structural_elements']['empty_lines'])}")
    
    print(f"\nLogical sections identified: {len(structure['logical_sections'])}")
    for section in structure['logical_sections']:
        print(f"  Section {section['section_id']}: Lines {section['start_line']}-{section['end_line']} ({section['word_count']} words)")
    
    print(f"\nContent characteristics:")
    chars = structure['content_characteristics']
    print(f"- Hierarchical structure: {chars['hierarchical_structure']}")
    print(f"- Code heavy: {chars['code_heavy']}")
    print(f"- Well formatted: {chars['well_formatted']}")
    
    print(f"\nManipulation recommendations:")
    recs = structure['manipulation_recommendations']
    print(f"- Good summarization candidates: {len(recs['good_summarization_candidates'])}")
    print(f"- Natural break points: {recs['natural_break_points']}")
    print(f"- Regex complexity: {recs['regex_operation_complexity']}")


def demo_content_comparison():
    """Demonstrate content comparison and difference analysis."""
    print("\n" + "=" * 60)
    print("DEMO 5: CONTENT COMPARISON")
    print("=" * 60)
    
    version1 = """def process_data(data):
    # Validate input
    if not data:
        return None
    
    # Process items
    results = []
    for item in data:
        results.append(item.upper())
    
    return results"""

    version2 = """def process_data(data):
    \"\"\"Process data with validation and transformation.\"\"\"
    # Validate input
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Process items with error handling
    results = []
    for item in data:
        try:
            results.append(item.upper())
        except AttributeError:
            results.append(str(item).upper())
    
    return results"""

    print("Comparing two versions of a function...")
    comparison = text_compare_sections(version1, version2)
    
    print(f"Similarity: {comparison['similarity_analysis']['word_similarity']:.2%}")
    print(f"Total differences: {comparison['content_comparison']['total_differences']}")
    
    print("\nLine differences:")
    for diff in comparison['content_comparison']['line_differences']:
        print(f"  Line {diff['line_number']}: {diff['difference_type']}")
        if diff['content1']:
            print(f"    Before: {diff['content1']}")
        if diff['content2']:
            print(f"    After:  {diff['content2']}")


def demo_search_and_extract():
    """Demonstrate content search and section extraction."""
    print("\n" + "=" * 60)
    print("DEMO 6: SEARCH & EXTRACTION")
    print("=" * 60)
    
    documentation = """# Configuration Guide

## Database Setup

The database configuration requires the following steps:

1. Install PostgreSQL
2. Create database: flowgen_db
3. Configure connection string
4. Run migrations

## API Configuration  

Configure the API endpoints in config.yaml:

```yaml
api:
  host: localhost
  port: 8000
  debug: false
```

## Security Setup

Ensure proper security configuration before deployment."""

    print("Searching for 'Configuration' with context...")
    search_result = text_search_content(documentation, "Configuration", context_lines=2)
    
    print(f"Found {search_result['total_matches']} matches:")
    for match in search_result['matches']:
        print(f"\nLine {match['line_number']}: {match['highlighted_line']}")
        print("Context:")
        print(match['context']['context_block'])
        print("-" * 40)
    
    print("\n--- Extracting section between markers ---")
    section_result = text_extract_section(
        documentation, 
        "## API Configuration", 
        "## Security Setup",
        include_markers=False
    )
    
    print("Extracted section:")
    print(section_result['extracted_content'])
    print(f"Section info: Lines {section_result['section_info']['extracted_range']}")


def run_comprehensive_demo():
    """Run all demonstrations in sequence."""
    print("TEXT CONTENT MANIPULATION TOOLS DEMONSTRATION")
    print("=" * 80)
    print("This demo shows how AI can perform precise text operations with full visibility")
    print("=" * 80)
    
    demo_line_operations()
    demo_content_replacement() 
    demo_regex_operations()
    demo_content_analysis()
    demo_content_comparison()
    demo_search_and_extract()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("These tools unlock AI's ability to:")
    print("✓ Extract exact line ranges with precision")
    print("✓ Replace content surgically without side effects")
    print("✓ Apply regex transformations with verification")
    print("✓ Analyze content structure for optimal manipulation")
    print("✓ Compare content versions and track changes")
    print("✓ Search and extract with contextual awareness")
    print("\nIntegrate these tools with your LLM for enhanced text processing!")


if __name__ == '__main__':
    run_comprehensive_demo()