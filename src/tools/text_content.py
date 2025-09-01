"""
Text Content Manipulation Tools for AI-Enhanced Content Processing

This module provides precise text manipulation capabilities that unlock AI's ability to work with content
at the line level, perform accurate replacements, and handle regex-based operations with full visibility.

Key Capabilities:
- Line-based content extraction and manipulation
- Precise content replacement with line number targeting
- Regex-based pattern matching and replacement with AI verification
- Content copying and transfer between sources
- Intelligent summarization and content condensation
- Text analysis with line-by-line precision

Design Philosophy:
While AI can see and understand content, it sometimes struggles with precise line-based operations.
These tools bridge that gap by providing exact line targeting, content verification, and structured
manipulation that maintains AI oversight and control.

Use Cases:
1. Extract specific line ranges for focused analysis
2. Replace verbose sections with concise summaries
3. Copy content from one section to another based on line numbers
4. Apply regex transformations with AI verification
5. Perform bulk text operations while maintaining content integrity
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union


SYSTEM_PROMPT = """You are a text content manipulation assistant. You have access to powerful tools for precise text processing:

Line-Based Tools:
- text_get_lines: Extract specific lines or line ranges from content
- text_get_content_stats: Get comprehensive statistics about text content
- text_copy_lines: Copy specific lines from content for reuse
- text_extract_section: Extract content between line markers

Content Replacement Tools:  
- text_replace_lines: Replace specific line ranges with new content
- text_replace_pattern: Replace text using regex patterns with verification
- text_summarize_section: Replace verbose content with intelligent summaries
- text_insert_at_line: Insert new content at specific line positions

Regex and Pattern Tools:
- text_find_patterns: Find and highlight regex patterns with line numbers
- text_validate_regex: Test regex patterns against content safely
- text_apply_regex: Apply regex transformations with before/after verification

Analysis Tools:
- text_analyze_structure: Analyze text structure and identify patterns
- text_compare_sections: Compare two text sections for differences
- text_search_content: Search for specific text with context

Each tool operates on text content as a string and provides precise line-level control.
Use these tools to perform exact text manipulations that require surgical precision."""


def text_get_lines(content: str, start_line: int, end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract specific lines or line ranges from text content.
    
    Purpose: Provides precise line-based content extraction for AI to work with exact portions
    of text without losing context or making approximations.
    
    Use Cases:
    - Extract function definitions by line numbers
    - Get specific paragraphs for analysis
    - Copy exact content for reuse elsewhere
    - Focus AI attention on specific text regions
    
    Args:
        content (str): The text content to extract from
        start_line (int): Starting line number (1-indexed)
        end_line (Optional[int]): Ending line number (1-indexed). If None, extracts single line
        
    Returns:
        Dict containing extracted lines, metadata, and context information
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    # Validate line numbers
    if start_line < 1 or start_line > total_lines:
        return {"error": f"Start line {start_line} out of range (1-{total_lines})"}
    
    if end_line is None:
        end_line = start_line
    
    if end_line < start_line or end_line > total_lines:
        return {"error": f"End line {end_line} invalid (must be >= {start_line} and <= {total_lines})"}
    
    # Extract lines (convert to 0-indexed)
    extracted_lines = lines[start_line-1:end_line]
    extracted_content = '\n'.join(extracted_lines)
    
    # Provide context (few lines before and after)
    context_before = max(1, start_line - 3)
    context_after = min(total_lines, end_line + 3)
    context_lines = lines[context_before-1:context_after]
    
    return {
        "extracted_content": extracted_content,
        "extracted_lines": extracted_lines,
        "line_range": f"{start_line}-{end_line}" if end_line != start_line else str(start_line),
        "line_count": len(extracted_lines),
        "word_count": len(extracted_content.split()),
        "char_count": len(extracted_content),
        "context": {
            "before_lines": context_before,
            "after_lines": context_after,
            "full_context": '\n'.join(context_lines)
        },
        "metadata": {
            "total_document_lines": total_lines,
            "percentage_extracted": round((len(extracted_lines) / total_lines) * 100, 2)
        }
    }


def text_get_content_stats(content: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics about text content structure and composition.
    
    Purpose: Provides AI with detailed understanding of content before manipulation.
    Essential for planning operations and understanding content scope.
    
    Use Cases:
    - Understand document structure before editing
    - Plan content manipulation strategies
    - Identify optimal points for summarization
    - Analyze content density and complexity
    
    Args:
        content (str): The text content to analyze
        
    Returns:
        Dict containing comprehensive content statistics
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    total_lines = len(lines)
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Basic stats
    total_chars = len(content)
    total_words = len(content.split())
    
    # Line analysis
    line_lengths = [len(line) for line in lines]
    avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    max_line_length = max(line_lengths) if line_lengths else 0
    
    # Content density analysis
    empty_lines = total_lines - len(non_empty_lines)
    content_density = len(non_empty_lines) / total_lines if total_lines > 0 else 0
    
    # Pattern detection
    code_pattern_lines = [i+1 for i, line in enumerate(lines) if re.match(r'^\s*(def |class |import |from |#include|public |private )', line.strip())]
    markdown_header_lines = [i+1 for i, line in enumerate(lines) if re.match(r'^#{1,6}\s+', line.strip())]
    bullet_list_lines = [i+1 for i, line in enumerate(lines) if re.match(r'^\s*[-*+]\s+', line.strip())]
    numbered_list_lines = [i+1 for i, line in enumerate(lines) if re.match(r'^\s*\d+\.\s+', line.strip())]
    
    return {
        "basic_stats": {
            "total_lines": total_lines,
            "non_empty_lines": len(non_empty_lines),
            "empty_lines": empty_lines,
            "total_characters": total_chars,
            "total_words": total_words,
            "content_density": round(content_density, 3)
        },
        "line_analysis": {
            "avg_line_length": round(avg_line_length, 1),
            "max_line_length": max_line_length,
            "lines_over_80_chars": len([l for l in line_lengths if l > 80]),
            "lines_over_120_chars": len([l for l in line_lengths if l > 120])
        },
        "content_patterns": {
            "code_like_lines": len(code_pattern_lines),
            "markdown_headers": len(markdown_header_lines),
            "bullet_lists": len(bullet_list_lines),
            "numbered_lists": len(numbered_list_lines)
        },
        "structure_hints": {
            "likely_code": len(code_pattern_lines) > total_lines * 0.1,
            "likely_markdown": len(markdown_header_lines) > 0,
            "has_lists": len(bullet_list_lines) + len(numbered_list_lines) > 0,
            "highly_structured": content_density > 0.8
        },
        "manipulation_suggestions": {
            "good_for_line_operations": total_lines < 500,
            "good_for_summarization": total_words > 200 and content_density > 0.6,
            "good_for_regex": total_lines < 200,
            "needs_chunking": total_lines > 1000
        }
    }


def text_copy_lines(content: str, start_line: int, end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    Copy specific lines from content for reuse in other contexts.
    
    Purpose: Enables precise content transfer between different sources or sections.
    Unlike simple extraction, this prepares content for insertion elsewhere.
    
    Use Cases:
    - Copy function signatures between files
    - Extract configuration sections for reuse
    - Move content blocks between documents
    - Create content templates from existing text
    
    Args:
        content (str): Source text content
        start_line (int): Starting line number (1-indexed)
        end_line (Optional[int]): Ending line number (1-indexed)
        
    Returns:
        Dict containing copied content ready for insertion
    """
    extraction_result = text_get_lines(content, start_line, end_line)
    
    if "error" in extraction_result:
        return extraction_result
    
    copied_content = extraction_result["extracted_content"]
    
    # Analyze indentation for smart pasting
    lines = extraction_result["extracted_lines"]
    if lines:
        # Find common indentation
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            indentations = [len(line) - len(line.lstrip()) for line in non_empty_lines]
            min_indent = min(indentations)
            max_indent = max(indentations)
        else:
            min_indent = max_indent = 0
    else:
        min_indent = max_indent = 0
    
    return {
        "copied_content": copied_content,
        "ready_for_paste": copied_content,
        "source_range": extraction_result["line_range"],
        "line_count": extraction_result["line_count"],
        "indentation_info": {
            "min_indent": min_indent,
            "max_indent": max_indent,
            "needs_deindent": min_indent > 0,
            "relative_indents": [len(line) - len(line.lstrip()) - min_indent for line in lines]
        },
        "paste_suggestions": {
            "preserve_indent": copied_content,
            "remove_indent": '\n'.join(line[min_indent:] if len(line) >= min_indent else line for line in lines),
            "add_indent": '\n'.join('    ' + line for line in lines)
        }
    }


def text_replace_lines(content: str, start_line: int, end_line: int, replacement: str) -> Dict[str, Any]:
    """
    Replace specific line ranges with new content.
    
    Purpose: Enables surgical content replacement while maintaining document structure.
    Essential for AI to make precise edits without affecting surrounding content.
    
    Use Cases:
    - Replace verbose explanations with summaries
    - Update function implementations
    - Swap content sections between documents
    - Apply AI-generated improvements to specific areas
    
    Args:
        content (str): Original text content
        start_line (int): Starting line to replace (1-indexed)
        end_line (int): Ending line to replace (1-indexed)
        replacement (str): New content to insert
        
    Returns:
        Dict containing modified content and operation details
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    # Validate line numbers
    if start_line < 1 or start_line > total_lines:
        return {"error": f"Start line {start_line} out of range (1-{total_lines})"}
    
    if end_line < start_line or end_line > total_lines:
        return {"error": f"End line {end_line} invalid (must be >= {start_line} and <= {total_lines})"}
    
    # Get original content for comparison
    original_section = '\n'.join(lines[start_line-1:end_line])
    
    # Perform replacement
    new_lines = (
        lines[:start_line-1] +  # Lines before replacement
        replacement.splitlines() +  # New content
        lines[end_line:]  # Lines after replacement
    )
    
    new_content = '\n'.join(new_lines)
    replacement_lines = replacement.splitlines()
    
    return {
        "modified_content": new_content,
        "operation_summary": {
            "lines_replaced": f"{start_line}-{end_line}",
            "original_line_count": end_line - start_line + 1,
            "new_line_count": len(replacement_lines),
            "net_line_change": len(replacement_lines) - (end_line - start_line + 1)
        },
        "content_comparison": {
            "original_section": original_section,
            "new_section": replacement,
            "original_word_count": len(original_section.split()),
            "new_word_count": len(replacement.split()),
            "compression_ratio": len(replacement.split()) / len(original_section.split()) if original_section.split() else 0
        },
        "document_stats": {
            "original_total_lines": total_lines,
            "new_total_lines": len(new_lines),
            "lines_changed": abs(len(replacement_lines) - (end_line - start_line + 1))
        }
    }


def text_insert_at_line(content: str, line_number: int, insertion: str, mode: str = "after") -> Dict[str, Any]:
    """
    Insert new content at a specific line position.
    
    Purpose: Enables precise content insertion without disrupting existing structure.
    Perfect for adding new sections, comments, or enhancements at exact locations.
    
    Use Cases:
    - Add explanatory comments above functions
    - Insert new sections in documentation
    - Add import statements at file top
    - Inject code snippets at specific locations
    
    Args:
        content (str): Original text content
        line_number (int): Line position for insertion (1-indexed)
        insertion (str): Content to insert
        mode (str): "before", "after", or "replace"
        
    Returns:
        Dict containing modified content and insertion details
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    if line_number < 1 or line_number > total_lines:
        return {"error": f"Line number {line_number} out of range (1-{total_lines})"}
    
    insertion_lines = insertion.splitlines()
    
    if mode == "before":
        new_lines = (
            lines[:line_number-1] +
            insertion_lines +
            lines[line_number-1:]
        )
        actual_insert_line = line_number
    elif mode == "after":
        new_lines = (
            lines[:line_number] +
            insertion_lines +
            lines[line_number:]
        )
        actual_insert_line = line_number + 1
    elif mode == "replace":
        new_lines = (
            lines[:line_number-1] +
            insertion_lines +
            lines[line_number:]
        )
        actual_insert_line = line_number
    else:
        return {"error": f"Invalid mode '{mode}'. Use 'before', 'after', or 'replace'"}
    
    new_content = '\n'.join(new_lines)
    
    return {
        "modified_content": new_content,
        "insertion_details": {
            "inserted_at_line": actual_insert_line,
            "insertion_mode": mode,
            "lines_inserted": len(insertion_lines),
            "inserted_content": insertion
        },
        "document_changes": {
            "original_lines": total_lines,
            "new_lines": len(new_lines),
            "net_change": len(new_lines) - total_lines
        },
        "context": {
            "line_before": lines[line_number-2] if line_number > 1 else "",
            "target_line": lines[line_number-1] if line_number <= total_lines else "",
            "line_after": lines[line_number] if line_number < total_lines else ""
        }
    }


def text_find_patterns(content: str, pattern: str, flags: str = "") -> Dict[str, Any]:
    """
    Find and highlight regex patterns in content with precise line information.
    
    Purpose: Enables AI to see exactly where patterns match in content, providing
    full visibility for regex operations before applying changes.
    
    Use Cases:
    - Find all function definitions in code
    - Locate specific formatting patterns
    - Identify areas that need standardization
    - Verify regex patterns before bulk operations
    
    Args:
        content (str): Text content to search
        pattern (str): Regex pattern to find
        flags (str): Regex flags (i=ignorecase, m=multiline, s=dotall, x=verbose)
        
    Returns:
        Dict containing all matches with line numbers and context
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    # Parse flags
    regex_flags = 0
    if 'i' in flags.lower():
        regex_flags |= re.IGNORECASE
    if 'm' in flags.lower():
        regex_flags |= re.MULTILINE
    if 's' in flags.lower():
        regex_flags |= re.DOTALL
    if 'x' in flags.lower():
        regex_flags |= re.VERBOSE
    
    try:
        compiled_pattern = re.compile(pattern, regex_flags)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {e}"}
    
    lines = content.splitlines()
    matches = []
    
    # Find matches line by line for precise location tracking
    for line_num, line in enumerate(lines, 1):
        for match in compiled_pattern.finditer(line):
            matches.append({
                "line_number": line_num,
                "line_content": line,
                "match_text": match.group(0),
                "match_start": match.start(),
                "match_end": match.end(),
                "groups": match.groups(),
                "highlighted_line": line[:match.start()] + f">>>{match.group(0)}<<<" + line[match.end():]
            })
    
    # Also find multiline matches if applicable
    multiline_matches = []
    if 'm' in flags.lower() or 's' in flags.lower():
        for match in compiled_pattern.finditer(content):
            # Find which lines this match spans
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers for start and end
            content_before_start = content[:start_pos]
            content_before_end = content[:end_pos]
            start_line = content_before_start.count('\n') + 1
            end_line = content_before_end.count('\n') + 1
            
            multiline_matches.append({
                "start_line": start_line,
                "end_line": end_line,
                "match_text": match.group(0),
                "groups": match.groups(),
                "spans_lines": end_line - start_line + 1
            })
    
    return {
        "pattern": pattern,
        "flags_used": flags,
        "line_matches": matches,
        "multiline_matches": multiline_matches,
        "summary": {
            "total_matches": len(matches) + len(multiline_matches),
            "lines_with_matches": len(set(m["line_number"] for m in matches)),
            "match_distribution": [m["line_number"] for m in matches]
        },
        "pattern_analysis": {
            "pattern_length": len(pattern),
            "uses_groups": '(' in pattern and ')' in pattern,
            "uses_quantifiers": any(q in pattern for q in ['*', '+', '?', '{']),
            "uses_anchors": pattern.startswith('^') or pattern.endswith('$')
        }
    }


def text_replace_pattern(content: str, pattern: str, replacement: str, flags: str = "", max_replacements: int = 0) -> Dict[str, Any]:
    """
    Replace text using regex patterns with full verification and rollback capability.
    
    Purpose: Provides AI with safe regex replacement operations that show exactly
    what changes before and after application, enabling verification and rollback.
    
    Use Cases:
    - Standardize code formatting patterns
    - Update variable names across content
    - Fix consistent typos or formatting issues
    - Apply style transformations with verification
    
    Args:
        content (str): Original text content
        pattern (str): Regex pattern to replace
        replacement (str): Replacement text (can use capture groups like \\1, \\2)
        flags (str): Regex flags (i=ignorecase, m=multiline, s=dotall, x=verbose)
        max_replacements (int): Maximum number of replacements (0 = unlimited)
        
    Returns:
        Dict containing modified content and detailed change information
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    # First, find all matches to show what will be changed
    find_result = text_find_patterns(content, pattern, flags)
    if "error" in find_result:
        return find_result
    
    if find_result["summary"]["total_matches"] == 0:
        return {
            "modified_content": content,
            "no_changes": True,
            "message": "No matches found for the pattern"
        }
    
    # Parse flags
    regex_flags = 0
    if 'i' in flags.lower():
        regex_flags |= re.IGNORECASE
    if 'm' in flags.lower():
        regex_flags |= re.MULTILINE
    if 's' in flags.lower():
        regex_flags |= re.DOTALL
    if 'x' in flags.lower():
        regex_flags |= re.VERBOSE
    
    try:
        compiled_pattern = re.compile(pattern, regex_flags)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {e}"}
    
    # Perform replacement
    count = max_replacements if max_replacements > 0 else 0
    try:
        if count > 0:
            modified_content, replacements_made = compiled_pattern.subn(replacement, content, count=count)
        else:
            modified_content, replacements_made = compiled_pattern.subn(replacement, content)
    except re.error as e:
        return {"error": f"Replacement error: {e}"}
    
    # Generate before/after comparison for verification
    changes = []
    lines = content.splitlines()
    new_lines = modified_content.splitlines()
    
    for match in find_result["line_matches"][:replacements_made]:
        line_num = match["line_number"]
        original_line = match["line_content"]
        new_line = new_lines[line_num-1] if line_num <= len(new_lines) else ""
        
        changes.append({
            "line_number": line_num,
            "before": original_line,
            "after": new_line,
            "match_replaced": match["match_text"]
        })
    
    return {
        "modified_content": modified_content,
        "replacement_summary": {
            "pattern": pattern,
            "replacement": replacement,
            "total_possible_matches": find_result["summary"]["total_matches"],
            "replacements_made": replacements_made,
            "max_limit_applied": max_replacements > 0 and replacements_made == max_replacements
        },
        "changes_made": changes,
        "verification": {
            "content_length_before": len(content),
            "content_length_after": len(modified_content),
            "lines_before": len(lines),
            "lines_after": len(new_lines),
            "change_summary": f"Applied {replacements_made} replacements"
        },
        "rollback_info": {
            "original_content": content,
            "can_rollback": True
        }
    }


def text_summarize_section(content: str, start_line: int, end_line: int, summary_length: str = "medium") -> Dict[str, Any]:
    """
    Replace a verbose text section with an intelligent summary.
    
    Purpose: Enables AI to condense verbose content while preserving key information.
    Provides multiple summary options for different use cases.
    
    Use Cases:
    - Condense long explanations in documentation
    - Create executive summaries of detailed sections
    - Reduce code comments to essential points
    - Compress verbose logs or outputs
    
    Args:
        content (str): Original text content
        start_line (int): Starting line of section to summarize (1-indexed)
        end_line (int): Ending line of section to summarize (1-indexed)
        summary_length (str): "brief" (1-2 lines), "medium" (3-5 lines), "detailed" (preserve key points)
        
    Returns:
        Dict containing summarized content and replacement options
    """
    # Extract the section to summarize
    extraction_result = text_get_lines(content, start_line, end_line)
    if "error" in extraction_result:
        return extraction_result
    
    section_content = extraction_result["extracted_content"]
    word_count = extraction_result["word_count"]
    
    # Generate different summary lengths
    words = section_content.split()
    
    if summary_length == "brief":
        # Extract first sentence and key points
        sentences = re.split(r'[.!?]+', section_content)
        first_sentence = sentences[0].strip() if sentences else ""
        summary = first_sentence + ("." if not first_sentence.endswith(('.', '!', '?')) else "")
        
    elif summary_length == "medium":
        # Extract key sentences and main points
        sentences = re.split(r'[.!?]+', section_content)
        important_sentences = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                important_sentences.append(sentence)
        
        summary = ". ".join(important_sentences)
        if summary and not summary.endswith(('.', '!', '?')):
            summary += "."
            
    else:  # detailed
        # Preserve key points but reduce redundancy
        lines = section_content.splitlines()
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('the ', 'this ', 'that ', 'it ', 'in order to')):
                # Keep lines that seem to contain key information
                if any(keyword in line.lower() for keyword in ['important', 'key', 'main', 'essential', 'critical', ':', '-']):
                    key_lines.append(line)
                elif len(key_lines) < 5:  # Ensure we have some content
                    key_lines.append(line)
        
        summary = '\n'.join(key_lines[:5])  # Limit to 5 key lines
    
    # Create replacement
    replacement_result = text_replace_lines(content, start_line, end_line, summary)
    
    return {
        "original_section": section_content,
        "summary_generated": summary,
        "summary_options": {
            "brief": summary if summary_length == "brief" else section_content.split('.')[0] + ".",
            "medium": summary if summary_length == "medium" else summary,
            "detailed": summary if summary_length == "detailed" else section_content
        },
        "compression_analysis": {
            "original_words": word_count,
            "summary_words": len(summary.split()),
            "compression_ratio": len(summary.split()) / word_count if word_count > 0 else 0,
            "space_saved": word_count - len(summary.split())
        },
        "modified_content": replacement_result.get("modified_content", content),
        "replacement_ready": True
    }


def text_search_content(content: str, search_term: str, context_lines: int = 2, case_sensitive: bool = False) -> Dict[str, Any]:
    """
    Search for specific text with surrounding context lines.
    
    Purpose: Helps AI locate specific content within large texts and understand
    the context around search terms for better decision making.
    
    Use Cases:
    - Find specific functions or variables in code
    - Locate sections that need updating
    - Search for keywords with context
    - Identify content patterns for bulk operations
    
    Args:
        content (str): Text content to search
        search_term (str): Text to search for
        context_lines (int): Number of lines to show before/after matches
        case_sensitive (bool): Whether search should be case sensitive
        
    Returns:
        Dict containing all matches with context and metadata
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    if not search_term:
        return {"error": "Empty search term provided"}
    
    lines = content.splitlines()
    matches = []
    
    # Search each line
    for line_num, line in enumerate(lines, 1):
        if case_sensitive:
            if search_term in line:
                match_positions = [m.start() for m in re.finditer(re.escape(search_term), line)]
            else:
                match_positions = []
        else:
            if search_term.lower() in line.lower():
                match_positions = [m.start() for m in re.finditer(re.escape(search_term), line, re.IGNORECASE)]
            else:
                match_positions = []
        
        for pos in match_positions:
            # Get context lines
            start_context = max(0, line_num - context_lines - 1)
            end_context = min(len(lines), line_num + context_lines)
            context_block = lines[start_context:end_context]
            
            # Highlight the match in the target line
            highlighted_line = (
                line[:pos] + 
                f">>>{search_term}<<<" + 
                line[pos + len(search_term):]
            )
            
            matches.append({
                "line_number": line_num,
                "match_position": pos,
                "line_content": line,
                "highlighted_line": highlighted_line,
                "context": {
                    "lines_before": context_lines,
                    "lines_after": context_lines,
                    "context_block": '\n'.join(context_block),
                    "context_range": f"{start_context + 1}-{end_context}"
                }
            })
    
    return {
        "search_term": search_term,
        "case_sensitive": case_sensitive,
        "total_matches": len(matches),
        "matches": matches,
        "summary": {
            "lines_with_matches": len(set(m["line_number"] for m in matches)),
            "match_distribution": [m["line_number"] for m in matches],
            "first_match_line": matches[0]["line_number"] if matches else None,
            "last_match_line": matches[-1]["line_number"] if matches else None
        },
        "content_stats": {
            "total_lines": len(lines),
            "match_density": len(matches) / len(lines) if lines else 0
        }
    }


def text_extract_section(content: str, start_marker: str, end_marker: str, include_markers: bool = False) -> Dict[str, Any]:
    """
    Extract content between specific text markers.
    
    Purpose: Enables extraction of content between logical boundaries rather than
    just line numbers, useful for content with semantic structure.
    
    Use Cases:
    - Extract content between function signatures
    - Get content between comment blocks
    - Extract sections between headers
    - Isolate content within specific delimiters
    
    Args:
        content (str): Text content to extract from
        start_marker (str): Text marking the beginning of section
        end_marker (str): Text marking the end of section
        include_markers (bool): Whether to include the marker lines in result
        
    Returns:
        Dict containing extracted section and position information
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    
    # Find start and end markers
    start_line = None
    end_line = None
    
    for line_num, line in enumerate(lines, 1):
        if start_marker in line and start_line is None:
            start_line = line_num
        if end_marker in line and start_line is not None:
            end_line = line_num
            break
    
    if start_line is None:
        return {"error": f"Start marker '{start_marker}' not found"}
    
    if end_line is None:
        return {"error": f"End marker '{end_marker}' not found after start marker"}
    
    # Extract section
    if include_markers:
        section_lines = lines[start_line-1:end_line]
        actual_start = start_line
        actual_end = end_line
    else:
        section_lines = lines[start_line:end_line-1]
        actual_start = start_line + 1
        actual_end = end_line - 1
    
    extracted_content = '\n'.join(section_lines)
    
    return {
        "extracted_content": extracted_content,
        "section_info": {
            "start_marker": start_marker,
            "end_marker": end_marker,
            "start_line": start_line,
            "end_line": end_line,
            "extracted_range": f"{actual_start}-{actual_end}",
            "includes_markers": include_markers
        },
        "content_analysis": {
            "line_count": len(section_lines),
            "word_count": len(extracted_content.split()),
            "char_count": len(extracted_content),
            "empty_lines": len([line for line in section_lines if not line.strip()])
        },
        "context": {
            "marker_lines": {
                "start_line_content": lines[start_line-1],
                "end_line_content": lines[end_line-1]
            },
            "surrounding_context": {
                "before": lines[max(0, start_line-3):start_line-1],
                "after": lines[end_line:min(len(lines), end_line+3)]
            }
        }
    }


def text_analyze_structure(content: str) -> Dict[str, Any]:
    """
    Analyze text structure to identify patterns, sections, and optimal manipulation points.
    
    Purpose: Provides AI with deep understanding of content structure to plan
    optimal manipulation strategies and identify logical break points.
    
    Use Cases:
    - Plan content reorganization strategies
    - Identify natural summarization boundaries
    - Find optimal points for content insertion
    - Understand document flow and hierarchy
    
    Args:
        content (str): Text content to analyze
        
    Returns:
        Dict containing structural analysis and manipulation recommendations
    """
    if not content or not content.strip():
        return {"error": "Empty content provided"}
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    # Analyze line patterns
    structural_markers = {
        "headers": [],
        "code_blocks": [],
        "list_items": [],
        "empty_lines": [],
        "long_lines": [],
        "indented_blocks": []
    }
    
    current_indent_level = 0
    indent_changes = []
    
    for line_num, line in enumerate(lines, 1):
        # Headers (markdown style)
        if re.match(r'^#{1,6}\s+', line.strip()):
            level = len(re.match(r'^(#{1,6})', line.strip()).group(1))
            structural_markers["headers"].append({
                "line": line_num,
                "level": level,
                "text": line.strip()
            })
        
        # Code-like lines
        if re.match(r'^\s*(def |class |import |function |var |let |const )', line.strip()):
            structural_markers["code_blocks"].append({
                "line": line_num,
                "type": "code_definition",
                "content": line.strip()
            })
        
        # List items
        if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            structural_markers["list_items"].append({
                "line": line_num,
                "content": line.strip()
            })
        
        # Empty lines (potential section breaks)
        if not line.strip():
            structural_markers["empty_lines"].append(line_num)
        
        # Long lines (potential candidates for summarization)
        if len(line) > 100:
            structural_markers["long_lines"].append({
                "line": line_num,
                "length": len(line),
                "content_preview": line[:80] + "..."
            })
        
        # Track indentation changes
        indent = len(line) - len(line.lstrip())
        if indent != current_indent_level:
            indent_changes.append({
                "line": line_num,
                "from_indent": current_indent_level,
                "to_indent": indent,
                "change_type": "increase" if indent > current_indent_level else "decrease"
            })
            current_indent_level = indent
    
    # Identify logical sections
    logical_sections = []
    section_boundaries = [0]  # Start of document
    
    # Add header positions as boundaries
    for header in structural_markers["headers"]:
        section_boundaries.append(header["line"] - 1)
    
    # Add significant empty line clusters as boundaries
    empty_clusters = []
    for i, empty_line in enumerate(structural_markers["empty_lines"]):
        if i == 0 or empty_line - structural_markers["empty_lines"][i-1] > 3:
            empty_clusters.append(empty_line)
    
    section_boundaries.extend(empty_clusters)
    section_boundaries.append(total_lines)  # End of document
    section_boundaries = sorted(set(section_boundaries))
    
    # Create logical sections
    for i in range(len(section_boundaries) - 1):
        start = section_boundaries[i] + 1
        end = section_boundaries[i + 1]
        if end > start:
            section_content = '\n'.join(lines[start-1:end-1])
            logical_sections.append({
                "section_id": i + 1,
                "start_line": start,
                "end_line": end - 1,
                "line_count": end - start,
                "word_count": len(section_content.split()),
                "content_preview": section_content[:100] + "..." if len(section_content) > 100 else section_content
            })
    
    return {
        "structural_elements": structural_markers,
        "logical_sections": logical_sections,
        "indentation_flow": {
            "indent_changes": indent_changes,
            "max_indent_level": max([change["to_indent"] for change in indent_changes] + [0]),
            "indentation_consistent": len(set([change["to_indent"] for change in indent_changes])) <= 3
        },
        "manipulation_recommendations": {
            "good_summarization_candidates": [
                section for section in logical_sections 
                if section["word_count"] > 50 and section["line_count"] > 5
            ],
            "natural_break_points": section_boundaries[1:-1],  # Exclude document start/end
            "optimal_insertion_points": structural_markers["empty_lines"],
            "regex_operation_complexity": "low" if total_lines < 100 else "medium" if total_lines < 500 else "high"
        },
        "content_characteristics": {
            "hierarchical_structure": len(structural_markers["headers"]) > 0,
            "code_heavy": len(structural_markers["code_blocks"]) > total_lines * 0.1,
            "list_heavy": len(structural_markers["list_items"]) > total_lines * 0.1,
            "well_formatted": len(structural_markers["empty_lines"]) > total_lines * 0.1
        }
    }


def text_compare_sections(content1: str, content2: str, normalize_whitespace: bool = True) -> Dict[str, Any]:
    """
    Compare two text sections to identify differences and similarities.
    
    Purpose: Enables AI to understand differences between text versions and make
    informed decisions about content updates or merging strategies.
    
    Use Cases:
    - Compare before/after content changes
    - Identify differences between similar sections
    - Verify content transformations
    - Merge similar content intelligently
    
    Args:
        content1 (str): First text content
        content2 (str): Second text content  
        normalize_whitespace (bool): Whether to normalize whitespace for comparison
        
    Returns:
        Dict containing detailed comparison analysis
    """
    if not content1 and not content2:
        return {"error": "Both contents are empty"}
    
    # Normalize if requested
    if normalize_whitespace:
        comp1 = re.sub(r'\s+', ' ', content1.strip())
        comp2 = re.sub(r'\s+', ' ', content2.strip())
    else:
        comp1 = content1
        comp2 = content2
    
    lines1 = content1.splitlines()
    lines2 = content2.splitlines()
    
    # Basic similarity
    words1 = set(comp1.lower().split())
    words2 = set(comp2.lower().split())
    common_words = words1.intersection(words2)
    all_words = words1.union(words2)
    word_similarity = len(common_words) / len(all_words) if all_words else 0
    
    # Line-by-line comparison
    line_differences = []
    max_lines = max(len(lines1), len(lines2))
    
    for i in range(max_lines):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        if line1 != line2:
            line_differences.append({
                "line_number": i + 1,
                "content1": line1,
                "content2": line2,
                "difference_type": "added" if not line1 else "removed" if not line2 else "modified"
            })
    
    # Content statistics comparison
    stats1 = text_get_content_stats(content1)
    stats2 = text_get_content_stats(content2)
    
    return {
        "similarity_analysis": {
            "word_similarity": round(word_similarity, 3),
            "exact_match": comp1 == comp2,
            "length_similarity": min(len(comp1), len(comp2)) / max(len(comp1), len(comp2)) if max(len(comp1), len(comp2)) > 0 else 0
        },
        "content_comparison": {
            "content1_stats": stats1.get("basic_stats", {}),
            "content2_stats": stats2.get("basic_stats", {}),
            "line_differences": line_differences,
            "total_differences": len(line_differences)
        },
        "change_summary": {
            "lines_added": len([d for d in line_differences if d["difference_type"] == "added"]),
            "lines_removed": len([d for d in line_differences if d["difference_type"] == "removed"]),
            "lines_modified": len([d for d in line_differences if d["difference_type"] == "modified"])
        },
        "merge_recommendations": {
            "high_similarity": word_similarity > 0.8,
            "simple_merge": len(line_differences) < 5,
            "requires_manual_review": len(line_differences) > 20 or word_similarity < 0.3
        }
    }


def text_validate_regex(pattern: str, test_content: str, flags: str = "") -> Dict[str, Any]:
    """
    Safely test and validate regex patterns against content.
    
    Purpose: Allows AI to verify regex patterns work correctly before applying
    them in bulk operations, preventing errors and unexpected results.
    
    Use Cases:
    - Test regex patterns before bulk replacements
    - Validate pattern syntax and behavior
    - Preview regex results safely
    - Debug complex regex expressions
    
    Args:
        pattern (str): Regex pattern to test
        test_content (str): Content to test pattern against
        flags (str): Regex flags to apply
        
    Returns:
        Dict containing validation results and pattern analysis
    """
    # Parse flags
    regex_flags = 0
    flag_descriptions = []
    if 'i' in flags.lower():
        regex_flags |= re.IGNORECASE
        flag_descriptions.append("Case insensitive")
    if 'm' in flags.lower():
        regex_flags |= re.MULTILINE
        flag_descriptions.append("Multiline mode")
    if 's' in flags.lower():
        regex_flags |= re.DOTALL
        flag_descriptions.append("Dot matches newlines")
    if 'x' in flags.lower():
        regex_flags |= re.VERBOSE
        flag_descriptions.append("Verbose mode")
    
    try:
        compiled_pattern = re.compile(pattern, regex_flags)
    except re.error as e:
        return {
            "valid": False,
            "error": f"Invalid regex syntax: {e}",
            "pattern": pattern,
            "flags": flags
        }
    
    # Test against content
    try:
        matches = list(compiled_pattern.finditer(test_content))
        
        # Analyze matches
        match_details = []
        for i, match in enumerate(matches[:10]):  # Limit to first 10 matches
            match_details.append({
                "match_number": i + 1,
                "matched_text": match.group(0),
                "start_position": match.start(),
                "end_position": match.end(),
                "groups": match.groups(),
                "groupdict": match.groupdict()
            })
        
        # Pattern complexity analysis
        complexity_indicators = {
            "has_groups": '(' in pattern and ')' in pattern,
            "has_quantifiers": any(q in pattern for q in ['*', '+', '?', '{']),
            "has_anchors": pattern.startswith('^') or pattern.endswith('$'),
            "has_character_classes": '[' in pattern and ']' in pattern,
            "has_lookahead": '(?=' in pattern or '(?!' in pattern,
            "has_lookbehind": '(?<=' in pattern or '(?<!' in pattern
        }
        
        return {
            "valid": True,
            "pattern": pattern,
            "flags": flags,
            "flag_descriptions": flag_descriptions,
            "test_results": {
                "total_matches": len(matches),
                "sample_matches": match_details,
                "covers_entire_content": len(matches) == 1 and matches[0].group(0) == test_content
            },
            "pattern_analysis": {
                "pattern_length": len(pattern),
                "complexity_score": sum(complexity_indicators.values()),
                "complexity_indicators": complexity_indicators,
                "estimated_performance": "fast" if len(pattern) < 50 and sum(complexity_indicators.values()) < 3 else "moderate"
            },
            "safety_assessment": {
                "safe_for_bulk_operations": len(matches) > 0 and len(matches) < 100,
                "requires_careful_review": sum(complexity_indicators.values()) > 4,
                "recommended_max_replacements": min(50, len(matches)) if len(matches) > 0 else 0
            }
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Pattern execution error: {e}",
            "pattern": pattern,
            "flags": flags
        }


# Tool function registry for easy integration
tool_functions = {
    "text_get_lines": text_get_lines,
    "text_get_content_stats": text_get_content_stats,
    "text_copy_lines": text_copy_lines,
    "text_replace_lines": text_replace_lines,
    "text_insert_at_line": text_insert_at_line,
    "text_find_patterns": text_find_patterns,
    "text_replace_pattern": text_replace_pattern,
    "text_summarize_section": text_summarize_section,
    "text_search_content": text_search_content,
    "text_extract_section": text_extract_section,
    "text_analyze_structure": text_analyze_structure,
    "text_validate_regex": text_validate_regex,
}


def run_example():
    """Example usage demonstrating text content manipulation capabilities."""
    from flowgen.llm.gemini import Gemini
    
    # Initialize LLM with text manipulation tools
    llm = Gemini(tools=list(tool_functions.values()))
    
    # Sample content for testing
    sample_content = """# Sample Document

This is an introduction paragraph that explains the purpose of this document.
It contains multiple lines and serves as a good example for text manipulation.

## Code Section

```python
def hello_world():
    print("Hello, World!")
    return True
```

The above function demonstrates a simple implementation.

## List Section

- First item in the list
- Second item with more details
- Third item for completeness

## Conclusion

This concludes our sample document with various content types.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this text content and show me its structure, then extract lines 5-8: {sample_content}"}
    ]

    # Run agentic loop
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        response = llm(messages)
        
        # Check for tool calls
        if 'tools' not in response or not response['tools']:
            print("=== FINAL RESPONSE ===")
            print(response.get('content', 'No content'))
            break
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant", 
            "tool_calls": [
                {
                    "id": tool_call.get('id', f"call_{i}"),
                    "function": {
                        "name": tool_call['name'],
                        "arguments": json.dumps(tool_call['arguments'])
                    },
                    "type": "function"
                }
                for i, tool_call in enumerate(response['tools'])
            ]
        })
        
        # Process tool calls
        for i, tool_call in enumerate(response['tools']):
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']
            tool_id = tool_call.get('id', f"call_{i}")
            
            print(f"\n=== EXECUTING: {tool_name} ===")
            print(f"Arguments: {tool_args}")
            
            # Execute tool
            tool_result = tool_functions[tool_name](**tool_args)
            
            print(f"Result: {json.dumps(tool_result, indent=2)}")
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id, 
                "name": tool_name,
                "content": str(tool_result)
            })
        
        iteration += 1


if __name__ == '__main__':
    run_example()