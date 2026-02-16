"""GitHub Repository Explorer MCP Server.

Lets an AI agent navigate and explore any GitHub repository
using the gh CLI (authenticated, structured, fast).

Usage: Given a GitHub URL like https://github.com/langchain-ai/langchain,
the agent can browse the file tree, read files, search code, list issues/PRs, etc.
"""

import subprocess
import json
import re
from typing import Optional
from fastmcp import FastMCP

mcp = FastMCP("GitHub Explorer")


def _parse_repo(repo_or_url: str) -> str:
    """Extract owner/repo from a GitHub URL or pass through owner/repo format."""
    # Handle full URLs
    m = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/.*)?$", repo_or_url)
    if m:
        return m.group(1)
    # Already owner/repo format
    if "/" in repo_or_url and not repo_or_url.startswith("http"):
        return repo_or_url.split("/")[0] + "/" + repo_or_url.split("/")[1]
    raise ValueError(f"Cannot parse repo from: {repo_or_url}")


def _gh(args: list[str], max_output: int = 50000) -> str:
    """Run a gh CLI command and return output."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    output = result.stdout
    if len(output) > max_output:
        output = output[:max_output] + f"\n... (truncated, {len(result.stdout)} total chars)"
    return output


@mcp.tool
def repo_info(repo: str) -> str:
    """Get repository overview: description, stars, language, topics, default branch.

    Args:
        repo: GitHub repo URL or owner/repo (e.g. "langchain-ai/langchain")
    """
    owner_repo = _parse_repo(repo)
    return _gh(["api", f"repos/{owner_repo}",
                "--jq", '{name: .name, full_name: .full_name, description: .description, '
                         'stars: .stargazers_count, forks: .forks_count, language: .language, '
                         'default_branch: .default_branch, topics: .topics, '
                         'open_issues: .open_issues_count, license: .license.spdx_id, '
                         'updated_at: .updated_at}'])


@mcp.tool
def list_tree(repo: str, path: str = "", branch: str = "", depth: int = 1) -> str:
    """List files and directories in a repo path (like ls).

    Args:
        repo: GitHub repo URL or owner/repo
        path: Directory path within the repo (empty = root)
        branch: Branch name (empty = default branch)
        depth: 1 = immediate children only, 0 = full recursive tree
    """
    owner_repo = _parse_repo(repo)
    if not branch:
        branch = json.loads(_gh(["api", f"repos/{owner_repo}", "--jq", '{"b": .default_branch}']))["b"]

    if depth == 0:
        # Full recursive tree
        jq = '.tree[] | .path + (if .type == "tree" then "/" else "" end)'
        result = _gh(["api", f"repos/{owner_repo}/git/trees/{branch}?recursive=1", "--jq", jq])
    else:
        # Get tree for specific path
        if path:
            # First get the tree SHA for the path
            jq_sha = f'.tree[] | select(.path == "{path}" and .type == "tree") | .sha'
            sha = _gh(["api", f"repos/{owner_repo}/git/trees/{branch}?recursive=1", "--jq", jq_sha]).strip()
            if not sha or sha.startswith("Error"):
                return f"Path not found: {path}"
            jq = '.tree[] | .path + (if .type == "tree" then "/" else "" end)'
            result = _gh(["api", f"repos/{owner_repo}/git/trees/{sha}", "--jq", jq])
        else:
            jq = '.tree[] | .path + (if .type == "tree" then "/" else "" end)'
            result = _gh(["api", f"repos/{owner_repo}/git/trees/{branch}", "--jq", jq])

    if path and depth == 1:
        return f"Contents of {path}/:\n{result}"
    return result


@mcp.tool
def read_file(repo: str, path: str, branch: str = "") -> str:
    """Read a file's contents from a GitHub repo.

    Args:
        repo: GitHub repo URL or owner/repo
        path: File path within the repo (e.g. "src/main.py")
        branch: Branch name (empty = default branch)
    """
    owner_repo = _parse_repo(repo)
    ref_param = f"?ref={branch}" if branch else ""
    # Use the contents API which returns base64-decoded content for small files
    # For larger files, use the raw media type
    return _gh(["api", f"repos/{owner_repo}/contents/{path}{ref_param}",
                "-H", "Accept: application/vnd.github.raw+json"])


@mcp.tool
def search_code(repo: str, query: str, max_results: int = 20) -> str:
    """Search for code within a GitHub repo.

    Args:
        repo: GitHub repo URL or owner/repo
        query: Search query (code, function names, etc.)
        max_results: Maximum results to return (default 20)
    """
    owner_repo = _parse_repo(repo)
    search_query = f"{query} repo:{owner_repo}"
    jq = f'.items[:{ max_results}] | .[] | {{path: .path, score: .score, url: .html_url, matches: [.text_matches[]?.fragment]}}'
    return _gh(["api", "search/code",
                "-X", "GET",
                "-f", f"q={search_query}",
                "-H", "Accept: application/vnd.github.text-match+json",
                "--jq", jq])


@mcp.tool
def read_readme(repo: str) -> str:
    """Get the rendered README of a repository as markdown.

    Args:
        repo: GitHub repo URL or owner/repo
    """
    owner_repo = _parse_repo(repo)
    return _gh(["api", f"repos/{owner_repo}/readme",
                "-H", "Accept: application/vnd.github.raw+json"])


@mcp.tool
def list_issues(repo: str, state: str = "open", labels: str = "", max_results: int = 20) -> str:
    """List issues in a repository.

    Args:
        repo: GitHub repo URL or owner/repo
        state: "open", "closed", or "all" (default: "open")
        labels: Comma-separated label names to filter by
        max_results: Maximum results (default 20)
    """
    owner_repo = _parse_repo(repo)
    params = f"state={state}&per_page={max_results}"
    if labels:
        params += f"&labels={labels}"
    jq = '.[] | {number: .number, title: .title, state: .state, labels: [.labels[].name], created: .created_at, comments: .comments}'
    return _gh(["api", f"repos/{owner_repo}/issues?{params}", "--jq", jq])


@mcp.tool
def list_prs(repo: str, state: str = "open", max_results: int = 20) -> str:
    """List pull requests in a repository.

    Args:
        repo: GitHub repo URL or owner/repo
        state: "open", "closed", or "all" (default: "open")
        max_results: Maximum results (default 20)
    """
    owner_repo = _parse_repo(repo)
    jq = '.[] | {number: .number, title: .title, state: .state, author: .user.login, created: .created_at, draft: .draft}'
    return _gh(["api", f"repos/{owner_repo}/pulls?state={state}&per_page={max_results}", "--jq", jq])


@mcp.tool
def list_releases(repo: str, max_results: int = 5) -> str:
    """List recent releases/tags of a repository.

    Args:
        repo: GitHub repo URL or owner/repo
        max_results: Maximum results (default 5)
    """
    owner_repo = _parse_repo(repo)
    jq = f'.[:{ max_results}] | .[] | {{tag: .tag_name, name: .name, date: .published_at, prerelease: .prerelease}}'
    return _gh(["api", f"repos/{owner_repo}/releases", "--jq", jq])


@mcp.tool
def list_branches(repo: str, max_results: int = 30) -> str:
    """List branches in a repository.

    Args:
        repo: GitHub repo URL or owner/repo
        max_results: Maximum results (default 30)
    """
    owner_repo = _parse_repo(repo)
    jq = f'.[:{ max_results}] | .[].name'
    return _gh(["api", f"repos/{owner_repo}/branches?per_page={max_results}", "--jq", jq])


@mcp.tool
def repo_commits(repo: str, path: str = "", branch: str = "", max_results: int = 10) -> str:
    """List recent commits, optionally filtered to a specific file/directory.

    Args:
        repo: GitHub repo URL or owner/repo
        path: Optional file/directory path to filter commits
        branch: Branch name (empty = default)
        max_results: Maximum results (default 10)
    """
    owner_repo = _parse_repo(repo)
    params = f"per_page={max_results}"
    if path:
        params += f"&path={path}"
    if branch:
        params += f"&sha={branch}"
    jq = '.[] | {sha: .sha[:8], message: .commit.message | split("\n")[0], author: .commit.author.name, date: .commit.author.date}'
    return _gh(["api", f"repos/{owner_repo}/commits?{params}", "--jq", jq])


@mcp.tool
def gh_raw(endpoint: str, method: str = "GET", jq_filter: str = "") -> str:
    """Run any GitHub API endpoint directly for advanced queries.

    Args:
        endpoint: API endpoint (e.g. "repos/owner/repo/languages")
        method: HTTP method (default GET)
        jq_filter: Optional jq filter for output
    """
    args = ["api", endpoint, "-X", method]
    if jq_filter:
        args += ["--jq", jq_filter]
    return _gh(args)


tool_functions = {
    "repo_info": repo_info,
    "list_tree": list_tree,
    "read_file": read_file,
    "search_code": search_code,
    "read_readme": read_readme,
    "list_issues": list_issues,
    "list_prs": list_prs,
    "list_releases": list_releases,
    "list_branches": list_branches,
    "repo_commits": repo_commits,
    "gh_raw": gh_raw,
}

if __name__ == "__main__":
    mcp.run()
