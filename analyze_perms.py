#!/usr/bin/env python3
"""Analyze transcripts to find read-only tool patterns for permission allowlist."""

import json
import os
import re
from collections import Counter
from pathlib import Path

# Read-only validation rules (from Claude Code source)
AUTO_ALLOW = {
    'cal', 'uptime', 'cat', 'head', 'tail', 'wc', 'stat', 'strings', 'hexdump',
    'od', 'nl', 'id', 'uname', 'free', 'df', 'du', 'locale', 'groups', 'nproc',
    'basename', 'dirname', 'realpath', 'cut', 'paste', 'tr', 'column', 'tac',
    'rev', 'fold', 'expand', 'unexpand', 'fmt', 'comm', 'cmp', 'numfmt',
    'readlink', 'diff', 'true', 'false', 'sleep', 'which', 'type', 'expr',
    'test', 'getconf', 'seq', 'tsort', 'pr', 'echo', 'printf', 'ls', 'cd', 'find',
    'pwd', 'whoami', 'alias', 'xargs', 'file', 'sed', 'sort', 'man', 'help',
    'netstat', 'ps', 'base64', 'grep', 'egrep', 'fgrep', 'sha256sum', 'sha1sum',
    'md5sum', 'tree', 'date', 'hostname', 'info', 'lsof', 'pgrep', 'tput', 'ss',
    'fd', 'fdfind', 'aki', 'rg', 'jq', 'uniq', 'history', 'arch', 'ifconfig',
    'pyright', 'node', 'python', 'python3', 'bun', 'deno', 'ruby', 'perl',
}

GIT_READ_ONLY = {
    'status', 'log', 'diff', 'show', 'blame', 'branch', 'tag', 'remote',
    'ls-files', 'ls-remote', 'config', 'rev-parse', 'describe', 'stash',
    'reflog', 'shortlog', 'cat-file', 'for-each-ref', 'worktree',
}

GH_READ_ONLY = {
    'pr', 'issue', 'run', 'workflow', 'repo', 'release', 'api', 'auth',
}

def extract_command(cmd):
    """Extract leading command+subcommand from bash command string."""
    # Remove env vars, timeout, sudo
    cmd = re.sub(r'^(timeout\s+\S+\s+|sudo\s+|[A-Z_]+=\S+\s+)*', '', cmd).strip()
    # Remove pipes, &&, ||
    cmd = re.split(r'[|&;]', cmd)[0].strip()

    parts = cmd.split()
    if not parts:
        return None

    main = parts[0]

    # Handle git/gh special
    if main in ('git', 'gh'):
        if len(parts) > 1:
            return f"{main} {parts[1]}"
        return main

    return main

def is_read_only(cmd):
    """Check if command is read-only."""
    if not cmd:
        return False

    parts = cmd.split()
    main = parts[0]

    # Auto-allow list
    if main in AUTO_ALLOW:
        return True

    # Git read-only
    if main == 'git' and len(parts) > 1:
        return parts[1] in GIT_READ_ONLY

    # GH read-only
    if main == 'gh' and len(parts) > 1:
        return parts[1] in GH_READ_ONLY

    # Docker read-only
    if main == 'docker' and len(parts) > 1:
        return parts[1] in ('ps', 'images', 'logs', 'inspect')

    # Drop anything that mutates
    if main in ('git', 'gh') and len(parts) > 1:
        subcmd = parts[1]
        if subcmd in ('push', 'commit', 'merge', 'rebase', 'reset', 'checkout',
                      'branch', 'tag', 'add', 'rm', 'mv', 'clone', 'pull', 'fetch',
                      'stash', 'apply', 'pop', 'delete', 'create'):
            return False

    return False

def scan_transcripts(transcript_dir, limit=50):
    """Scan transcripts for tool usage."""
    tool_counts = Counter()

    all_files = []
    for root, dirs, files in os.walk(transcript_dir):
        for f in files:
            if f.endswith('.jsonl'):
                all_files.append(os.path.join(root, f))

    all_files.sort(key=os.path.getmtime, reverse=True)

    for filepath in all_files[:limit]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get('role') == 'assistant' and 'content' in obj:
                            for item in obj.get('content', []):
                                if item.get('type') == 'tool_use':
                                    tool_name = item.get('name')

                                    if tool_name == 'Bash':
                                        cmd = item.get('input', {}).get('command', '')
                                        extracted = extract_command(cmd)
                                        if extracted and is_read_only(extracted):
                                            tool_counts[f'Bash({extracted})'] += 1
                                    elif tool_name and tool_name.startswith('mcp__'):
                                        tool_counts[tool_name] += 1
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            pass

    return tool_counts

transcript_dir = Path.home() / '.claude' / 'projects'
tool_counts = scan_transcripts(str(transcript_dir))

# Filter to >= 3 occurrences, sort by count
filtered = [(k, v) for k, v in tool_counts.items() if v >= 3]
filtered.sort(key=lambda x: x[1], reverse=True)

print("Top permission patterns to allowlist:")
print()
for i, (pattern, count) in enumerate(filtered[:20], 1):
    print(f"{i:2d}. {pattern:40s} | {count:3d}x")

print()
print("Suggested permissions.allow entries:")
print("[")
for pattern, count in filtered[:20]:
    print(f'  "{pattern}",')
print("]")
