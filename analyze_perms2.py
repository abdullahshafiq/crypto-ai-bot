#!/usr/bin/env python3
"""Analyze Claude Code transcripts for read-only tool patterns."""

import json
import os
import re
from collections import Counter
from pathlib import Path

# Auto-allow commands (no rule needed)
AUTO_ALLOW = {
    'cat', 'head', 'tail', 'wc', 'diff', 'ls', 'cd', 'find', 'echo', 'printf',
    'pwd', 'whoami', 'stat', 'file', 'which', 'grep', 'rg', 'jq', 'tree',
    'ps', 'netstat', 'lsof', 'date', 'hostname', 'sort', 'uniq', 'tr', 'cut',
}

GIT_READ_ONLY = {
    'status', 'log', 'diff', 'show', 'blame', 'branch', 'tag', 'remote',
    'ls-files', 'ls-remote', 'config', 'rev-parse', 'describe', 'stash',
    'reflog', 'shortlog', 'cat-file', 'worktree',
}

GH_READ_ONLY = {
    'pr', 'issue', 'run', 'workflow', 'repo', 'release', 'api', 'auth',
}

def extract_command(cmd):
    """Extract leading command+subcommand from bash command."""
    if not cmd:
        return None

    # Remove env vars, sudo, timeout
    cmd = re.sub(r'^(timeout\s+\S+\s+|sudo\s+|[A-Z_]+=\S+\s+)*', '', cmd).strip()
    # Handle pipes, &&, etc
    cmd = re.split(r'[|&;]', cmd)[0].strip()

    parts = cmd.split()
    if not parts:
        return None

    main = parts[0]
    if main in ('git', 'gh') and len(parts) > 1:
        return f"{main} {parts[1]}"
    return main

def is_read_only(cmd):
    """Check if command is read-only."""
    if not cmd:
        return False

    parts = cmd.split()
    main = parts[0]

    # Auto-allow
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

    return False

def scan_transcripts(transcript_dir, limit=50):
    """Scan JSONL transcripts for tool usage."""
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

                        # Look for assistant messages with tool_use
                        if (obj.get('type') == 'assistant' and
                            isinstance(obj.get('message'), dict) and
                            obj['message'].get('role') == 'assistant'):

                            content = obj['message'].get('content', [])
                            for item in content:
                                if item.get('type') == 'tool_use':
                                    tool_name = item.get('name')

                                    if tool_name == 'Bash':
                                        cmd_input = item.get('input', {})
                                        if isinstance(cmd_input, dict):
                                            cmd = cmd_input.get('command', '')
                                        else:
                                            cmd = str(cmd_input)

                                        extracted = extract_command(cmd)
                                        if extracted and is_read_only(extracted):
                                            tool_counts[f'Bash({extracted})'] += 1

                                    elif tool_name and tool_name.startswith('mcp__'):
                                        # MCP tools
                                        if 'read' in tool_name or 'get' in tool_name or 'list' in tool_name:
                                            tool_counts[tool_name] += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception as e:
            pass

    return tool_counts

transcript_dir = Path.home() / '.claude' / 'projects'
tool_counts = scan_transcripts(str(transcript_dir))

# Filter >= 3, sort by count
filtered = [(k, v) for k, v in tool_counts.items() if v >= 3]
filtered.sort(key=lambda x: x[1], reverse=True)

print("Read-only tool patterns found:")
print()
for i, (pattern, count) in enumerate(filtered[:25], 1):
    print(f"{i:2d}. {pattern:50s} | {count:3d}x")

if filtered:
    print("\n\nAdd these to .claude/settings.json permissions.allow:")
    for pattern, _ in filtered[:20]:
        print(f'  "{pattern}",')
else:
    print("\nNo frequently-used read-only patterns found.")
