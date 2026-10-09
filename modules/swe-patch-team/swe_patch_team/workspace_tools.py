# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox-style workspace tools for the SWE patch team sample.

The tools mirror the contract of the swegemma harness tools: every tool returns
a JSON string holding either ``{"status": "ok", ...}`` or ``{"status":
"error", "error_type": ..., "error_message": ...}``, file reads are capped at
150 lines / 10,000 characters, and command output is capped at 5,000
characters. The workspace root comes from ``SWE_WORKSPACE`` (default: the
current directory) instead of a fixed ``/workspace``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
from typing import Any

from google.adk.tools.tool_context import ToolContext

COMMAND_TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 5_000
MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10_000

PATCH_SUBMITTED_KEY = "patch_submitted"
SUBMITTED_PATCH_KEY = "submitted_patch"


def workspace_root() -> Path:
  return Path(os.environ.get("SWE_WORKSPACE", os.getcwd())).resolve()


def _ok(**fields: Any) -> str:
  return json.dumps({"status": "ok", **fields})


def _error(error_type: str, message: str, **details: Any) -> str:
  payload: dict[str, Any] = {
      "status": "error",
      "error_type": error_type,
      "error_message": message,
  }
  if details:
    payload["details"] = details
  return json.dumps(payload)


class _PathError(ValueError):
  pass


def _resolve(filepath: str) -> Path:
  """Maps a model-supplied path into the workspace, rejecting traversal."""
  if not filepath or "\x00" in filepath:
    raise _PathError("filepath must be a non-empty path.")
  relative = filepath
  if relative.startswith("/workspace/"):
    relative = relative[len("/workspace/") :]
  relative = relative.lstrip("/")
  if ".." in Path(relative).parts:
    raise _PathError("Path traversal ('..') is not allowed.")
  root = workspace_root()
  resolved = (root / relative).resolve()
  if resolved != root and root not in resolved.parents:
    raise _PathError("Path resolves outside the workspace.")
  return resolved


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
  if len(text) <= limit:
    return text
  return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


async def _run_shell(command: str, timeout: float) -> tuple[int, str, str]:
  proc = await asyncio.create_subprocess_exec(
      "/bin/bash",
      "-c",
      command,
      cwd=workspace_root(),
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      start_new_session=True,
  )
  try:
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
  except asyncio.TimeoutError:
    proc.kill()
    await proc.wait()
    raise
  return (
      proc.returncode or 0,
      stdout.decode("utf-8", errors="replace"),
      stderr.decode("utf-8", errors="replace"),
  )


async def run_command(command: str) -> str:
  """Runs a shell command with bash inside the workspace.

  Put scratch scripts in /tmp, not in the workspace, so they do not end up in
  the submitted patch. Output is truncated to 5,000 characters.

  Args:
    command: The shell command to run.

  Returns:
    A JSON string with status, stdout, stderr and exit_code.
  """
  try:
    code, stdout, stderr = await _run_shell(command, COMMAND_TIMEOUT_SECONDS)
  except asyncio.TimeoutError:
    return _error(
        "TimeoutExceeded",
        f"Command exceeded {COMMAND_TIMEOUT_SECONDS}s and was killed.",
    )
  stdout, stderr = _truncate(stdout), _truncate(stderr)
  if code == 0:
    return _ok(stdout=stdout, stderr=stderr, exit_code=0)
  return _error(
      "CommandError",
      f"Command exited with code {code}.",
      stdout=stdout,
      stderr=stderr,
      exit_code=code,
  )


def read_file(
    filepath: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
  """Reads part of a file in the workspace (1-indexed, inclusive lines).

  A single call returns at most 150 lines and 10,000 characters; use
  start_line and end_line to page through longer files.

  Args:
    filepath: Path relative to the workspace root.
    start_line: First line to return. Defaults to 1.
    end_line: Last line to return. Defaults to the end of the file.

  Returns:
    A JSON string with content, start_line, end_line, total_lines and
    is_truncated.
  """
  try:
    path = _resolve(filepath)
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not path.is_file():
    return _error("FileNotFound", f"No such file: {filepath}")
  lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  total = len(lines)
  first = max(1, start_line or 1)
  last = min(total, end_line or total)
  if first > max(total, 1) or last < first:
    return _error(
        "ValidationError", f"Invalid range {first}-{last} for {total} lines."
    )
  selected: list[str] = []
  chars = 0
  truncated = False
  for number in range(first, last + 1):
    line = lines[number - 1]
    if len(selected) >= MAX_FILE_LINES or chars + len(line) + 1 > MAX_FILE_CHARS:
      truncated = True
      break
    selected.append(line)
    chars += len(line) + 1
  return _ok(
      filepath=filepath,
      content="\n".join(selected),
      start_line=first,
      end_line=first + len(selected) - 1,
      total_lines=total,
      is_truncated=truncated,
  )


def _flexible_replace(
    text: str, old: str, new: str, allow_multiple: bool
) -> tuple[str, int] | None:
  """Line-by-line match ignoring indentation; re-indents ``new`` to fit."""
  text_lines = text.split("\n")
  old_lines = [line.strip() for line in old.strip("\n").split("\n")]
  new_lines = new.strip("\n").split("\n")
  span = len(old_lines)
  starts = [
      i
      for i in range(len(text_lines) - span + 1)
      if [line.strip() for line in text_lines[i : i + span]] == old_lines
  ]
  if not starts or (len(starts) > 1 and not allow_multiple):
    return (text, len(starts)) if starts else None
  new_base = len(new_lines[0]) - len(new_lines[0].lstrip())
  for start in reversed(starts):
    first = text_lines[start]
    indent = first[: len(first) - len(first.lstrip())]
    reindented = [
        indent + line[new_base:] if line.strip() else line for line in new_lines
    ]
    text_lines[start : start + span] = reindented
  return "\n".join(text_lines), len(starts)


def edit_file(
    filepath: str,
    old_string: str,
    new_string: str,
    allow_multiple: bool = False,
) -> str:
  """Replaces old_string with new_string in an existing, non-empty file.

  Matching tries an exact substring first, then a whitespace-insensitive
  line-by-line match. old_string must be unique unless allow_multiple is True.
  Keep edits small and focused.

  Args:
    filepath: Path relative to the workspace root.
    old_string: The text to replace.
    new_string: The replacement text.
    allow_multiple: Replace every occurrence instead of requiring one.

  Returns:
    A JSON string with occurrences and the matching strategy.
  """
  try:
    path = _resolve(filepath)
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not old_string:
    return _error("FileEditError", "old_string must not be empty.")
  if not path.is_file() or path.stat().st_size == 0:
    return _error("FileEditError", f"{filepath} is missing or empty.")
  text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
  old = old_string.replace("\r\n", "\n")
  new = new_string.replace("\r\n", "\n")

  count = text.count(old)
  strategy = "exact"
  if count == 1 or (count > 1 and allow_multiple):
    updated = text.replace(old, new)
  elif count > 1:
    return _error(
        "FileEditError",
        f"old_string matches {count} places; add context or set"
        " allow_multiple=True.",
    )
  else:
    strategy = "flexible"
    result = _flexible_replace(text, old, new, allow_multiple)
    if result is None:
      return _error("FileEditError", "old_string was not found in the file.")
    updated, count = result
    if count > 1 and not allow_multiple:
      return _error(
          "FileEditError",
          f"old_string matches {count} places; add context or set"
          " allow_multiple=True.",
      )
  path.write_text(updated, encoding="utf-8")
  return _ok(filepath=filepath, occurrences=count, strategy=strategy)


def write_file(filepath: str, content: str) -> str:
  """Creates or overwrites a file in the workspace, creating parent dirs.

  Args:
    filepath: Path relative to the workspace root.
    content: Full file content.

  Returns:
    A JSON string with the written size in bytes.
  """
  try:
    path = _resolve(filepath)
  except _PathError as e:
    return _error("ValidationError", str(e))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(content, encoding="utf-8")
  return _ok(filepath=filepath, size=len(content.encode("utf-8")))


async def search_code(pattern: str, path: str = ".") -> str:
  """Searches the workspace with a regular expression (read-only grep).

  Args:
    pattern: Extended regular expression to search for.
    path: File or directory, relative to the workspace root.

  Returns:
    A JSON string with matching "file:line:text" lines.
  """
  try:
    target = _resolve(path)
    re.compile(pattern)
  except (_PathError, re.error) as e:
    return _error("ValidationError", str(e))
  proc = await asyncio.create_subprocess_exec(
      "grep",
      "-rnIE",
      "--exclude-dir=.git",
      "--exclude-dir=__pycache__",
      "-m",
      "20",
      "-e",
      pattern,
      str(target),
      cwd=workspace_root(),
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
  )
  stdout, _ = await proc.communicate()
  root_prefix = str(workspace_root()) + os.sep
  text = stdout.decode("utf-8", errors="replace").replace(root_prefix, "")
  matches = text.splitlines()
  return _ok(matches=_truncate("\n".join(matches)), count=len(matches))


async def list_files(path: str = ".", max_depth: int = 3) -> str:
  """Lists files under a workspace directory, skipping .git and caches.

  Args:
    path: Directory relative to the workspace root.
    max_depth: How many directory levels to descend.

  Returns:
    A JSON string with up to 150 relative paths.
  """
  try:
    target = _resolve(path)
  except _PathError as e:
    return _error("ValidationError", str(e))
  if not target.is_dir():
    return _error("FileNotFound", f"No such directory: {path}")
  _, stdout, _ = await _run_shell(
      f"find {_shell_quote(str(target))} -maxdepth {int(max_depth)}"
      " -not -path '*/.git*' -not -path '*/__pycache__*' -not -name '*.pyc'"
      " | sort | head -150",
      30,
  )
  root_prefix = str(workspace_root()) + os.sep
  return _ok(paths=stdout.replace(root_prefix, "").splitlines())


def _shell_quote(value: str) -> str:
  return "'" + value.replace("'", "'\\''") + "'"


async def get_diff() -> str:
  """Returns the current uncommitted change set as a unified diff (read-only).

  Returns:
    A JSON string with the diff, truncated to 5,000 characters.
  """
  code, stdout, stderr = await _run_shell(
      "git add -N . && git diff HEAD", COMMAND_TIMEOUT_SECONDS
  )
  if code != 0:
    return _error("CommandError", _truncate(stderr), exit_code=code)
  return _ok(diff=_truncate(stdout), is_empty=not stdout.strip())


async def submit_patch(tool_context: ToolContext) -> str:
  """Captures the final git diff as the patch and ends the task.

  Call this exactly once, as the very last action, after verifying the change
  and removing scratch files from the workspace.

  Returns:
    A JSON string with patch_size and files_changed.
  """
  code, stdout, stderr = await _run_shell(
      "git add -N . && git diff --binary HEAD", COMMAND_TIMEOUT_SECONDS
  )
  if code != 0:
    return _error("CommandError", _truncate(stderr), exit_code=code)
  tool_context.state[SUBMITTED_PATCH_KEY] = stdout
  tool_context.state[PATCH_SUBMITTED_KEY] = True
  files_changed = len(re.findall(r"^diff --git ", stdout, flags=re.MULTILINE))
  return _ok(patch_size=len(stdout), files_changed=files_changed)
