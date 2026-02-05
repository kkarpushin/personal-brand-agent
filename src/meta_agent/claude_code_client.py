"""
Client for running Claude Code in headless mode for code generation.

Wraps the ``claude`` CLI tool to execute code generation, file modification,
prompt evolution, and general development tasks.  Used by the Meta-Agent
subsystem when the self-modifying code engine determines that new code or
prompt changes are needed.

Key design decisions:
    - **Subprocess-based**: Invokes ``claude`` as a child process rather than
      using an SDK, keeping the integration simple and stateless.
    - **JSON output**: All responses are parsed from Claude Code's JSON
      output format for structured result handling.
    - **Fail-fast**: Raises ``RuntimeError`` immediately if Claude Code is
      not installed.  Execution failures are captured in
      ``ClaudeCodeResult.error`` rather than raising, to allow the caller
      to decide how to handle partial results.
    - **Budget & turn limits**: Configurable safety limits prevent runaway
      executions.

Architecture reference:
    - ``architecture.md`` lines 18754-19029 (Claude Code Client)
    - ``architecture.md`` lines 19183-19229 (Server Configuration)

Usage::

    client = ClaudeCodeClient(project_root="/path/to/project")

    # Generate a new module
    result = client.generate_module(
        purpose="Hook type selector based on content type",
        context={"patterns": ["question hooks perform +20%"]},
        target_path="src/generated/hook_selector.py",
    )

    if result.success:
        print(f"Created: {result.files_created}")
    else:
        print(f"Failed: {result.error}")
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from src.meta_agent.models import ClaudeCodeResult

logger = logging.getLogger(__name__)


class ClaudeCodeClient:
    """Client for running Claude Code in headless mode.

    Used by the Meta-Agent for code generation and modification.
    Requires the ``claude`` CLI to be installed and accessible on ``PATH``.

    Args:
        project_root: Absolute path to the project root directory.
            Claude Code will operate within this directory.
        allowed_tools: List of Claude Code tools to enable.
            Defaults to ``["Read", "Write", "Edit", "Bash", "Glob", "Grep"]``.
        max_turns: Maximum number of agentic turns per execution.
            Defaults to ``15``.
        max_budget_usd: Maximum budget in USD per execution.
            Defaults to ``5.0``.

    Raises:
        RuntimeError: If Claude Code CLI is not installed or not
            functioning correctly.

    Attributes:
        project_root: Path to the project root.
        allowed_tools: Enabled Claude Code tools.
        max_turns: Turn limit per execution.
        max_budget_usd: Budget limit per execution.
    """

    # Default tools that cover typical code generation/modification needs.
    # Intentionally excludes WebSearch/WebFetch for security.
    DEFAULT_ALLOWED_TOOLS: List[str] = [
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
    ]

    # Subprocess timeout in seconds (5 minutes).
    _EXECUTION_TIMEOUT_SECONDS: int = 300

    def __init__(
        self,
        project_root: str,
        allowed_tools: Optional[List[str]] = None,
        max_turns: int = 15,
        max_budget_usd: float = 5.0,
    ) -> None:
        self.project_root = Path(project_root)
        self.allowed_tools = allowed_tools or list(self.DEFAULT_ALLOWED_TOOLS)
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

        # Verify Claude Code is available before accepting any work.
        self._verify_installation()

        logger.info(
            "[CLAUDE_CODE] Client initialised. root=%s tools=%s max_turns=%d budget=$%.2f",
            self.project_root,
            self.allowed_tools,
            self.max_turns,
            self.max_budget_usd,
        )

    # -----------------------------------------------------------------
    # Installation verification
    # -----------------------------------------------------------------

    def _verify_installation(self) -> None:
        """Check that the ``claude`` CLI is available and functional.

        Runs ``claude --version`` with a short timeout to confirm the
        binary is on PATH and responds correctly.

        Raises:
            RuntimeError: If the CLI cannot be found or returns a
                non-zero exit code.
        """
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude Code returned non-zero exit code ({result.returncode}). "
                    f"stderr: {result.stderr.strip()}"
                )
            logger.debug(
                "[CLAUDE_CODE] Verified installation: %s",
                result.stdout.strip(),
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found on PATH. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Claude Code CLI timed out during version check. "
                "Ensure it is properly installed and configured."
            )

    # -----------------------------------------------------------------
    # Public API: Task-specific methods
    # -----------------------------------------------------------------

    def generate_module(
        self,
        purpose: str,
        context: dict,
        target_path: str,
    ) -> ClaudeCodeResult:
        """Generate a new Python module using Claude Code.

        Claude Code will read the existing project structure, generate the
        module at ``target_path``, create tests, and verify them.

        Args:
            purpose: Human-readable description of what the module should do.
            context: Knowledge and research findings that inform the generation.
                Will be serialized to JSON and included in the prompt.
            target_path: Relative path (from project root) where the module
                should be saved.

        Returns:
            ``ClaudeCodeResult`` with generation outcome.
        """
        logger.info(
            "[CLAUDE_CODE] generate_module: purpose='%s' target='%s'",
            purpose[:80],
            target_path,
        )

        prompt = (
            f"Create a new Python module at {target_path}\n\n"
            f"PURPOSE:\n{purpose}\n\n"
            f"CONTEXT (research findings):\n{json.dumps(context, indent=2, default=str)}\n\n"
            "REQUIREMENTS:\n"
            "1. Read existing code in src/ to understand project style\n"
            "2. Create the module with proper type hints and docstrings\n"
            '3. Include "# AUTO-GENERATED by Meta-Agent" header\n'
            "4. Add unit tests in tests/ directory\n"
            "5. Run the tests to verify the code works\n"
            "6. If tests fail, fix the code\n\n"
            "After creating the module, run: python -m pytest tests/ -v\n\n"
            "Report what you created and test results."
        )

        return self._execute(prompt)

    def modify_file(
        self,
        file_path: str,
        modification: str,
        reason: str,
    ) -> ClaudeCodeResult:
        """Modify an existing file based on learnings.

        Claude Code reads the current file, understands the existing code,
        makes minimal focused changes, adds explanatory comments, and
        verifies by running tests.

        Args:
            file_path: Path to the file to modify (relative or absolute).
            modification: Description of the change to make.
            reason: Why this change is needed (from reflection/research).

        Returns:
            ``ClaudeCodeResult`` with modification outcome.
        """
        logger.info(
            "[CLAUDE_CODE] modify_file: path='%s' reason='%s'",
            file_path,
            reason[:80],
        )

        prompt = (
            f"Modify the file: {file_path}\n\n"
            f"CHANGE NEEDED:\n{modification}\n\n"
            f"REASON (from self-reflection):\n{reason}\n\n"
            "REQUIREMENTS:\n"
            "1. Read the current file first\n"
            "2. Understand the existing code\n"
            "3. Make minimal, focused changes\n"
            "4. Add a comment explaining the change\n"
            "5. Run relevant tests after modification\n"
            "6. If tests fail, fix or rollback\n\n"
            "Report what you changed and why."
        )

        return self._execute(prompt)

    def evolve_prompt(
        self,
        prompt_path: str,
        learnings: list,
        examples: list,
    ) -> ClaudeCodeResult:
        """Evolve a system prompt based on learnings.

        Creates a new version of the prompt file while preserving the
        version history in a ``prompts/versions/`` directory.

        Args:
            prompt_path: Path to the current prompt file.
            learnings: List of learning strings to incorporate.
            examples: List of good example dicts to weave into the prompt.

        Returns:
            ``ClaudeCodeResult`` with the new prompt version info.
        """
        logger.info(
            "[CLAUDE_CODE] evolve_prompt: path='%s' learnings=%d examples=%d",
            prompt_path,
            len(learnings),
            len(examples),
        )

        prompt = (
            f"Evolve the system prompt at: {prompt_path}\n\n"
            f"LEARNINGS TO INCORPORATE:\n{json.dumps(learnings, indent=2, default=str)}\n\n"
            f"GOOD EXAMPLES TO ADD:\n{json.dumps(examples, indent=2, default=str)}\n\n"
            "REQUIREMENTS:\n"
            "1. Read the current prompt\n"
            "2. Read the version history in prompts/versions/\n"
            "3. Create a new version with improvements\n"
            "4. Save old version to prompts/versions/\n"
            "5. Update changelog.json with what changed and why\n"
            "6. Keep what works, improve what doesn't\n\n"
            "Report the changes made."
        )

        return self._execute(prompt)

    def run_complex_task(self, task_description: str) -> ClaudeCodeResult:
        """Run any complex development task.

        Claude Code will autonomously determine what needs to be done,
        execute the necessary steps, and verify the results.

        Args:
            task_description: Free-form description of the task to accomplish.

        Returns:
            ``ClaudeCodeResult`` with task outcome.
        """
        logger.info(
            "[CLAUDE_CODE] run_complex_task: '%s'",
            task_description[:120],
        )

        prompt = (
            f"TASK:\n{task_description}\n\n"
            f"You have full access to the project at {self.project_root}.\n"
            "Figure out what needs to be done and do it.\n"
            "Run tests to verify your changes work.\n"
            "Report what you did."
        )

        return self._execute(prompt)

    # -----------------------------------------------------------------
    # Internal execution engine
    # -----------------------------------------------------------------

    def _execute(self, prompt: str) -> ClaudeCodeResult:
        """Execute a prompt via the Claude Code CLI.

        Builds the command line with the configured tools, output format,
        working directory, and turn limits.  Runs the subprocess with
        a timeout and parses the JSON response.

        Args:
            prompt: The full prompt text to send to Claude Code.

        Returns:
            ``ClaudeCodeResult`` capturing success/failure, output text,
            session metadata, cost, and file change information.
        """
        cmd = [
            "claude",
            "-p", prompt,
            "--allowedTools", ",".join(self.allowed_tools),
            "--output-format", "json",
            "--cwd", str(self.project_root),
            "--max-turns", str(self.max_turns),
        ]

        logger.debug(
            "[CLAUDE_CODE] Executing command with %d-char prompt, timeout=%ds",
            len(prompt),
            self._EXECUTION_TIMEOUT_SECONDS,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._EXECUTION_TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Non-zero exit code"
                logger.error(
                    "[CLAUDE_CODE] Execution failed (rc=%d): %s",
                    result.returncode,
                    error_msg[:200],
                )
                return ClaudeCodeResult(
                    success=False,
                    result="",
                    session_id="",
                    cost_usd=0.0,
                    duration_ms=0,
                    files_created=[],
                    files_modified=[],
                    error=error_msg,
                )

            # Parse structured JSON output from Claude Code
            output = json.loads(result.stdout)

            parsed_result = ClaudeCodeResult(
                success=True,
                result=output.get("result", ""),
                session_id=output.get("session_id", ""),
                cost_usd=output.get("total_cost_usd", 0.0),
                duration_ms=output.get("duration_ms", 0),
                files_created=output.get("files_created", []),
                files_modified=output.get("files_modified", []),
            )

            logger.info(
                "[CLAUDE_CODE] Execution succeeded. session=%s cost=$%.4f "
                "duration=%dms created=%d modified=%d",
                parsed_result.session_id,
                parsed_result.cost_usd,
                parsed_result.duration_ms,
                len(parsed_result.files_created),
                len(parsed_result.files_modified),
            )

            return parsed_result

        except subprocess.TimeoutExpired:
            logger.error(
                "[CLAUDE_CODE] Execution timed out after %ds",
                self._EXECUTION_TIMEOUT_SECONDS,
            )
            return ClaudeCodeResult(
                success=False,
                result="",
                session_id="",
                cost_usd=0.0,
                duration_ms=0,
                files_created=[],
                files_modified=[],
                error=f"Timeout: task took longer than {self._EXECUTION_TIMEOUT_SECONDS} seconds",
            )

        except json.JSONDecodeError as exc:
            logger.error(
                "[CLAUDE_CODE] Failed to parse JSON response: %s", exc,
            )
            return ClaudeCodeResult(
                success=False,
                result="",
                session_id="",
                cost_usd=0.0,
                duration_ms=0,
                files_created=[],
                files_modified=[],
                error=f"Failed to parse Claude Code JSON response: {exc}",
            )

        except OSError as exc:
            logger.error(
                "[CLAUDE_CODE] OS error during execution: %s", exc,
            )
            return ClaudeCodeResult(
                success=False,
                result="",
                session_id="",
                cost_usd=0.0,
                duration_ms=0,
                files_created=[],
                files_modified=[],
                error=f"OS error: {exc}",
            )
