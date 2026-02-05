"""
Code Evolution Engine for the Self-Modifying Code subsystem.

Generates and validates new Python code using Claude LLM API, then
packages the results as ``GeneratedModule`` or ``PromptEvolution``
objects that can be loaded by the Module Registry and Hot Reloader.

This engine is distinct from ``ClaudeCodeClient`` which uses the Claude
Code *CLI* for file-level operations.  ``CodeEvolutionEngine`` uses the
standard Anthropic *Messages API* (via ``ClaudeClient``) for pure text
generation of code snippets and prompt text, then handles validation,
naming, and versioning itself.

Key features:
    - Syntax validation via ``ast.parse`` before accepting generated code
    - Deterministic module naming with hash-based uniqueness
    - Prompt versioning with automatic version number discovery
    - Comprehensive ``[CODE_GEN]`` logging for debugging self-modification

Architecture references:
    - ``architecture.md`` lines 19232-19520  (CodeEvolutionEngine)
    - ``architecture.md`` lines 15714-15862  (CodeGenerationEngine)
    - ``architecture.md`` lines 18601-18684  (ReflectionEngine / Reflection)

Usage::

    from src.meta_agent.code_evolution import CodeEvolutionEngine
    from src.meta_agent.models import Reflection

    engine = CodeEvolutionEngine(project_root="/path/to/project")

    reflection = Reflection(
        critique_valid=True,
        critique_validity_reasoning="Hook lacked metrics",
        is_recurring_pattern=True,
        knowledge_gaps=["How top posts use numbers in hooks"],
        code_changes=["Create hook_selector.py with metric-first hooks"],
    )

    module = await engine.generate_module(
        purpose="Hook selector that prioritises metric-first openings",
        knowledge={"patterns": ["Metric hooks get +20% engagement"]},
        reflection=reflection,
    )

    print(module.name, module.path, module.validated)
"""

import ast
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.meta_agent.models import GeneratedModule, PromptEvolution, Reflection
from src.tools.claude_client import ClaudeClient
from src.utils import utc_now

logger = logging.getLogger(__name__)


class CodeEvolutionEngine:
    """Generates and validates new Python code using Claude LLM.

    Responsible for:
        1. Building detailed prompts from purpose, knowledge, and reflection
        2. Calling the Claude Messages API to generate code/prompt text
        3. Extracting code from markdown-fenced responses
        4. Validating Python syntax before acceptance
        5. Generating deterministic module names
        6. Managing prompt version numbering

    Args:
        claude_client: An instance of ``ClaudeClient`` for LLM calls.
            If ``None``, a default client is created.
        project_root: Path to the project root directory.  Defaults to
            the current working directory.

    Attributes:
        claude: The Claude API client.
        project_root: Resolved project root path.
        generated_dir: Directory where generated modules are saved
            (``<project_root>/src/generated/``).
    """

    # System prompt used for all code generation requests.
    # Instructs the model to produce clean, production-ready Python.
    CODE_GEN_SYSTEM_PROMPT: str = (
        "You are a Python code generator for a LinkedIn content agent.\n"
        "Generate clean, well-documented, production-ready code.\n\n"
        "Requirements:\n"
        "1. Include docstrings explaining purpose\n"
        "2. Add type hints\n"
        '3. Include "# AUTO-GENERATED" header with metadata\n'
        "4. Make code modular and testable\n"
        "5. Follow existing code style in the project\n\n"
        "The code will be automatically integrated into the agent.\n"
        "It must be correct and safe."
    )

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        project_root: str = ".",
    ) -> None:
        self.claude = claude_client or ClaudeClient()
        self.project_root = Path(project_root).resolve()
        self.generated_dir = self.project_root / "src" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[CODE_GEN] Engine initialised. project_root=%s generated_dir=%s",
            self.project_root,
            self.generated_dir,
        )

    # -----------------------------------------------------------------
    # Module generation
    # -----------------------------------------------------------------

    async def generate_module(
        self,
        purpose: str,
        knowledge: Dict[str, Any],
        reflection: Reflection,
    ) -> GeneratedModule:
        """Generate a new Python module based on learnings.

        Builds a detailed prompt from the provided purpose, knowledge
        dictionary, and reflection object, then calls Claude to generate
        the code.  The response is validated for correct Python syntax
        before being packaged as a ``GeneratedModule``.

        Args:
            purpose: Human-readable description of what the module should do.
            knowledge: Research findings and data that inform generation.
            reflection: Self-reflection output containing knowledge gaps,
                process changes, and code change recommendations.

        Returns:
            A ``GeneratedModule`` with the generated code, module name,
            target path, and validation status.

        Raises:
            ValueError: If the generated code fails syntax validation.
        """
        # Build a deterministic generation ID for logging/tracing
        knowledge_hash = hashlib.sha256(
            json.dumps(knowledge, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        generation_id = (
            f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{knowledge_hash}"
        )

        logger.info(
            "[CODE_GEN] Starting module generation: %s\n"
            "  Purpose: %s\n"
            "  Knowledge hash: %s\n"
            "  Reflection gaps: %d",
            generation_id,
            purpose,
            knowledge_hash,
            len(reflection.knowledge_gaps) if reflection.knowledge_gaps else 0,
        )

        # Assemble the prompt
        prompt = (
            "Generate a Python module for the following purpose:\n\n"
            f"PURPOSE: {purpose}\n\n"
            "KNOWLEDGE (from research):\n"
            f"{json.dumps(knowledge, indent=2, default=str)}\n\n"
            "REFLECTION (what we learned):\n"
            f"- Knowledge gaps: {reflection.knowledge_gaps}\n"
            f"- Process changes needed: {reflection.process_changes}\n"
            f"- Suggested code changes: {reflection.code_changes}\n\n"
            "Generate a complete Python module that:\n"
            "1. Encapsulates this knowledge\n"
            "2. Can be imported and used by the Writer agent\n"
            "3. Is well-documented and tested\n"
            "4. Includes example usage\n\n"
            "Return the complete Python code."
        )

        # Call the LLM
        start_time = time.monotonic()
        response = await self.claude.generate(
            prompt=prompt,
            system=self.CODE_GEN_SYSTEM_PROMPT,
            max_tokens=8192,
            temperature=0.3,
        )
        generation_duration = time.monotonic() - start_time

        logger.info(
            "[CODE_GEN] LLM generation completed in %.1fs for %s",
            generation_duration,
            generation_id,
        )

        # Extract code from the response (may be wrapped in markdown fences)
        code = self._extract_code(response)

        code_lines = len(code.split("\n"))
        logger.debug(
            "[CODE_GEN] Extracted code: %d lines for %s",
            code_lines,
            generation_id,
        )

        # Validate syntax
        if not self._validate_syntax(code):
            # Save invalid code for debugging
            invalid_path = self.generated_dir / f"_invalid_{generation_id}.py.failed"
            invalid_path.write_text(code, encoding="utf-8")
            logger.error(
                "[CODE_GEN] Syntax validation FAILED for %s. "
                "Invalid code saved to: %s",
                generation_id,
                invalid_path,
            )
            raise ValueError(
                f"Generated code has syntax errors. "
                f"Invalid code saved to {invalid_path}"
            )

        # Create module metadata
        module_name = self._generate_module_name(purpose)
        module_path = self.generated_dir / f"{module_name}.py"

        logger.info(
            "[CODE_GEN] SUCCESS: %s\n"
            "  Module: %s\n"
            "  Path: %s\n"
            "  Lines: %d\n"
            "  Duration: %.1fs",
            generation_id,
            module_name,
            module_path,
            code_lines,
            generation_duration,
        )

        return GeneratedModule(
            name=module_name,
            path=module_path,
            code=code,
            purpose=purpose,
            generated_at=utc_now(),
            knowledge_source=knowledge,
            validated=True,
        )

    # -----------------------------------------------------------------
    # Prompt evolution
    # -----------------------------------------------------------------

    async def evolve_prompt(
        self,
        current_prompt_path: str,
        reflection: Reflection,
        knowledge: Dict[str, Any],
    ) -> PromptEvolution:
        """Evolve an existing system prompt based on learnings.

        Reads the current prompt file, sends it along with reflection
        insights to Claude, and returns a new version of the prompt.
        The caller is responsible for persisting the new version.

        Args:
            current_prompt_path: Filesystem path to the current prompt file.
            reflection: Self-reflection output driving the evolution.
            knowledge: Research findings to incorporate.

        Returns:
            A ``PromptEvolution`` with the new prompt text, version
            string, and change details.

        Raises:
            FileNotFoundError: If ``current_prompt_path`` does not exist.
        """
        prompt_file = Path(current_prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {current_prompt_path}"
            )

        current_prompt = prompt_file.read_text(encoding="utf-8")

        logger.info(
            "[CODE_GEN] Starting prompt evolution for: %s (%d chars)",
            current_prompt_path,
            len(current_prompt),
        )

        prompt = (
            "Current system prompt:\n"
            "=== START ===\n"
            f"{current_prompt}\n"
            "=== END ===\n\n"
            "Based on these learnings, improve this prompt:\n\n"
            "REFLECTION:\n"
            f"- Valid critique: {reflection.critique_validity_reasoning}\n"
            f"- Pattern detected: {reflection.pattern_description}\n"
            f"- Process changes: {reflection.process_changes}\n\n"
            "KNOWLEDGE:\n"
            f"{json.dumps(knowledge, indent=2, default=str)}\n\n"
            "Rules for evolution:\n"
            "1. Keep what works, change what doesn't\n"
            "2. Add specific rules based on learnings\n"
            "3. Include examples where helpful\n"
            "4. Make instructions clearer and more actionable\n\n"
            "Return the complete new prompt (not a diff)."
        )

        new_prompt = await self.claude.generate(
            prompt=prompt,
            system=(
                "You are a prompt engineer. Improve prompts based on "
                "feedback and learnings."
            ),
            max_tokens=8192,
            temperature=0.4,
        )

        # Determine the next version number
        version = self._get_next_version(current_prompt_path)

        logger.info(
            "[CODE_GEN] Prompt evolved to %s for %s (%d -> %d chars)",
            version,
            current_prompt_path,
            len(current_prompt),
            len(new_prompt),
        )

        return PromptEvolution(
            original_path=current_prompt_path,
            new_prompt=new_prompt,
            version=version,
            changes_made=reflection.prompt_changes,
            knowledge_source=knowledge,
            evolved_at=utc_now(),
        )

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax using the ``ast`` module.

        Args:
            code: Python source code string to validate.

        Returns:
            ``True`` if the code parses successfully, ``False`` otherwise.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as exc:
            logger.error("[CODE_GEN] Syntax error in generated code: %s", exc)
            return False

    def _extract_code(self, response: str) -> str:
        """Extract Python code from an LLM response.

        Handles responses that wrap code in markdown fences
        (``\\`\\`\\`python ... \\`\\`\\```) as well as raw code responses.

        Args:
            response: Raw text response from the LLM.

        Returns:
            Cleaned Python source code string.
        """
        if "```python" in response:
            # Extract from ```python ... ``` block
            code = response.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in response:
            # Extract from generic ``` ... ``` block
            code = response.split("```", 1)[1].split("```", 1)[0]
        else:
            # Assume the entire response is code
            code = response

        return code.strip()

    # -----------------------------------------------------------------
    # Naming and versioning helpers
    # -----------------------------------------------------------------

    def _generate_module_name(self, purpose: str) -> str:
        """Generate a valid Python module name from a purpose description.

        Creates a deterministic name by:
            1. Stripping non-alphanumeric characters
            2. Converting to snake_case
            3. Truncating to 40 characters
            4. Appending a 6-character MD5 hash for uniqueness
            5. Ensuring the result is a valid Python identifier

        Args:
            purpose: Description of what the module does.

        Returns:
            A valid, unique Python module name (e.g. ``hook_selector_a3f2b1``).
        """
        # Clean: remove special chars, lowercase, replace spaces with underscores
        clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", purpose.lower())
        clean_name = re.sub(r"\s+", "_", clean_name.strip())

        # Remove leading/trailing underscores
        clean_name = clean_name.strip("_")

        # Truncate if too long
        if len(clean_name) > 40:
            clean_name = clean_name[:40].rstrip("_")

        # Add hash suffix for uniqueness
        hash_input = f"{purpose}_{utc_now().isoformat()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]

        # Ensure valid Python identifier (can't start with digit)
        if not clean_name or clean_name[0].isdigit():
            clean_name = f"mod_{clean_name}"

        return f"{clean_name}_{hash_suffix}"

    def _get_next_version(self, prompt_path: str) -> str:
        """Determine the next version number for a prompt file.

        Scans the prompt file's directory for existing versioned copies
        matching the pattern ``<stem>_v<N>.txt`` and returns the next
        version string.

        Args:
            prompt_path: Path to the current prompt file.

        Returns:
            Next version string (e.g. ``"v2"``, ``"v3"``).
        """
        prompt_file = Path(prompt_path)
        prompt_dir = prompt_file.parent
        prompt_stem = prompt_file.stem  # filename without extension

        # Collect existing version numbers
        existing_versions: List[int] = []
        for found in prompt_dir.glob(f"{prompt_stem}_v*.txt"):
            match = re.search(r"_v(\d+)\.txt$", found.name)
            if match:
                existing_versions.append(int(match.group(1)))

        # The original file is implicitly v1
        if prompt_file.exists():
            existing_versions.append(1)

        # Compute next version
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1

        return f"v{next_version}"
