"""
Tests for the meta-agent subsystem of the LinkedIn Super Agent.

Covers:
    - Import accessibility for all meta_agent modules
    - Class existence and attribute checks
    - Enum value verification (ModificationRiskLevel)
    - Basic instantiation with mocked dependencies
    - Method existence on key classes

These are primarily import and instantiation tests since we cannot call
real LLM APIs in a test environment.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# 1-2. IMPORT TESTS: Verify every meta_agent module is importable and
#       exports the expected classes.
# ============================================================================


class TestMetaAgentImports:
    """Verify that all meta_agent submodules are importable and export
    the expected classes or functions."""

    def test_import_single_call_evaluator(self):
        from src.meta_agent.single_call_evaluator import SingleCallEvaluator

        assert SingleCallEvaluator is not None

    def test_import_modification_safety(self):
        from src.meta_agent.modification_safety import ModificationSafetySystem

        assert ModificationSafetySystem is not None

    def test_import_reflection_engine(self):
        from src.meta_agent.reflection_engine import ReflectionEngine

        assert ReflectionEngine is not None

    def test_import_research_agent(self):
        from src.meta_agent.research_agent import ResearchAgent

        assert ResearchAgent is not None

    def test_import_claude_code_client(self):
        from src.meta_agent.claude_code_client import ClaudeCodeClient

        assert ClaudeCodeClient is not None

    def test_import_code_evolution(self):
        from src.meta_agent.code_evolution import CodeEvolutionEngine

        assert CodeEvolutionEngine is not None

    def test_import_knowledge_base(self):
        from src.meta_agent.knowledge_base import KnowledgeBase

        assert KnowledgeBase is not None

    def test_import_deep_improvement_loop(self):
        from src.meta_agent.deep_improvement_loop import DeepImprovementLoop

        assert DeepImprovementLoop is not None

    def test_import_experimentation(self):
        from src.meta_agent.experimentation import ExperimentationEngine

        assert ExperimentationEngine is not None

    def test_import_meta_agent(self):
        from src.meta_agent.meta_agent import MetaAgent

        assert MetaAgent is not None

    def test_import_package_init_exports(self):
        """The package __init__.py re-exports all main classes."""
        import src.meta_agent as pkg

        expected_names = [
            "MetaAgent",
            "SingleCallEvaluator",
            "ModificationSafetySystem",
            "ResearchAgent",
            "ReflectionEngine",
            "KnowledgeBase",
            "ClaudeCodeClient",
            "CodeEvolutionEngine",
            "DeepImprovementLoop",
            "ExperimentationEngine",
        ]
        for name in expected_names:
            assert hasattr(pkg, name), f"src.meta_agent missing export: {name}"


# ============================================================================
# 3. MODELS IMPORT TESTS
# ============================================================================


class TestModelsImports:
    """Verify that models.py exports key types and constants."""

    def test_import_modification_risk_level(self):
        from src.meta_agent.models import ModificationRiskLevel

        assert ModificationRiskLevel is not None

    def test_import_modification_request(self):
        from src.meta_agent.models import ModificationRequest

        assert ModificationRequest is not None

    def test_import_single_call_evaluation(self):
        from src.meta_agent.models import SingleCallEvaluation

        assert SingleCallEvaluation is not None

    def test_import_evaluation_criterion(self):
        from src.meta_agent.models import EvaluationCriterion

        assert EvaluationCriterion is not None

    def test_import_evaluation_rubric(self):
        from src.meta_agent.models import evaluation_rubric

        assert isinstance(evaluation_rubric, dict)
        assert len(evaluation_rubric) > 0

    def test_import_visual_evaluation(self):
        from src.meta_agent.models import VisualEvaluation

        assert VisualEvaluation is not None

    def test_import_rollback_trigger(self):
        from src.meta_agent.models import RollbackTrigger

        assert RollbackTrigger is not None

    def test_import_modification_risk_classification(self):
        from src.meta_agent.models import modification_risk_classification

        assert isinstance(modification_risk_classification, dict)
        assert len(modification_risk_classification) > 0

    def test_import_config_path_mapping(self):
        from src.meta_agent.models import CONFIG_PATH_MAPPING, get_config_path

        assert isinstance(CONFIG_PATH_MAPPING, dict)
        assert callable(get_config_path)

    def test_import_research_models(self):
        from src.meta_agent.models import (
            ResearchTrigger,
            ResearchQuery,
            ResearchFinding,
            ResearchRecommendation,
            ResearchReport,
        )

        assert ResearchTrigger is not None
        assert ResearchQuery is not None
        assert ResearchFinding is not None
        assert ResearchRecommendation is not None
        assert ResearchReport is not None

    def test_import_experiment_models(self):
        from src.meta_agent.models import (
            ExperimentStatus,
            ExperimentVariant,
            Experiment,
        )

        assert ExperimentStatus is not None
        assert ExperimentVariant is not None
        assert Experiment is not None

    def test_import_reflection_and_dialogue(self):
        from src.meta_agent.models import DialogueSummary, Reflection

        assert DialogueSummary is not None
        assert Reflection is not None

    def test_import_improvement_result(self):
        from src.meta_agent.models import ImprovementResult

        assert ImprovementResult is not None

    def test_import_learning(self):
        from src.meta_agent.models import Learning

        assert Learning is not None

    def test_import_code_generation_models(self):
        from src.meta_agent.models import (
            CapabilityType,
            CapabilityGap,
            GeneratedCode,
            GeneratedModule,
            PromptEvolution,
        )

        assert CapabilityType is not None
        assert CapabilityGap is not None
        assert GeneratedCode is not None
        assert GeneratedModule is not None
        assert PromptEvolution is not None


# ============================================================================
# 4. MODIFICATION RISK LEVEL ENUM VALUES
# ============================================================================


class TestModificationRiskLevel:
    """Verify ModificationRiskLevel enum has the expected members."""

    def test_has_low(self):
        from src.meta_agent.models import ModificationRiskLevel

        assert hasattr(ModificationRiskLevel, "LOW")
        assert ModificationRiskLevel.LOW.value == "low"

    def test_has_medium(self):
        from src.meta_agent.models import ModificationRiskLevel

        assert hasattr(ModificationRiskLevel, "MEDIUM")
        assert ModificationRiskLevel.MEDIUM.value == "medium"

    def test_has_high(self):
        from src.meta_agent.models import ModificationRiskLevel

        assert hasattr(ModificationRiskLevel, "HIGH")
        assert ModificationRiskLevel.HIGH.value == "high"

    def test_has_critical(self):
        from src.meta_agent.models import ModificationRiskLevel

        assert hasattr(ModificationRiskLevel, "CRITICAL")
        assert ModificationRiskLevel.CRITICAL.value == "critical"

    def test_exactly_four_members(self):
        from src.meta_agent.models import ModificationRiskLevel

        members = list(ModificationRiskLevel)
        assert len(members) == 4


# ============================================================================
# 5. SINGLE CALL EVALUATION DATACLASS
# ============================================================================


class TestSingleCallEvaluation:
    """Verify the SingleCallEvaluation dataclass can be instantiated."""

    def test_create_single_call_evaluation(self):
        from src.meta_agent.models import SingleCallEvaluation

        evaluation = SingleCallEvaluation(
            scores={"hook_strength": 8, "specificity": 7},
            weighted_total=7.5,
            criterion_feedback={
                "hook_strength": {
                    "quote": "test quote",
                    "score": 8,
                    "explanation": "Good hook",
                },
            },
            strengths=["Strong opening"],
            weaknesses=["Needs more data"],
            specific_suggestions=["Add numbers to claims"],
            passes_threshold=True,
            recommended_revisions=None,
            patterns_detected=["vague claims"],
            knowledge_gaps=["industry benchmarks"],
        )
        assert evaluation.weighted_total == 7.5
        assert evaluation.passes_threshold is True
        assert len(evaluation.scores) == 2
        assert evaluation.recommended_revisions is None

    def test_evaluation_with_revisions(self):
        from src.meta_agent.models import SingleCallEvaluation

        evaluation = SingleCallEvaluation(
            scores={"hook_strength": 4},
            weighted_total=4.0,
            criterion_feedback={},
            strengths=[],
            weaknesses=["Weak hook"],
            specific_suggestions=["Rewrite first line"],
            passes_threshold=False,
            recommended_revisions=["Improve hook"],
            patterns_detected=[],
            knowledge_gaps=[],
        )
        assert evaluation.passes_threshold is False
        assert evaluation.recommended_revisions == ["Improve hook"]


# ============================================================================
# 6. SINGLE CALL EVALUATOR INSTANTIATION
# ============================================================================


class TestSingleCallEvaluatorInstantiation:
    """Verify SingleCallEvaluator can be created with mocked dependencies."""

    @patch("src.meta_agent.single_call_evaluator._get_default_claude")
    def test_instantiation_with_default_client(self, mock_get_default):
        """SingleCallEvaluator() should call _get_default_claude when no
        client is provided."""
        mock_get_default.return_value = MagicMock()

        from src.meta_agent.single_call_evaluator import SingleCallEvaluator

        evaluator = SingleCallEvaluator()
        assert evaluator is not None
        assert evaluator.claude is not None
        mock_get_default.assert_called_once()

    @patch("src.meta_agent.single_call_evaluator._get_default_claude")
    def test_instantiation_with_injected_client(self, mock_get_default):
        """When a claude_client is injected, _get_default_claude is NOT called."""
        mock_client = MagicMock()

        from src.meta_agent.single_call_evaluator import SingleCallEvaluator

        evaluator = SingleCallEvaluator(claude_client=mock_client)
        assert evaluator.claude is mock_client
        mock_get_default.assert_not_called()

    @patch("src.meta_agent.single_call_evaluator._get_default_claude")
    def test_has_evaluate_method(self, mock_get_default):
        mock_get_default.return_value = MagicMock()

        from src.meta_agent.single_call_evaluator import SingleCallEvaluator

        evaluator = SingleCallEvaluator()
        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)

    @patch("src.meta_agent.single_call_evaluator._get_default_claude")
    def test_has_rubric(self, mock_get_default):
        mock_get_default.return_value = MagicMock()

        from src.meta_agent.single_call_evaluator import SingleCallEvaluator

        evaluator = SingleCallEvaluator()
        assert hasattr(evaluator, "rubric")
        assert isinstance(evaluator.rubric, dict)
        assert len(evaluator.rubric) > 0


# ============================================================================
# 7. MODIFICATION SAFETY SYSTEM INSTANTIATION
# ============================================================================


class TestModificationSafetySystemInstantiation:
    """Verify ModificationSafetySystem can be created with mocked deps."""

    def test_instantiation(self):
        from src.meta_agent.modification_safety import ModificationSafetySystem

        mock_db = MagicMock()
        mock_telegram = MagicMock()
        system = ModificationSafetySystem(db=mock_db, telegram_notifier=mock_telegram)
        assert system is not None
        assert system.db is mock_db
        assert system.telegram is mock_telegram

    def test_has_request_modification(self):
        from src.meta_agent.modification_safety import ModificationSafetySystem

        system = ModificationSafetySystem(db=MagicMock(), telegram_notifier=MagicMock())
        assert hasattr(system, "request_modification")
        assert callable(system.request_modification)

    def test_has_check_rollback_triggers(self):
        from src.meta_agent.modification_safety import ModificationSafetySystem

        system = ModificationSafetySystem(db=MagicMock(), telegram_notifier=MagicMock())
        assert hasattr(system, "check_rollback_triggers")


# ============================================================================
# 8. REFLECTION ENGINE INSTANTIATION
# ============================================================================


class TestReflectionEngineInstantiation:
    """Verify ReflectionEngine can be created with mocked dependencies."""

    @patch("src.meta_agent.reflection_engine.get_claude")
    def test_instantiation_default_client(self, mock_get_claude):
        mock_get_claude.return_value = MagicMock()

        from src.meta_agent.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()
        assert engine is not None
        assert engine.claude is not None
        mock_get_claude.assert_called_once()

    def test_instantiation_with_injected_client(self):
        from src.meta_agent.reflection_engine import ReflectionEngine

        mock_client = MagicMock()
        engine = ReflectionEngine(claude_client=mock_client)
        assert engine.claude is mock_client

    @patch("src.meta_agent.reflection_engine.get_claude")
    def test_has_reflect_method(self, mock_get_claude):
        mock_get_claude.return_value = MagicMock()

        from src.meta_agent.reflection_engine import ReflectionEngine

        engine = ReflectionEngine()
        assert hasattr(engine, "reflect")
        assert callable(engine.reflect)


# ============================================================================
# 9. RESEARCH AGENT INSTANTIATION
# ============================================================================


class TestResearchAgentInstantiation:
    """Verify ResearchAgent can be created with mocked dependencies."""

    @patch("src.meta_agent.research_agent.get_claude")
    def test_instantiation(self, mock_get_claude):
        mock_get_claude.return_value = MagicMock()

        from src.meta_agent.research_agent import ResearchAgent

        agent = ResearchAgent(
            perplexity_client=MagicMock(),
            linkedin_scraper=MagicMock(),
            analytics_db=MagicMock(),
        )
        assert agent is not None
        assert agent.perplexity is not None
        assert agent.linkedin is not None
        assert agent.db is not None

    def test_instantiation_with_injected_claude(self):
        from src.meta_agent.research_agent import ResearchAgent

        mock_claude = MagicMock()
        agent = ResearchAgent(
            perplexity_client=MagicMock(),
            linkedin_scraper=MagicMock(),
            analytics_db=MagicMock(),
            claude_client=mock_claude,
        )
        assert agent.claude is mock_claude

    @patch("src.meta_agent.research_agent.get_claude")
    def test_has_research_method(self, mock_get_claude):
        mock_get_claude.return_value = MagicMock()

        from src.meta_agent.research_agent import ResearchAgent

        agent = ResearchAgent(
            perplexity_client=MagicMock(),
            linkedin_scraper=MagicMock(),
            analytics_db=MagicMock(),
        )
        assert hasattr(agent, "research")
        assert callable(agent.research)

    @patch("src.meta_agent.research_agent.get_claude")
    def test_has_should_research_method(self, mock_get_claude):
        mock_get_claude.return_value = MagicMock()

        from src.meta_agent.research_agent import ResearchAgent

        agent = ResearchAgent(
            perplexity_client=MagicMock(),
            linkedin_scraper=MagicMock(),
            analytics_db=MagicMock(),
        )
        assert hasattr(agent, "should_research")
        assert callable(agent.should_research)


# ============================================================================
# 10. CLAUDE CODE CLIENT -- import only (instantiation requires CLI)
# ============================================================================


class TestClaudeCodeClientImport:
    """ClaudeCodeClient's constructor calls subprocess to verify the
    ``claude`` CLI is installed.  We only test that the class is importable
    and has the expected API surface without actually instantiating."""

    def test_class_exists(self):
        from src.meta_agent.claude_code_client import ClaudeCodeClient

        assert ClaudeCodeClient is not None

    def test_has_expected_methods(self):
        from src.meta_agent.claude_code_client import ClaudeCodeClient

        for method_name in [
            "generate_module",
            "modify_file",
            "evolve_prompt",
            "run_complex_task",
        ]:
            assert hasattr(ClaudeCodeClient, method_name), (
                f"ClaudeCodeClient missing method: {method_name}"
            )

    @patch("src.meta_agent.claude_code_client.subprocess")
    def test_instantiation_with_mocked_subprocess(self, mock_subprocess):
        """If the CLI check is mocked, the client can be instantiated."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "claude-code 1.0.0"
        mock_subprocess.run.return_value = mock_result

        from src.meta_agent.claude_code_client import ClaudeCodeClient

        client = ClaudeCodeClient(project_root=".")
        assert client is not None
        assert client.max_turns == 15


# ============================================================================
# 11. CODE EVOLUTION ENGINE INSTANTIATION
# ============================================================================


class TestCodeEvolutionEngineInstantiation:
    """Verify CodeEvolutionEngine can be created with mocked deps."""

    @patch("src.meta_agent.code_evolution.ClaudeClient")
    def test_instantiation_default_client(self, mock_claude_cls):
        mock_claude_cls.return_value = MagicMock()

        from src.meta_agent.code_evolution import CodeEvolutionEngine

        engine = CodeEvolutionEngine()
        assert engine is not None
        assert engine.generated_dir.exists()

    def test_instantiation_with_injected_client(self):
        from src.meta_agent.code_evolution import CodeEvolutionEngine

        mock_client = MagicMock()
        engine = CodeEvolutionEngine(claude_client=mock_client, project_root=".")
        assert engine.claude is mock_client

    @patch("src.meta_agent.code_evolution.ClaudeClient")
    def test_has_generate_module_method(self, mock_claude_cls):
        mock_claude_cls.return_value = MagicMock()

        from src.meta_agent.code_evolution import CodeEvolutionEngine

        engine = CodeEvolutionEngine()
        assert hasattr(engine, "generate_module")
        assert callable(engine.generate_module)

    @patch("src.meta_agent.code_evolution.ClaudeClient")
    def test_has_evolve_prompt_method(self, mock_claude_cls):
        mock_claude_cls.return_value = MagicMock()

        from src.meta_agent.code_evolution import CodeEvolutionEngine

        engine = CodeEvolutionEngine()
        assert hasattr(engine, "evolve_prompt")
        assert callable(engine.evolve_prompt)


# ============================================================================
# 12. KNOWLEDGE BASE INSTANTIATION
# ============================================================================


class TestKnowledgeBaseInstantiation:
    """Verify KnowledgeBase can be created with a mocked DB."""

    def test_instantiation(self):
        from src.meta_agent.knowledge_base import KnowledgeBase

        mock_db = MagicMock()
        kb = KnowledgeBase(db=mock_db)
        assert kb is not None
        assert kb.db is mock_db

    def test_has_store_learning(self):
        from src.meta_agent.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db=MagicMock())
        assert hasattr(kb, "store_learning")
        assert callable(kb.store_learning)

    def test_has_query_relevant(self):
        from src.meta_agent.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db=MagicMock())
        assert hasattr(kb, "query_relevant")
        assert callable(kb.query_relevant)

    def test_has_get_applicable_rules(self):
        from src.meta_agent.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(db=MagicMock())
        assert hasattr(kb, "get_applicable_rules")
        assert callable(kb.get_applicable_rules)


# ============================================================================
# 13. DEEP IMPROVEMENT LOOP INSTANTIATION
# ============================================================================


class TestDeepImprovementLoopInstantiation:
    """Verify DeepImprovementLoop can be created with mocked deps."""

    def test_instantiation(self):
        from src.meta_agent.deep_improvement_loop import DeepImprovementLoop

        loop = DeepImprovementLoop(
            critic=MagicMock(),
            reflector=MagicMock(),
            researcher=MagicMock(),
            code_evolver=MagicMock(),
            knowledge_base=MagicMock(),
            db=MagicMock(),
        )
        assert loop is not None

    def test_has_run_method(self):
        from src.meta_agent.deep_improvement_loop import DeepImprovementLoop

        loop = DeepImprovementLoop(
            critic=MagicMock(),
            reflector=MagicMock(),
            researcher=MagicMock(),
            code_evolver=MagicMock(),
            knowledge_base=MagicMock(),
            db=MagicMock(),
        )
        assert hasattr(loop, "run")
        assert callable(loop.run)


# ============================================================================
# 14. EXPERIMENTATION ENGINE INSTANTIATION
# ============================================================================


class TestExperimentationEngineInstantiation:
    """Verify ExperimentationEngine can be created with a mocked DB."""

    def test_instantiation(self):
        from src.meta_agent.experimentation import ExperimentationEngine

        engine = ExperimentationEngine(db=MagicMock())
        assert engine is not None
        assert engine.current_experiment is None

    def test_has_create_experiment(self):
        from src.meta_agent.experimentation import ExperimentationEngine

        engine = ExperimentationEngine(db=MagicMock())
        assert hasattr(engine, "create_experiment")
        assert callable(engine.create_experiment)

    def test_has_assign_variant(self):
        from src.meta_agent.experimentation import ExperimentationEngine

        engine = ExperimentationEngine(db=MagicMock())
        assert hasattr(engine, "assign_variant")
        assert callable(engine.assign_variant)


# ============================================================================
# 15. META AGENT INSTANTIATION
# ============================================================================


class TestMetaAgentInstantiation:
    """Verify MetaAgent can be created with mocked dependencies."""

    def test_instantiation(self):
        from src.meta_agent.meta_agent import MetaAgent

        meta = MetaAgent(
            evaluator=MagicMock(),
            researcher=MagicMock(),
            safety_system=MagicMock(),
            experimenter=MagicMock(),
            db=MagicMock(),
        )
        assert meta is not None

    def test_has_evaluate_draft(self):
        from src.meta_agent.meta_agent import MetaAgent

        meta = MetaAgent(
            evaluator=MagicMock(),
            researcher=MagicMock(),
            safety_system=MagicMock(),
            experimenter=MagicMock(),
            db=MagicMock(),
        )
        assert hasattr(meta, "evaluate_draft")
        assert callable(meta.evaluate_draft)

    def test_has_run_improvement_cycle(self):
        from src.meta_agent.meta_agent import MetaAgent

        meta = MetaAgent(
            evaluator=MagicMock(),
            researcher=MagicMock(),
            safety_system=MagicMock(),
            experimenter=MagicMock(),
            db=MagicMock(),
        )
        assert hasattr(meta, "run_improvement_cycle")
        assert callable(meta.run_improvement_cycle)

    def test_has_suggest_experiment(self):
        from src.meta_agent.meta_agent import MetaAgent

        meta = MetaAgent(
            evaluator=MagicMock(),
            researcher=MagicMock(),
            safety_system=MagicMock(),
            experimenter=MagicMock(),
            db=MagicMock(),
        )
        assert hasattr(meta, "suggest_experiment")
        assert callable(meta.suggest_experiment)


# ============================================================================
# 16. ADDITIONAL MODEL TESTS
# ============================================================================


class TestModificationRequest:
    """Verify the ModificationRequest dataclass can be instantiated."""

    def test_create_modification_request(self):
        from src.meta_agent.models import ModificationRequest, ModificationRiskLevel

        req = ModificationRequest(
            id="test-001",
            modification_type="posting_time_adjustment",
            risk_level=ModificationRiskLevel.LOW,
            component="scheduler",
            before_state={"posting_hour": 9},
            after_state={"posting_hour": 11},
            reasoning="Analytics show higher engagement at 11am",
            triggered_by="research",
            supporting_data={"avg_engagement_9am": 45, "avg_engagement_11am": 72},
            status="pending",
        )
        assert req.id == "test-001"
        assert req.risk_level == ModificationRiskLevel.LOW
        assert req.component == "scheduler"
        assert req.status == "pending"


class TestGetConfigPath:
    """Verify config path lookup works correctly."""

    def test_known_component(self):
        from src.meta_agent.models import get_config_path

        path = get_config_path("writer")
        assert "writer" in path
        assert path.endswith(".json")

    def test_unknown_component_raises(self):
        from src.meta_agent.models import get_config_path

        with pytest.raises(ValueError, match="Unknown component"):
            get_config_path("nonexistent_component")


class TestGeneratedCode:
    """Verify the GeneratedCode dataclass and its is_valid method."""

    def test_is_valid_all_passing(self):
        from src.meta_agent.models import GeneratedCode

        code = GeneratedCode(
            module_name="test_module",
            file_path="src/generated/test.py",
            code="print('hello')",
            description="A test module",
            gap_id="gap-001",
            syntax_valid=True,
            type_check_passed=True,
            tests_passed=True,
            security_passed=True,
        )
        assert code.is_valid() is True

    def test_is_valid_type_check_not_required(self):
        """type_check_passed is informational only -- not required for
        is_valid to return True."""
        from src.meta_agent.models import GeneratedCode

        code = GeneratedCode(
            module_name="test_module",
            file_path="src/generated/test.py",
            code="print('hello')",
            description="A test module",
            gap_id="gap-001",
            syntax_valid=True,
            type_check_passed=False,  # should not block
            tests_passed=True,
            security_passed=True,
        )
        assert code.is_valid() is True

    def test_is_valid_fails_on_syntax(self):
        from src.meta_agent.models import GeneratedCode

        code = GeneratedCode(
            module_name="test_module",
            file_path="src/generated/test.py",
            code="invalid code",
            description="A test module",
            gap_id="gap-001",
            syntax_valid=False,
            tests_passed=True,
            security_passed=True,
        )
        assert code.is_valid() is False


class TestDialogueSummaryDefaults:
    """Verify DialogueSummary works with all-default fields."""

    def test_defaults(self):
        from src.meta_agent.models import DialogueSummary

        summary = DialogueSummary()
        assert summary.weaknesses == []
        assert summary.suggestions == []
        assert summary.knowledge_gaps == []
        assert summary.research_queries == []
        assert summary.confidence_in_suggestions == 0.5


class TestAutonomyLevel:
    """Verify AutonomyLevel IntEnum members and ordering."""

    def test_level_values(self):
        from src.meta_agent.models import AutonomyLevel

        assert AutonomyLevel.HUMAN_ALL == 1
        assert AutonomyLevel.HUMAN_POSTS == 2
        assert AutonomyLevel.AUTO_HIGH_SCORE == 3
        assert AutonomyLevel.FULL_AUTONOMY == 4

    def test_comparison(self):
        from src.meta_agent.models import AutonomyLevel

        assert AutonomyLevel.AUTO_HIGH_SCORE > AutonomyLevel.HUMAN_POSTS
        assert AutonomyLevel.HUMAN_ALL < AutonomyLevel.FULL_AUTONOMY


class TestExperimentStatus:
    """Verify ExperimentStatus enum members."""

    def test_status_values(self):
        from src.meta_agent.models import ExperimentStatus

        assert ExperimentStatus.DRAFT.value == "draft"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.STOPPED_EARLY.value == "stopped_early"
        assert ExperimentStatus.WINNER_APPLIED.value == "winner_applied"
