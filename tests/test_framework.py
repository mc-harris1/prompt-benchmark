from __future__ import annotations

from pathlib import Path

import pytest

from prompt_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkRunner,
    ExactMatchMetric,
    JsonlResultStore,
    PromptTemplate,
    ProviderRegistry,
    ProviderResponse,
)


class EchoExpectedProvider:
    name = "echo-expected"

    def generate(
        self,
        prompt: str,
        *,
        case: BenchmarkCase,
        template: PromptTemplate,
    ) -> ProviderResponse:
        del prompt, template
        return ProviderResponse(output=case.expected_output or "", metadata={"provider": self.name})


def test_runner_renders_prompts_evaluates_metrics_and_persists_results(tmp_path: Path) -> None:
    dataset = BenchmarkDataset.from_cases(
        "greetings",
        [
            BenchmarkCase(
                case_id="case-1",
                input_variables={"name": "Ada"},
                expected_output="Hello Ada!",
            ),
            BenchmarkCase(
                case_id="case-2",
                input_variables={"name": "Grace"},
                expected_output="Hello Grace!",
            ),
        ],
    )
    template = PromptTemplate(name="hello-template", template="Hello {name}!")
    store = JsonlResultStore(tmp_path / "results" / "runs.jsonl")
    runner = BenchmarkRunner(metrics=[ExactMatchMetric()], store=store)

    result = runner.run(
        dataset=dataset,
        provider=EchoExpectedProvider(),
        template=template,
        run_id="run-1",
        metadata={"suite": "smoke"},
    )

    assert result.summary_metrics == {"exact_match": 1.0}
    assert [case_result.prompt for case_result in result.case_results] == [
        "Hello Ada!",
        "Hello Grace!",
    ]
    saved_runs = store.load_runs()
    assert saved_runs[0]["run_id"] == "run-1"
    assert saved_runs[0]["metadata"] == {"suite": "smoke"}
    assert saved_runs[0]["case_results"][0]["response"]["metadata"] == {
        "provider": "echo-expected",
    }


def test_template_render_rejects_missing_variables() -> None:
    template = PromptTemplate(
        name="question-answer", template="Question: {question}\nAnswer: {answer}"
    )

    with pytest.raises(ValueError, match="missing required variables: answer"):
        template.render({"question": "What is 2 + 2?"})


def test_provider_registry_rejects_duplicate_names() -> None:
    registry = ProviderRegistry()
    provider = EchoExpectedProvider()

    registry.register(provider)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(provider)
