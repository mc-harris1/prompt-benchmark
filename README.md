# prompt-benchmark

Evaluate prompts, models, and LLM providers using reproducible benchmarks.

## Features

- `BenchmarkRunner` for executing benchmark datasets against pluggable model providers
- `PromptTemplate` objects for reusable prompt rendering
- `BenchmarkDataset` and `BenchmarkCase` models for structured benchmark inputs
- `EvaluationMetric` hooks for configurable scoring
- `JsonlResultStore` for append-only structured run storage
- `uv`-managed development workflow with Ruff, Pyright, pytest, pre-commit, and GitHub Actions

## Quick start

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run pyright
```

```python
from prompt_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkRunner,
    ExactMatchMetric,
    PromptTemplate,
    ProviderResponse,
)


class EchoProvider:
    name = "echo"

    def generate(self, prompt: str, *, case: BenchmarkCase, template: PromptTemplate) -> ProviderResponse:
        del template
        return ProviderResponse(output=prompt, metadata={"case_id": case.case_id})


dataset = BenchmarkDataset.from_cases(
    "demo",
    [
        BenchmarkCase(case_id="1", input_variables={"name": "Ada"}, expected_output="Hello Ada!"),
    ],
)
template = PromptTemplate(name="greeting", template="Hello {name}!")
runner = BenchmarkRunner(metrics=[ExactMatchMetric()])
result = runner.run(dataset=dataset, provider=EchoProvider(), template=template)

print(result.summary_metrics)
```
