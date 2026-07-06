# prompt-benchmark

Evaluate prompts, models, and LLM providers using reproducible benchmarks.

## What it provides

- Structured benchmark inputs with `BenchmarkDataset` and `BenchmarkCase`
- Reusable templates with `PromptTemplate`
- Pluggable providers via the `ModelProvider` protocol
- Pluggable metrics via the `EvaluationMetric` protocol
- End-to-end execution and aggregation with `BenchmarkRunner`
- Optional JSONL persistence with `JsonlResultStore`

## Installation

This project targets Python 3.12+ and uses `uv` for local development.

```bash
uv sync --dev
```

## Quick start

```python
from prompt_benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkRunner,
    ExactMatchMetric,
    PromptTemplate,
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
        # This provider ignores the rendered prompt and returns expected output.
        del prompt, template
        return ProviderResponse(output=case.expected_output or "")


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
runner = BenchmarkRunner(metrics=[ExactMatchMetric()])

result = runner.run(
    dataset=dataset,
    provider=EchoExpectedProvider(),
    template=template,
    run_id="demo-run",
)

print(result.summary_metrics)  # {'exact_match': 1.0}
```

## Core concepts

### 1. Provider interface

Implement a provider by exposing:

- `name: str`
- `generate(prompt, *, case, template) -> ProviderResponse`

The runner renders each case using the template, then calls your provider.

### 2. Metric interface

Implement a metric by exposing:

- `name: str`
- `evaluate(*, case, prompt, response) -> MetricResult`

`ExactMatchMetric` is included as a baseline metric.

### 3. Result persistence

Pass `JsonlResultStore(path)` to `BenchmarkRunner` to append each run as one JSON line.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run pyright
```

## License

Apache 2.0
