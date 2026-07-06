from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from string import Formatter
from typing import Any, Protocol, runtime_checkable

JSONDict = dict[str, Any]


def _new_json_dict() -> JSONDict:
    return {}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    input_variables: JSONDict
    expected_output: str | None = None
    metadata: JSONDict = field(default_factory=_new_json_dict)

    def to_dict(self) -> JSONDict:
        return {
            "case_id": self.case_id,
            "input_variables": self.input_variables,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    name: str
    cases: tuple[BenchmarkCase, ...]
    description: str | None = None
    metadata: JSONDict = field(default_factory=_new_json_dict)

    @classmethod
    def from_cases(
        cls,
        name: str,
        cases: list[BenchmarkCase],
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkDataset:
        return cls(
            name=name,
            cases=tuple(cases),
            description=description,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    template: str
    metadata: JSONDict = field(default_factory=_new_json_dict)

    def render(self, variables: Mapping[str, Any]) -> str:
        required_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.template)
            if field_name is not None
        }
        missing_fields = sorted(field for field in required_fields if field not in variables)
        if missing_fields:
            joined_fields = ", ".join(missing_fields)
            raise ValueError(
                f"Template {self.name!r} is missing required variables: {joined_fields}",
            )
        return self.template.format_map(dict(variables))


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    output: str
    metadata: JSONDict = field(default_factory=_new_json_dict)

    def to_dict(self) -> JSONDict:
        return {"output": self.output, "metadata": self.metadata}


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    score: float
    passed: bool | None = None
    details: JSONDict = field(default_factory=_new_json_dict)

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    prompt: str
    expected_output: str | None
    response: ProviderResponse
    metrics: tuple[MetricResult, ...]
    case_metadata: JSONDict = field(default_factory=_new_json_dict)

    def to_dict(self) -> JSONDict:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "response": self.response.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "case_metadata": self.case_metadata,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    run_id: str
    dataset_name: str
    provider_name: str
    template_name: str
    started_at: datetime
    completed_at: datetime
    case_results: tuple[CaseResult, ...]
    summary_metrics: dict[str, float]
    metadata: JSONDict = field(default_factory=_new_json_dict)

    def to_dict(self) -> JSONDict:
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "provider_name": self.provider_name,
            "template_name": self.template_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "case_results": [case_result.to_dict() for case_result in self.case_results],
            "summary_metrics": self.summary_metrics,
            "metadata": self.metadata,
        }


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        case: BenchmarkCase,
        template: PromptTemplate,
    ) -> ProviderResponse: ...


@runtime_checkable
class EvaluationMetric(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        prompt: str,
        response: ProviderResponse,
    ) -> MetricResult: ...


@runtime_checkable
class ResultStore(Protocol):
    def save_run(self, result: BenchmarkRunResult) -> None: ...


@dataclass(frozen=True, slots=True)
class ExactMatchMetric:
    name: str = "exact_match"

    def evaluate(
        self,
        *,
        case: BenchmarkCase,
        prompt: str,
        response: ProviderResponse,
    ) -> MetricResult:
        del prompt
        if case.expected_output is None:
            raise ValueError(
                "ExactMatchMetric requires each benchmark case to define expected_output"
            )
        matched = response.output == case.expected_output
        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details={"expected": case.expected_output, "actual": response.output},
        )


@dataclass(slots=True)
class JsonlResultStore:
    path: Path

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save_run(self, result: BenchmarkRunResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(result.to_dict(), sort_keys=True))
            file_handle.write("\n")

    def load_runs(self) -> list[JSONDict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as file_handle:
            return [json.loads(line) for line in file_handle if line.strip()]


@runtime_checkable
class HasName(Protocol):
    @property
    def name(self) -> str: ...


class Registry[T: HasName]:
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, item: T) -> T:
        if item.name in self._items:
            raise ValueError(f"Component {item.name!r} is already registered")
        self._items[item.name] = item
        return item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as error:
            raise KeyError(f"Unknown component {name!r}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def values(self) -> tuple[T, ...]:
        return tuple(self._items[name] for name in self.names())


class ProviderRegistry(Registry[ModelProvider]):
    pass


class TemplateRegistry(Registry[PromptTemplate]):
    pass


class DatasetRegistry(Registry[BenchmarkDataset]):
    pass


class MetricRegistry(Registry[EvaluationMetric]):
    pass


@dataclass(slots=True)
class BenchmarkRunner:
    metrics: tuple[EvaluationMetric, ...]
    store: ResultStore | None = None

    def __init__(
        self,
        *,
        metrics: list[EvaluationMetric] | tuple[EvaluationMetric, ...],
        store: ResultStore | None = None,
    ) -> None:
        self.metrics = tuple(metrics)
        self.store = store

    def run(
        self,
        *,
        dataset: BenchmarkDataset,
        provider: ModelProvider,
        template: PromptTemplate,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkRunResult:
        started_at = _utcnow()
        case_results: list[CaseResult] = []

        for case in dataset.cases:
            prompt = template.render(case.input_variables)
            response = provider.generate(prompt, case=case, template=template)
            metric_results = tuple(
                metric.evaluate(case=case, prompt=prompt, response=response)
                for metric in self.metrics
            )
            case_results.append(
                CaseResult(
                    case_id=case.case_id,
                    prompt=prompt,
                    expected_output=case.expected_output,
                    response=response,
                    metrics=metric_results,
                    case_metadata=case.metadata,
                ),
            )

        completed_at = _utcnow()
        result = BenchmarkRunResult(
            run_id=run_id or _default_run_id(started_at),
            dataset_name=dataset.name,
            provider_name=provider.name,
            template_name=template.name,
            started_at=started_at,
            completed_at=completed_at,
            case_results=tuple(case_results),
            summary_metrics=_aggregate_metrics(case_results),
            metadata=dict(metadata or {}),
        )
        if self.store is not None:
            self.store.save_run(result)
        return result


def _default_run_id(started_at: datetime) -> str:
    return started_at.strftime("%Y%m%dT%H%M%S%fZ")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aggregate_metrics(case_results: tuple[CaseResult, ...] | list[CaseResult]) -> dict[str, float]:
    scores_by_name: dict[str, list[float]] = defaultdict(list)
    for case_result in case_results:
        for metric in case_result.metrics:
            scores_by_name[metric.name].append(metric.score)
    return {
        name: sum(scores) / len(scores) for name, scores in sorted(scores_by_name.items()) if scores
    }
