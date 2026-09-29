"""Form specification of the Streamlit client, derived from ``variable_mapping.yaml``.

Nothing about the variables is hardcoded in the pages: labels, units,
domains, defaults (median or mode of the training set) and the code
labels come from the mapping, and the highlighted variables from
``app.highlighted_variables`` (global importance of the champion).

The wizard has four steps. The ``health`` block of the mapping (diagnoses,
self-rated health, angina, sleep medication) is shown together with the
``lifestyle`` block in the third step so the interface keeps the four
steps of the specification: sociodemographic, anthropometry, behaviour
and health history, laboratory exams.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.data.mapping import RawVariable, VariableMapping

NOT_INFORMED = "Não informado"


@dataclass(frozen=True)
class FieldSpec:
    """One input widget."""

    name: str
    label: str
    kind: str  # "numeric" | "categorical"
    dtype: str  # "int" | "float"
    nullable: bool
    highlighted: bool
    minimum: float | None = None
    maximum: float | None = None
    step: float = 1.0
    default: float | None = None
    options: dict[int, str] = field(default_factory=dict)
    help: str | None = None

    @property
    def display_label(self) -> str:
        """Label with the highlight marker used by the pages."""
        return f"{self.label} ★" if self.highlighted else self.label


@dataclass(frozen=True)
class StepSpec:
    """One ``st.form`` of the wizard."""

    key: str
    title: str
    description: str
    fields: tuple[FieldSpec, ...]


_STEPS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("socio", "1. Sociodemográfico", "Idade, sexo, moradia e renda.", ("socio",)),
    ("anthro", "2. Antropometria", "Peso, altura e circunferência da cintura.", ("anthro",)),
    (
        "lifestyle",
        "3. Comportamento e histórico de saúde",
        "Alimentação, atividade física, tabagismo, diagnósticos prévios e percepção de saúde.",
        ("lifestyle", "health"),
    ),
    (
        "lab",
        "4. Exames laboratoriais (opcionais)",
        "Quando ausentes, a API imputa os exames pelo KNN treinado e sinaliza a imputação (P3).",
        ("lab",),
    ),
)


def _label(variable: RawVariable) -> str:
    return f"{variable.label_pt} ({variable.unit})" if variable.unit else variable.label_pt


def default_value(variable: RawVariable) -> float | None:
    """Median (numeric) or mode (categorical) of the training set; ``None`` when nullable.

    Nullable variables are answered only by a subset of respondents
    (income sources, past smoking, exercise details), so their default is
    "not informed" rather than a statistic of the answering minority.
    """
    if variable.nullable:
        return None
    observed = variable.observed
    value = observed.mode if variable.domain.type == "categorical" else observed.median
    if value is None:
        return None
    return float(value)


def field_spec(variable: RawVariable, highlighted: Sequence[str]) -> FieldSpec:
    """Build the widget specification of one raw variable."""
    name, label = variable.en, _label(variable)
    default, help_text = default_value(variable), f"PNS 2013, variável {variable.pns}"
    is_highlighted = variable.en in highlighted
    if variable.domain.type == "categorical":
        return FieldSpec(
            name=name,
            label=label,
            kind="categorical",
            dtype=variable.dtype,
            nullable=variable.nullable,
            highlighted=is_highlighted,
            default=default,
            options=dict(variable.value_labels_pt or {}),
            help=help_text,
        )
    return FieldSpec(
        name=name,
        label=label,
        kind="numeric",
        dtype=variable.dtype,
        nullable=variable.nullable,
        highlighted=is_highlighted,
        minimum=variable.domain.min,
        maximum=variable.domain.max,
        step=1.0 if variable.dtype == "int" else 0.1,
        default=default,
        help=help_text,
    )


def build_steps(mapping: VariableMapping, highlighted: Sequence[str]) -> list[StepSpec]:
    """Build the four wizard steps in order, each with its fields in mapping order."""
    by_block: dict[str, list[FieldSpec]] = {}
    for variable in mapping.raw_variables:
        by_block.setdefault(variable.block, []).append(field_spec(variable, highlighted))
    steps = [
        StepSpec(
            key=key,
            title=title,
            description=description,
            fields=tuple(spec for block in blocks for spec in by_block.get(block, [])),
        )
        for key, title, description, blocks in _STEPS
    ]
    covered = {spec.name for step in steps for spec in step.fields}
    missing = [v.en for v in mapping.raw_variables if v.en not in covered]
    if missing:
        raise ValueError(f"variables without a wizard step: {missing}")
    return steps


def initial_record(steps: Sequence[StepSpec]) -> dict[str, float | None]:
    """Record filled with the defaults of every field."""
    return {spec.name: spec.default for step in steps for spec in step.fields}


def option_labels(spec: FieldSpec) -> list[str]:
    """Selectbox entries: ``"code - label"``, plus ``NOT_INFORMED`` when nullable."""
    labels = [f"{code} - {label}" for code, label in spec.options.items()]
    return [NOT_INFORMED, *labels] if spec.nullable else labels


def option_to_code(choice: str) -> int | None:
    """Inverse of :func:`option_labels`."""
    if choice == NOT_INFORMED:
        return None
    return int(choice.split(" - ", 1)[0])


def label_of(spec: FieldSpec, value: float | None) -> str:
    """Human readable value for the review summary."""
    if value is None:
        return NOT_INFORMED
    if spec.kind == "categorical":
        code = int(value)
        return f"{code} - {spec.options.get(code, '?')}"
    return f"{int(value)}" if spec.dtype == "int" else f"{value:g}"
