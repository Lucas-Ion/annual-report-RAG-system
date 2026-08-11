"""What to pull out of every report"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Field:
    key: str
    label: str
    instruction: str
    queries: tuple[str, ...]
    multiple: bool
    unit_hint: str | None = None
    max_tokens: int = 2048


FTE = Field(
    key="fte",
    label="Employees (FTE)",
    instruction=(
        "The total number of people the group employed, for the reporting "
        "year covered by this report.\n\n"
        "Take the group total. Ignore figures for a single region, country, "
        "segment or subsidiary, and ignore counts of a subset such as R&D "
        "staff, contractors or new hires.\n\n"
        "Prefer an exact figure over a rounded one. Reports usually state "
        "something like 'more than 80,000 employees' in an introduction and "
        "the precise number in the financial statements or a personnel note. "
        "The precise number is the answer. Only fall back to a rounded figure "
        "if no exact one appears in the excerpts.\n\n"
        "Where a table gives a total and then deducts a category from it, "
        "take the total. ASML reports 'Total 44,209', then 'Less: Temporary "
        "employees 689', then 'Payroll employees 43,520'. The answer is "
        "44,209, because that is what the company itself totals and what it "
        "quotes publicly.\n\n"
        "Companies report this differently and all of these count: 'average "
        "number of employees (FTE)', 'full-time equivalent employees', "
        "'total employees', 'headcount at year end'. Record which basis was "
        "used in the unit field, for example 'FTE' or 'headcount', because "
        "the two are not the same measure and should not be compared as if "
        "they were."
    ),
    queries=(
        "average number of employees full-time equivalent FTE",
        "total number of employees at year end headcount",
        "number of FTEs as at 31 December",
        "our workforce total people employed by the group",
    ),
    multiple=False,
    unit_hint="FTE",
)

SUSTAINABILITY_GOAL = Field(
    key="sustainability_goal",
    label="Sustainability goals",
    instruction=(
        "A specific sustainability target the company has committed to, "
        "stated in this report.\n\n"
        "Take forward looking commitments, not past achievements. 'We reduced "
        "emissions by 12% in 2025' is a result and does not belong here. 'We "
        "aim to reach net zero by 2050' is a target and does.\n\n"
        "Prefer targets that are measurable and dated: a percentage, an "
        "absolute figure or a clear end state, together with the year it is "
        "to be reached by. Skip vague statements of ambition with no number "
        "and no date.\n\n"
        "Record each distinct target separately rather than summarising "
        "several into one. Where a target has a number, put it in the numeric "
        "field and give its unit, for example '%', 'tCO2e' or 'MWh'. Where it "
        "has a target year, mention that year in the raw value so it is "
        "visible without opening the source."
    ),
    queries=(
        "sustainability targets net zero emissions reduction by 2030 2050",
        "climate targets scope 1 2 3 greenhouse gas emission reduction target",
        "we aim to achieve our commitment target renewable energy",
        "science based targets decarbonisation ambition",
    ),
    multiple=True,
    max_tokens=8192,
)


FIELDS: tuple[Field, ...] = (FTE, SUSTAINABILITY_GOAL)

# This is a keyed lookup for the interface and for extracting a single field by name.
BY_KEY: dict[str, Field] = {field.key: field for field in FIELDS}
