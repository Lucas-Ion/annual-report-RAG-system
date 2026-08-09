"""What to pull out of every report, and how to go looking for it.

Adding a datapoint is an entry in FIELDS and nothing else. No schema change,
no new table, no code: extracted_facts stores a key and a value rather than a
column per field precisely so this file can be the only thing that changes.

The instructions below are unusually specific, and every specific in them
comes from reading the five parsed reports rather than from imagining how an
annual report is written. Those findings are recorded next to the fields they
shaped, because a future reader deleting an odd looking sentence would quietly
break extraction on a report they have not opened.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Field:
    """One datapoint to extract from every report.

    Attributes:
        key: Stored in extracted_facts.field_key. Never displayed.
        label: What the interface calls it.
        instruction: Handed to the model verbatim. Says what counts as an
            answer and, more usefully, what does not.
        queries: Phrasings used to retrieve candidate chunks. Several, because
            reports state the same fact as a sentence in one place and an
            unlabelled table row in another, and no single phrasing finds both.
        multiple: Whether a report can legitimately have more than one value.
            Drives the prompt and how the interface renders the result.
        unit_hint: Suggested unit, when there is an obvious one.
        max_tokens: Ceiling on the model's reply for this field.

            Worth setting per field rather than globally, because the sizes are
            not comparable. A headcount is one short object. Heineken's
            sustainability targets came back as eleven, each carrying a
            verbatim quotation, and a ceiling that comfortably fits the first
            truncates the second in the middle of a JSON string. The result is
            not a short answer, it is an unparseable one, and the whole
            document's extraction fails.
    """

    key: str
    label: str
    instruction: str
    queries: tuple[str, ...]
    multiple: bool
    unit_hint: str | None = None
    max_tokens: int = 2048


# Headcount.
#
# Three things about this corpus make it much harder than it sounds.
#
# Every report leads with a rounded marketing figure and buries the real one.
# ABN AMRO says "More than 20,000 employees" on page 9. Heineken says "more
# than 80,000" on page 31 and 87,870 on pages 8 and 96. An extractor that takes
# the first plausible number is wrong on most of the set.
#
# Shell never uses the word FTE at all, not once in 462 pages, and states
# headcount instead. Defining this field by the letters F, T and E finds
# nothing for one report in five.
#
# Reports state many other headcounts nearby. Heineken alone gives regional
# breakdowns, a Netherlands-only figure, and a subsidiary with nine people, all
# within a page or two of the group total.
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


# Sustainability goals.
#
# Deliberately targets, not achievements. A report contains far more prose
# about what a company did last year than about what it has committed to, and
# without that distinction the extractor returns a wall of progress updates.
#
# The dated, measurable ones are also the only ones worth showing next to each
# other across five companies, which is the point of putting them in a table.
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
    # Measured: Heineken produced eleven targets and ASML enough to overflow
    # 2048 tokens. Unused headroom costs nothing, since billing is on tokens
    # produced rather than tokens allowed.
    max_tokens=8192,
)


FIELDS: tuple[Field, ...] = (FTE, SUSTAINABILITY_GOAL)

# Keyed lookup for the interface and for extracting a single field by name.
BY_KEY: dict[str, Field] = {field.key: field for field in FIELDS}
