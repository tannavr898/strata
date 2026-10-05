"""status.py — summarising a field's soil into something glanceable, honestly.

WHY THIS IS NOT A SCORE
-----------------------
The obvious thing to put on a dashboard is "Soil health: 72/100". This module
deliberately does not produce one, and the reason is the project's own history: a
circular IQR-based scorer existed here once and was deleted, and the README states
plainly that Strata does not invent composite targets. A 0-100 number requires
weights across pH, P, K, organic matter and biology that no Penn State publication
supplies. Inventing them would produce a confident figure a farmer might fertilise
against, built on arithmetic nobody can audit.

So a field's soil is summarised as a STATUS: a roll-up of findings that each trace
to a citation already in `recommender.CITATIONS`. Nothing is combined that the
agronomy does not combine; the summary only counts and ranks what the cited rules
already concluded.

WHAT CAN AND CANNOT BE IN IT
----------------------------
The rule engine covers exactly three measures -- pH (lime), Mehlich-3 P, and
Mehlich-3 K. Those are the only things in the status.

CEC and organic matter are carried through the whole pipeline and shown to the
user, but `recommender` has NO citation for either, so they appear as context
labelled "no rule" rather than as part of the judgement. Organic matter is the
classic soil-health indicator, which makes its absence worth stating out loud
rather than papering over: adding a cited OM interpretation is a research task,
and until someone does it, OM cannot carry weight here.

Soil CO2 is the project's differentiator and likewise has no AASL threshold. It is
a trend shown to the farmer, never a component of this status.

"NOT MEASURED" IS NOT "FINE"
----------------------------
The state that matters most is `unmeasured`. A field with no soil test is not a
healthy field -- it is an unknown one, and a green checkmark there would be the
most dishonest pixel in the app. `unknown` findings are counted separately from
good ones and never average away against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

#: Worst-first ranking used to pick a field's headline state.
#:
#: `environmental` outranks `deficiency` because excessive phosphorus is a runoff
#: and P-Index concern -- it can carry a regulatory obligation and affects the
#: Chesapeake watershed, not just this season's yield. `maintenance` is a normal,
#: healthy finding (replace what the harvest removed), so it ranks below `ok` only
#: in the sense that it implies an action.
_SEVERITY_ORDER = ["environmental", "deficiency", "maintenance", "ok"]

#: How each severity reads to a farmer, and which honesty colour it takes.
_SEVERITY_LABEL = {
    "environmental": ("excessive", "env"),
    "deficiency": ("below optimum", "low"),
    "maintenance": ("optimum", "mid"),
    "ok": ("sufficient", "good"),
    "unknown": ("not measured", "unknown"),
}

#: Short names for the three ruled measures, for a chip row read at a glance.
_SHORT = {
    "Lime": "pH",
    "Phosphate (P2O5)": "P",
    "Potash (K2O)": "K",
}


@dataclass(frozen=True)
class Chip:
    """One cited finding, compressed to a glanceable token."""
    key: str            # "pH" | "P" | "K"
    category: str       # the recommender's own category name
    value: str          # the measured value, or "—" when absent
    severity: str
    label: str          # "below optimum", "not measured", ...
    tone: str           # low | mid | good | env | unknown
    citation: str       # the source publication, so the chip stays auditable


@dataclass(frozen=True)
class SoilStatus:
    """A field's soil, summarised from cited findings only.

    `state` is the single word a dashboard card leads with:
        unmeasured  no soil test on file; nothing can be assessed
        attention   at least one cited finding needs action
        partial     what was measured is fine, but something is unmeasured
        ok          every ruled measure is at or above optimum
    """
    state: str
    headline: str
    chips: list[Chip] = dc_field(default_factory=list)
    n_ruled: int = 0
    n_action: int = 0
    n_unknown: int = 0
    context: list[tuple[str, str]] = dc_field(default_factory=list)  # (label, value)

    @property
    def needs_attention(self) -> bool:
        return self.state in ("attention", "unmeasured")


def _context_rows(lab) -> list[tuple[str, str]]:
    """CEC and organic matter: shown, never judged.

    Both are real measurements the farmer paid for, so hiding them would be
    unhelpful. But `recommender` has no citation for either, so they are returned
    as plain context for the template to label "no rule" -- visibly outside the
    judgement rather than silently folded into it.
    """
    rows: list[tuple[str, str]] = []
    if lab is None:
        return rows
    if lab["cec"] is not None:
        rows.append(("CEC", f"{lab['cec']:.1f}"))
    if lab["om_pct"] is not None:
        rows.append(("Organic matter", f"{lab['om_pct']:.1f}%"))
    return rows


def soil_status(lab, recommendations: list) -> SoilStatus:
    """Summarise one field from its latest lab test and the cited findings on it.

    `recommendations` are `recommender.Recommendation` objects -- already the
    output of the audited rule engine. This function only counts and ranks them;
    it never re-interprets a measurement.
    """
    if lab is None:
        return SoilStatus(
            state="unmeasured",
            headline="No soil test on file",
            n_unknown=3,
        )

    chips: list[Chip] = []
    for rec in recommendations:
        key = _SHORT.get(rec.category)
        if key is None:          # a rule outside the three ruled measures
            continue
        label, tone = _SEVERITY_LABEL.get(rec.severity, (rec.severity, "unknown"))
        # `measured` reads like "Mehlich-3 P 22 ppm" or "soil pH 5.72, buffer pH
        # 6.50"; the leading clause is the number a chip should show.
        value = rec.measured.split(",")[0]
        for prefix in ("Mehlich-3 P ", "Mehlich-3 K ", "soil pH "):
            value = value.replace(prefix, "")
        chips.append(Chip(
            key=key, category=rec.category,
            value="—" if rec.severity == "unknown" else value,
            severity=rec.severity, label=label, tone=tone,
            citation=rec.citation.source,
        ))

    n_unknown = sum(1 for c in chips if c.severity == "unknown")
    actionable = [c for c in chips if c.severity in ("environmental", "deficiency")]
    n_ruled = len(chips) - n_unknown

    if n_ruled == 0:
        return SoilStatus(
            state="unmeasured",
            headline="Nothing measurable on file",
            chips=chips, n_unknown=n_unknown,
            context=_context_rows(lab),
        )

    if actionable:
        worst = min(actionable, key=lambda c: _SEVERITY_ORDER.index(c.severity))
        names = ", ".join(c.key for c in actionable)
        headline = (
            f"{names} below Penn State optimum"
            if worst.severity == "deficiency"
            else f"{names} excessive — runoff concern"
        )
        state = "attention"
    elif n_unknown:
        headline = f"{n_ruled} of 3 measured, all at or above optimum"
        state = "partial"
    else:
        headline = "All three measures at or above optimum"
        state = "ok"

    return SoilStatus(
        state=state, headline=headline, chips=chips,
        n_ruled=n_ruled, n_action=len(actionable), n_unknown=n_unknown,
        context=_context_rows(lab),
    )


# --------------------------------------------------------------------------- #
# The attention list
# --------------------------------------------------------------------------- #
def _join(items: list[str]) -> str:
    """"pH, P and K" — an Oxford-free list that reads as prose in a headline."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


@dataclass(frozen=True)
class Alert:
    """One thing worth a farmer's attention, with somewhere to go about it."""
    tone: str            # low | env | unknown | slate
    title: str
    detail: str
    href: Optional[str] = None
    action: Optional[str] = None


def alerts_for(field, status: SoilStatus, prediction: dict,
               uncalibrated: int = 0) -> list[Alert]:
    """What this field needs, worst first.

    Deliberately includes "we cannot tell you" items alongside "you should act"
    items. A dashboard that only listed actions would quietly imply that an
    unmeasured field is a fine one.
    """
    out: list[Alert] = []
    base = f"/field/{field['id']}"

    if status.state == "unmeasured":
        out.append(Alert(
            tone="unknown",
            title=f"{field['name']} has no soil test",
            detail="Without Mehlich-3 P, K and buffer pH, Strata cannot recommend "
                   "anything for this field.",
            href=f"{base}/soil-test", action="Add a soil test",
        ))
    else:
        # Grouped per field, not one alert per measure. Three near-identical
        # "below optimum" rows for the same field read as three problems when it
        # is one soil test with three findings, and the repetition buries the
        # genuinely different items (an unmeasured field, a runoff risk) further
        # down the list.
        excessive = [c for c in status.chips if c.severity == "environmental"]
        low = [c for c in status.chips if c.severity == "deficiency"]
        missing = [c for c in status.chips if c.severity == "unknown"]

        # Excess phosphorus stays its own alert: it is a runoff and P-Index
        # concern with the opposite instruction (apply none), so folding it in
        # with shortfalls would blur two opposite actions.
        for chip in excessive:
            out.append(Alert(
                tone="env",
                title=f"{field['name']}: {chip.key} excessive at {chip.value}",
                detail="Phosphorus loss and P-Index concern. Apply none; a Penn "
                       "State P-Index assessment may be required before manure.",
                href=base, action="See the recommendation",
            ))

        if low:
            names = _join([c.key for c in low])
            values = ", ".join(f"{c.key} {c.value}" for c in low)
            needs_lime = any(c.key == "pH" for c in low)
            detail = f"Measured {values}."
            if needs_lime:
                detail += (" Soil is more acidic than this crop's target; the lime "
                           "rate needs a buffer pH reading.")
            if any(c.key != "pH" for c in low):
                detail += " Yield response to the shortfall is likely."
            out.append(Alert(
                tone="low",
                title=f"{field['name']}: {names} below Penn State optimum",
                detail=detail,
                href=base, action="See the recommendations",
            ))

        if missing:
            names = _join([c.key for c in missing])
            out.append(Alert(
                tone="unknown",
                title=f"{field['name']}: {names} not measured",
                detail="These cannot be interpreted without a laboratory soil test, "
                       "so no rate is offered for them.",
                href=f"{base}/soil-test", action="Add them",
            ))

    if prediction.get("state") == "withheld":
        out.append(Alert(
            tone="slate",
            title=f"{field['name']}: no yield estimate",
            detail="Tested on counties it had never seen, the model was not reliably "
                   "better than quoting the county average, so Strata withholds it.",
            href="/models", action="Why",
        ))

    if uncalibrated:
        out.append(Alert(
            tone="slate",
            title=f"{field['name']}: {uncalibrated} uncalibrated readings",
            detail="Raw sensor values are a trend over time, but are not "
                   "lab-comparable until a calibration is fitted.",
            href="/calibration", action="Calibrate",
        ))

    return out
