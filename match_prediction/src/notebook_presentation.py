"""Presentation-only notebook views; never modify model or export objects."""

from dataclasses import dataclass
from contextlib import contextmanager
from html import escape
import json
import re

import pandas as pd


@dataclass
class NotebookHTML:
    html: str
    label: str = "Notebook table (HTML view)"

    def _repr_html_(self):
        return self.html

    def __repr__(self):
        return self.label


def full_table_html(value):
    """Render a DataFrame or Styler completely, including long literal cells.

    Explicit local Styler limits are essential: display.max_rows alone has no
    effect on Styler. Global defaults are restored on leaving this function.
    """
    frame = value if isinstance(value, pd.DataFrame) else value.data
    rows, columns = frame.shape
    with pd.option_context(
        "display.max_colwidth", None,
        "styler.render.max_rows", max(1, rows),
        "styler.render.max_columns", max(1, columns),
        "styler.render.max_elements", max(262144, (rows + 1) * (columns + 1)),
    ):
        if isinstance(value, pd.DataFrame):
            classes = ["nb-presentation-table"]
            rules = [
                ".nb-presentation-table th,.nb-presentation-table td{text-align:left;vertical-align:top;padding:.35em .6em}",
                ".nb-presentation-table td{white-space:normal;overflow-wrap:normal;word-break:normal}",
            ]
            for position, column in enumerate(frame.columns, 1):
                if column in {"feature", "input", "structural", "market"}:
                    name = f"nb-nowrap-{position}"
                    classes.append(name)
                    rules.append(f".{name} td:nth-child({position}){{white-space:nowrap}}")
            return ("<style>" + "".join(rules) + "</style>"
                    + value.to_html(index=False, max_rows=None, max_cols=None,
                                    escape=True, classes=classes))
        return value.to_html(max_rows=max(1, rows), max_columns=max(1, columns))


def details_html(html, title):
    # No `open` attribute: saved notebook output starts collapsed.
    return f'<details><summary>{escape(title)}</summary><div style="overflow-x:auto">{html}</div></details>'


def markdown_fragment(text):
    """Render the headings and emphasis used in captured league-table captions."""
    blocks = []
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        tag = f"h{len(heading[1])}" if heading else "p"
        content = escape(heading[2] if heading else line)
        content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
        blocks.append(f"<{tag}>{content}</{tag}>")
    return "".join(blocks)


@contextmanager
def collapsed_output(title):
    """Collect display-only league sections without changing their data."""
    from IPython.display import display
    from IPython.utils.capture import capture_output
    with capture_output() as captured:
        yield
    html = []
    for output in captured.outputs:
        data = output.data
        if "text/html" in data:
            html.append(data["text/html"])
        elif "text/markdown" in data:
            html.append(markdown_fragment(data["text/markdown"]))
        elif "text/plain" in data:
            html.append("<pre>" + escape(data["text/plain"]) + "</pre>")
    for stream in [captured.stdout, captured.stderr]:
        if stream:
            html.append("<pre>" + escape(stream) + "</pre>")
    display(NotebookHTML(details_html("".join(html), title), title))


def table_view(value, title="Table", *, detail=False):
    frame = value if isinstance(value, pd.DataFrame) else value.data
    label = f"{title} ({len(frame):,} rows; complete)"
    html = full_table_html(value)
    if detail:
        html = details_html(html, label)
    else:
        html = f'<p><strong>{escape(label)}</strong></p><div style="overflow-x:auto">{html}</div>'
    return NotebookHTML(html, label)


def reader_frame(frame):
    """Format only a display copy; accept already formatted saved output too."""
    result = frame.copy()
    for column in result:
        if not pd.api.types.is_numeric_dtype(result[column]):
            continue
        if column in {"accuracy", "balanced_accuracy", "best_accuracy", "macro_f1", "draw_recall", "home_win_recall", "mean_stability"}:
            result[column] = result[column].map(lambda value: f"{value:.2%}")
        elif column in {"log_loss", "brier_score"}:
            result[column] = result[column].map(lambda value: f"{value:.4f}")
    return result


def compact_with_detail(value, columns, title="Summary", detail_title="Technical detail: complete table"):
    """Keep a narrow reader view and the entire original table in one output."""
    frame = value if isinstance(value, pd.DataFrame) else value.data
    visible = reader_frame(frame.loc[:, columns])
    return NotebookHTML(
        table_view(visible, title).html + table_view(value, detail_title, detail=True).html,
        f"{title}; {len(frame):,} complete records in technical detail",
    )


def grouped_table_view(value, group, title="Comparison", columns=None):
    """Small complete groups above an optional full-metric detail."""
    frame = value if isinstance(value, pd.DataFrame) else value.data
    html = []
    for key, part in frame.groupby(group, sort=False, dropna=False):
        selected = part if columns is None else part.loc[:, columns]
        html.append(table_view(reader_frame(selected), f"{title}: {key}").html)
    if columns is not None:
        html.append(table_view(value, "Technical detail: all comparison metrics", detail=True).html)
    return NotebookHTML("".join(html), f"{title}; {len(frame):,} records grouped by {group}")


def selection_stage_counts(audit):
    rows = []
    for (scope, branch), part in audit.groupby(["scope", "branch"], sort=True):
        quality = part["screen_reason"].eq("retained")
        correlation = part["correlation_reason"].eq("retained")
        core = part["mandatory"].eq(True) & quality
        accepted = part["shadow_status"].isin(["confirmed", "tentative"])
        fallback = part["fallback_after_shadow"].eq(True)
        rows.append({
            "scope": scope, "branch": branch, "candidates": len(part),
            "after_quality": int(quality.sum()),
            "available_core": int(core.sum()),
            "after_Spearman": int(correlation.sum()),
            "shadow_accepted_supplement": int(accepted.sum()),
            "fallback_added": int(fallback.sum()),
            "RFE_supplement_pool": int((accepted | fallback).sum()),
            "final_inputs": int(part["final_selected"].eq(True).sum()),
        })
    return pd.DataFrame(rows)


def feature_meaning(feature):
    """Explain the existing naming grammar; this never selects a feature."""
    exact = {
        "elo_diff_pre": "Home minus away internal Elo before the match",
        "clubelo_diff": "Home minus away dated ClubElo rating",
        "clubelo_available": "Indicator that both teams have a dated ClubElo rating",
        "sofifa_available": "Indicator that both teams have a dated SoFIFA squad rating",
        "market_prob_home": "Normalised inverse odds: home-win probability",
        "market_prob_draw": "Normalised inverse odds: draw probability",
        "market_prob_away": "Normalised inverse odds: away-win probability",
        "market_overround": "Sum of the three inverse odds; bookmaker margin is this sum minus one",
        "market_available": "Indicator that a valid pre-match odds row is available",
        "league_id": "League identity; one categorical input, later one-hot encoded",
    }
    if feature in exact:
        return exact[feature]
    side = next((prefix for prefix in ["home_", "away_", "diff_"] if feature.startswith(prefix)), "")
    prefix = {"home_": "Home team: ", "away_": "Away team: ", "diff_": "Home minus away: ", "": ""}[side]
    body = feature[len(side):]
    window = ""
    patterns = [
        (r"_venue_mean_last_(\d+)$", lambda n: f"mean over up to {n} earlier matches in the same venue role"),
        (r"_mean_last_(\d+)$", lambda n: f"mean over up to {n} earlier matches in the current top-flight spell"),
        (r"_ewm_span_(\d+)$", lambda n: f"exponentially weighted mean, span {n}, through the previous match"),
        (r"_season_mean_pre$", lambda n: "current-season mean through the previous match"),
    ]
    for pattern, description in patterns:
        match = re.search(pattern, body)
        if match:
            window = description(match.group(1) if match.lastindex else None)
            body = body[:match.start()]
            break
    concepts = {
        "elo_pre": "internal Elo before the match", "clubelo_rating": "dated ClubElo rating",
        "points": "league points per match", "goal_diff": "goals scored minus conceded",
        "goals_for": "goals scored", "goals_against": "goals conceded",
        "xg_for": "expected goals created", "xg_against": "expected goals conceded",
        "xg_diff": "expected goals created minus conceded",
        "np_xg_diff": "non-penalty expected goals created minus conceded",
        "expected_points": "Understat expected points",
        "ppda": "opponent passes per defensive action (pressing measure)",
        "deep_completions": "completed passes close to the opponent goal, excluding crosses",
        "rest_days": "days since the previous recorded match",
        "promoted_or_returning": "new or returning top-flight spell indicator",
        "sofifa_top11_overall": "mean overall rating of the top 11 players in the dated squad snapshot",
        "sofifa_top3_att": "mean rating of the top three attackers in the dated squad snapshot",
        "sofifa_best_gk": "best goalkeeper rating in the dated squad snapshot",
    }
    meaning = concepts.get(body, body.replace("_", " "))
    return prefix + meaning + (f"; {window}" if window else "")


def registry_overview(multiclass, binary):
    from src.final_model_registry import registry_frame
    html = []
    for target, registry in [("1X2", multiclass), ("Binary home win", binary)]:
        rows = registry_frame(registry, target)[["scope", "branch", "algorithm", "features"]]
        rows = rows.rename(columns={"features": "original_inputs"})
        html.append(table_view(rows, f"{target}: locked models").html)
    return NotebookHTML("".join(html), "24 locked models: 12 per target")


def registry_feature_guide(multiclass, binary, notebooks_dir):
    """Derive membership and core roles from the actual preserved registries."""
    from src.final_model_registry import SCOPE_CODES, _find_table, _notebook_tables
    role_tables = [
        _find_table(_notebook_tables(notebooks_dir / "03_multiclass_models.ipynb"),
                    {"scope", "branch", "position", "feature", "selection_role"}, minimum_rows=100),
        _find_table(_notebook_tables(notebooks_dir / "05_binary_models.ipynb"),
                    {"scope", "branch", "position", "input", "selection_role"}, minimum_rows=100),
    ]
    entries = []
    for (target, registry), roles, name in zip(
        [("1X2", multiclass), ("Binary home win", binary)], role_tables, ["feature", "input"]
    ):
        for (scope, branch), entry in sorted(registry.items()):
            scope_name = next(label for label, code in SCOPE_CODES.items() if code == scope)
            part = roles.loc[roles["scope"].eq(scope_name) & roles["branch"].eq(branch)].sort_values("position")
            assert part[name].tolist() == entry["features"], "Role table and locked feature order differ"
            core = set(part.loc[part["selection_role"].eq("mandatory core"), name])
            entries.append((target, scope, scope_name, branch, entry, core))
    common = set.intersection(*(core for *_, core in entries))
    common_order = [f for f in entries[0][4]["features"] if f in common]
    html = [table_view(pd.DataFrame([
        {"feature": f, "meaning": feature_meaning(f)} for f in common_order
    ]), "Structural core retained in every locked model").html]
    html.append('<p>This intersection describes the retained mandatory roles in the saved registries. '
                'It does not create or alter any selected list. Branch-specific retained core inputs, '
                'if any, are marked below.</p>')
    special = sorted({f for *_, entry, core in entries for f in entry["features"]
                      if f.startswith("market_") or f == "league_id"})
    html.append(table_view(pd.DataFrame([
        {"feature": f, "meaning": feature_meaning(f),
         "present_in_models": sum(f in item[4]["features"] for item in entries)} for f in special
    ]), "Market inputs and pooled league identity").html)
    for target, registry in [("1X2", multiclass), ("Binary home win", binary)]:
        html.append(f'<h3>{escape(target)}: supplementary inputs by league and branch</h3>')
        for scope in sorted({s for s, b in registry}):
            pair = [item for item in entries if item[0] == target and item[1] == scope]
            supplements = sorted({f for item in pair for f in item[4]["features"] if f not in common and f not in special})
            rows = []
            for f in supplements:
                row = {"feature": f, "meaning": feature_meaning(f)}
                for _, _, _, branch, entry, core in pair:
                    row[branch] = ("retained core" if f in core else "supplement") if f in entry["features"] else "—"
                rows.append(row)
            html.append(table_view(pd.DataFrame(rows), pair[0][2]).html)
        exact_parts = []
        for item in [item for item in entries if item[0] == target]:
            _, _, scope_name, branch, entry, core = item
            rows = pd.DataFrame([
                {"position": i, "feature": f, "meaning": feature_meaning(f),
                 "role": "retained core" if f in core else "supplement"}
                for i, f in enumerate(entry["features"], 1)
            ])
            params = entry.get("components", entry.get("parameters"))
            exact_parts.append(f'<h4>{escape(scope_name)} / {escape(branch)}: {escape(entry["model_label"])}</h4>'
                               + '<pre style="white-space:pre-wrap">' + escape(json.dumps(params, ensure_ascii=False, indent=2)) + '</pre>'
                               + full_table_html(rows))
        html.append(details_html("".join(exact_parts), f"Technical detail: {target}, complete ordered inputs and hyperparameters"))
    return NotebookHTML("".join(html), "Feature guide derived from the 24 locked registries; full order in technical details")
