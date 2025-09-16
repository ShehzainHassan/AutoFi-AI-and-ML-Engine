import html
import simplejson as json
from typing import Optional, Union


class UIBlockBuilder:
    """
    Builds safe, escaped HTML blocks for AI responses.
    Supports TEXT, TABLE, CARD_GRID, CALCULATOR, CHART.
    """

    @staticmethod
    def _safe(val: Union[str, int, float, None]) -> str:
        """Escape values safely for HTML rendering."""
        if val is None:
            return ""
        return html.escape(str(val))

    @staticmethod
    def build(ui_type: str, data, answer: str, chart_type: Optional[str] = None) -> str:
        """
        Build HTML UI blocks for AI responses.
        Always escapes text to prevent XSS.
        """
        escaped_answer = UIBlockBuilder._safe(answer)
        answer_html = f"<p>{escaped_answer}</p>"

        if ui_type == "TEXT":
            return answer_html

        if ui_type == "TABLE" and isinstance(data, list) and len(data) > 0:
            headers = list(data[0].keys())
            thead = "".join([f"<th>{UIBlockBuilder._safe(h)}</th>" for h in headers])
            rows = "".join([
                "<tr>" + "".join([
                    f"<td>{UIBlockBuilder._safe(row.get(h, ''))}</td>" for h in headers
                ]) + "</tr>"
                for row in data
            ])
            return (
                f"{answer_html}"
                f"<div class='table-wrapper'>"
                f"<table class='ai-table'>"
                f"<thead><tr>{thead}</tr></thead>"
                f"<tbody>{rows}</tbody>"
                f"</table>"
                f"</div>"
            )

        if ui_type == "CARD_GRID" and isinstance(data, list) and len(data) > 0:
            cards = "".join([
                "<div class='card'>" +
                "".join([
                    f"<p><strong>{UIBlockBuilder._safe(k)}:</strong> {UIBlockBuilder._safe(v)}</p>"
                    for k, v in row.items()
                ]) +
                "</div>"
                for row in data
            ])
            return f"{answer_html}<div class='card-grid'>{cards}</div>"

        if ui_type == "CALCULATOR" and isinstance(data, dict):
            calc_html = "".join([
                f"<p><strong>{UIBlockBuilder._safe(k.replace('_', ' ').title())}:</strong> {UIBlockBuilder._safe(v)}</p>"
                for k, v in data.items()
            ])
            return (
                f"{answer_html}"
                f"<div class='card-grid'>"
                f"<div class='card'>{calc_html}</div>"
                f"</div>"
            )

        if ui_type == "CHART":
            chart_type = chart_type or "bar"

            if isinstance(data, dict) and "data" in data:
                chart_type = data.get("chart_type", chart_type)
                data = data.get("data", [])

            if chart_type not in ("bar", "line", "pie"):
                chart_type = "bar"

            chart_data_json = json.dumps(data, default=str)
            chart_data_attr = html.escape(chart_data_json)

            return (
                f"{answer_html}"
                f"<div class='chart-block' "
                f"data-chart-type='{chart_type}' "
                f"data-chart='{chart_data_attr}'></div>"
            )

        return answer_html
