import html
import orjson
import re
from typing import Optional, Union

class UIBlockBuilder:
    """
    Builds safe, styled HTML blocks for AI responses.
    Supports TEXT, TABLE, CARD_GRID, CALCULATOR, CHART.
    Converts simple Markdown formatting (**bold**, *italic*) to HTML.
    """

    @staticmethod
    def _escape_html(val: Union[str, int, float, None]) -> str:
        """Escape unsafe HTML characters."""
        if val is None:
            return ""
        return html.escape(str(val))

    @staticmethod
    def _markdown_to_html(text: str) -> str:
        """Convert basic Markdown formatting to HTML tags."""
        if not text:
            return ""
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
        return text

    @staticmethod
    def build(ui_type: str, data, answer: str, chart_type: Optional[str] = None) -> str:
        """
        Build HTML UI blocks for AI responses.
        Preserves Markdown formatting in the answer.
        """
        styled_answer = UIBlockBuilder._markdown_to_html(answer)
        answer_html = f"<div class='ai-answer'><p>{styled_answer}</p></div>"

        if ui_type == "TEXT":
            return answer_html

        if ui_type == "TABLE" and isinstance(data, list) and data:
            headers = list(data[0].keys())
            thead = "".join([f"<th>{UIBlockBuilder._escape_html(h)}</th>" for h in headers])
            rows = "".join([
                "<tr>" + "".join([
                    f"<td>{UIBlockBuilder._escape_html(row.get(h, ''))}</td>" for h in headers
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

        if ui_type == "CARD_GRID" and isinstance(data, list) and data:
            cards = "".join([
                "<div class='card'>" +
                "".join([
                    f"<p><strong>{UIBlockBuilder._escape_html(k)}:</strong> {UIBlockBuilder._escape_html(v)}</p>"
                    for k, v in row.items()
                ]) +
                "</div>"
                for row in data
            ])
            return f"{answer_html}<div class='card-grid'>{cards}</div>"

        if ui_type == "CALCULATOR" and isinstance(data, dict):
            calc_html = "".join([
                f"<p><strong>{UIBlockBuilder._escape_html(k.replace('_', ' ').title())}:</strong> {UIBlockBuilder._escape_html(v)}</p>"
                for k, v in data.items()
            ])
            return (
                f"{answer_html}"
                f"<div class='card-grid'><div class='card'>{calc_html}</div></div>"
            )

        if ui_type == "CHART":
            chart_type = chart_type or "bar"
            if isinstance(data, dict) and "data" in data:
                chart_type = data.get("chart_type", chart_type)
                data = data.get("data", [])

            if chart_type not in ("bar", "line", "pie"):
                chart_type = "bar"

            chart_data_json = orjson.dumps(data, default=str)
            chart_data_attr = html.escape(chart_data_json)

            return (
                f"{answer_html}"
                f"<div class='chart-block' "
                f"data-chart-type='{chart_type}' "
                f"data-chart='{chart_data_attr}'></div>"
            )

        return answer_html
