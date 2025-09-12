import html
import simplejson as json
from typing import Optional
class UIBlockBuilder:
    @staticmethod
    def build(ui_type: str, data, answer: str, chart_type: Optional[str] = None) -> str:
        answer_html = f"<p>{answer}</p>"

        if ui_type == "TEXT":
            return answer_html

        elif ui_type == "TABLE" and isinstance(data, list) and len(data) > 0:
            headers = data[0].keys()
            thead = "".join([f"<th>{h}</th>" for h in headers])
            rows = "".join([
                "<tr>" + "".join([f"<td>{row.get(h, '')}</td>" for h in headers]) + "</tr>"
                for row in data
            ])
            return f"""
                {answer_html}
                <div class="table-wrapper">
                    <table class="ai-table">
                        <thead><tr>{thead}</tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
            """

        elif ui_type == "CARD_GRID" and isinstance(data, list) and len(data) > 0:
            cards = "".join([
                "<div class='card'>" +
                "".join([f"<p><strong>{k}:</strong> {v}</p>" for k, v in row.items()]) +
                "</div>"
                for row in data
            ])
            return f"""
                {answer_html}
                <div class='card-grid'>{cards}</div>
            """

        elif ui_type == "CALCULATOR" and isinstance(data, dict):
            calc_html = "".join([
                f"<p><strong>{k.replace('_', ' ').title()}:</strong> {v}</p>" for k, v in data.items()
            ])
            return f"""
                {answer_html}
                <div class="card-grid">
                    <div class="card">{calc_html}</div>
                </div>
            """

        elif ui_type == "CHART":
            chart_type = chart_type or "bar"

            if isinstance(data, dict) and "data" in data:
                chart_type = data.get("chart_type", chart_type)
                data = data.get("data", [])

            if chart_type not in ("bar", "line", "pie"):
                chart_type = "bar"

            chart_data_json = html.escape(json.dumps(data, default=str))

            return f"""
                {answer_html}
                <div class="chart-block" data-chart-type="{chart_type}" data-chart="{chart_data_json}"></div>
            """

        else:
            return answer_html