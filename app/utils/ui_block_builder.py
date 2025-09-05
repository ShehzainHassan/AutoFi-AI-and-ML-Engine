import simplejson as json

class UIBlockBuilder:
    @staticmethod
    def build(ui_type: str, data, answer: str) -> str:
        """
        Build HTML block for the given UI type, always showing the answer first.
        """

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

        elif ui_type == "CALCULATOR":
            return f"""
                {answer_html}
                <div class="card-grid">
                    <div class="card"><strong>Calculation Result:</strong> {answer}</div>
                </div>
            """

        elif ui_type == "CHART":
            return f"""
                {answer_html}
                <div class="chart-block">
                    <strong>Chart Data:</strong>
                    <pre>{json.dumps(data, indent=2)}</pre>
                </div>
            """

        else:
            return answer_html
