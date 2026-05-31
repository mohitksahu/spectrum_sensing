"""Minimal HTML report generator for SpectraSense experiments."""

from jinja2 import Template
from pathlib import Path
from typing import Dict

HTML_TEMPLATE = """
<html>
<head><title>SpectraSense Report</title></head>
<body>
<h1>{{ title }}</h1>
<ul>
{% for k,v in metrics.items() %}
  <li>{{ k }}: {{ v }}</li>
{% endfor %}
</ul>
</body>
</html>
"""


def write_report(metrics: Dict[str, float], out_path: str = "report.html", title: str = "SpectraSense Report"):
    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(title=title, metrics=metrics)
    Path(out_path).write_text(html)
    print(f"Report written to {out_path}")
