from datetime import datetime
from typing import Any
import os

import jinja2

from karldbot.environment import DataScienceProblem

TEMPLATE = """# Report for {{ problem_name }} using the {{ model_name }} LLM model
This report describes the steps taken by Karl the Koder to solve the problem described below.

## Problem Description

{{ description | safe }}

Date: {{ date }}

### Proposed solution
#### Step-by-Step Solution
{% for code, review in solution_steps %}
### Step {{ loop.index }}
{% if loop.index == 1 %}
Based on the problem description, Karl the Koder generated the following code:
{% else %}
Then the coder produced the following improved code:
{% endif %}

```python
{{ code.code }}
```
#### Explanation

{{ code.explanation | safe }}

### Code Review
The review of the code above is as follows:

- **Correctness:** {{ review.review.correctness }}
- **Efficiency:** {{ review.review.efficiency }}
- **Clarity:** {{ review.review.clarity }}
- **Recommendations:** {{ review.review.recommendations }}

{% if review.review.approved %}
This code was approved by the reviewer.
{% else %}
This code was not approved by the reviewer.
{% endif %}
{% endfor %}
"""


class Report:
    def __init__(self, problem: DataScienceProblem, model_name: str):
        self.model_name = model_name
        self.problem = problem
        self.report: str | None = None
        self.filename: str | None = None
        self.coding_steps: list[dict[str, Any]] = []
        self.review_steps: list[dict[str, Any]] = []

    def add_coding_step(self, info: dict[str, Any]) -> None:
        explanation = info.get("code_explanation", "")
        prompt = info.get("code_prompt", "")
        code = info.get("solution", "")
        code = code.strip("```python").strip("```").strip()
        self.coding_steps.append(
            {
                "prompt": prompt,
                "code": code,
                "explanation": explanation,
            }
        )

    def add_review_step(self, info: dict[str, Any]) -> None:
        prompt = info.get("review_prompt", "")
        review = info.get("review", "")
        if not isinstance(review, str) and hasattr(review, "model_dump"):
            review = review.model_dump()
        elif not isinstance(review, str) and hasattr(review, "dict"):
            review = review.dict()
        self.review_steps.append({"prompt": prompt, "review": review})

    def render(self) -> str:
        template = jinja2.Template(TEMPLATE)
        solution_steps = zip(self.coding_steps, self.review_steps)
        try:
            self.report = template.render(
                problem_name=self.problem.problem_name,
                model_name=self.model_name,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                description=self.problem.description,
                solution_steps=solution_steps,
            )
        except jinja2.TemplateError as e:
            raise RuntimeError(f"Error rendering template: {e}")
        return self.report

    def save(self, filename: str) -> None:
        self.filename = filename
        content = self.render()
        try:
            with open(filename, "w") as f:
                f.write(content)
        except IOError as e:
            raise RuntimeError(f"Error saving report to {filename}: {e}")

    def open(self) -> None:
        if self.filename is None:
            raise ValueError("No filename set. Call save() first.")
        import platform

        system = platform.system()
        if system == "Windows":
            os.system(f"start {self.filename}")
        elif system == "Darwin":
            os.system(f"open {self.filename}")
        elif system == "Linux":
            os.system(f"xdg-open {self.filename}")
        else:
            raise ValueError(f"OS not supported: {system}")
