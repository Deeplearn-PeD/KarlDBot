"""
In this model we implement a Report generation class that will generate a markdown document describing all the steps
of the code generation and review process. untill the problem is solved.
"""

from datetime import datetime
import os
import jinja2
from karldbot.rle.environment import DataScienceProblem

TEMPLATE = """# Report for {{ Problem_name }} using the {{ model_name }} LLM model
This report describes the steps taken by Karl the Koder to solve the problem described below.

## Problem Description

{{ description | safe }}

Date: {{ date }}

### Proposed solution
#### Step-by-Step Solution
{% for code, review  in solution_steps %}
### Step {{ loop.index }}
{%if loop.index == 1 %}
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
    def __init__(self, Problem: DataScienceProblem, model_name: str):
        self.model_name = model_name
        self.problem = Problem
        self.report = None
        self.filename = None
        self.coding_steps = []
        self.review_steps = []

    def add_coding_step(self, info: dict):
        explanation = '' if 'code_explanation' not in info else info['code_explanation']
        prompt = '' if 'code_prompt' not in info else info['code_prompt']
        code = '' if 'solution' not in info else info['solution']
        code = code.strip("```python").strip("```").strip()
        self.coding_steps.append({'prompt': prompt, 'code': code, 'explanation': explanation})

    def render(self):
        template = jinja2.Template(TEMPLATE)
        solution_steps = zip(self.coding_steps, self.review_steps)
        try:
            self.report = template.render(
                Problem_name=self.problem.problem_name,
                model_name=self.model_name,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                description=self.problem.description,
                solution_steps=solution_steps
            )
        except jinja2.TemplateError as e:
            raise RuntimeError(f"Error rendering template: {e}")
        return self.report

    def add_review_step(self, info: dict):
        """
        Add a review step to the report.
        :param info: A dictionary containing review information.
        """
        prompt = '' if 'review_prompt' not in info else info['review_prompt']
        report = '' if 'review' not in info else info['review']
        report = '' if  isinstance(report, str) else report.dict()
        print(report)
        self.review_steps.append({'prompt': prompt, 'review': report})

    def save(self, filename):
        self.filename = filename

        self.render()
        try:
            with open(filename, 'w') as f:
                f.write(self.report)
        except IOError as e:
            raise RuntimeError(f"Error saving report to {filename}: {e}")

    def open(self):
        """
        Open the generated report in the default markdown viewer.
        :return:
        """
        """Open the generated report in the default markdown viewer."""
        import platform
        if platform.system() == 'Windows':
            os.system(f'start {self.filename}')
        elif platform.system() == 'Darwin':
            os.system(f'open {self.filename}')
        elif platform.system() == 'Linux':
            os.system(f'xdg-open {self.filename}')
        else:
            raise ValueError("OS not supported")
