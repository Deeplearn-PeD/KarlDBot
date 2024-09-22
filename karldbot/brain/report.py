"""
In this model we implement a Report generation class that will generate a markdown document describing all the steps
of the code generation and review process. untill the problem is solved.
"""

from datetime import datetime
import jinja2
import os
import webbrowser
from karldbot.rle.environment import DataScienceProblem

TEMPLATE = """# Report for {{ Problem_name }} using {{ model_name }}
This report describes the steps taken by Karl the Koder to solve the problem described below.
## Problem Description
{{ description | safe }}

Date: {{ date }}

## Step-by-Step Solution
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
{{ code.explanation | safe }}

The review of the code is as follows:
- **Correctness:** {{ review.correctness }}
- **Efficiency:** {{ review.efficiency }}
- **Style:** {{ review.style }}
 - **Recommendations:** {{ review.recommendations }}
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
        code = '' if 'code' not in info else info['solution']
        self.coding_steps.append({'prompt': prompt, 'code': code, 'explanation': explanation})

    def render(self):
        template = jinja2.Template(TEMPLATE)
        solution_steps = zip(self.coding_steps, self.review_steps)
        self.report =  template.render(Problem_name=self.problem.problem_name,
                               model_name=self.model_name,
                               date=datetime.now(),
                               description=self.problem.description
                               )
        return self.report

    def add_review_step(self, info: dict):
        """
        Add a review step to the report.
        :param prompt: The prompt that was used to generate the code.
        :param code: The code that was generated.
        :param report: The report that was generated.
        :return:
        """
        prompt = '' if 'review_prompt' not in info else info['review_prompt']
        report = '' if 'recomendations' not in info else info['recommendations']
        self.review_steps.append({'prompt': prompt, 'report': report})

    def save(self, filename):
        self.filename = filename

        self.render()
        with open(filename,'w') as f:
            f.write(self.report)

    def open(self):
        """
        Open the generated report in the default markdown viewer.
        :return:
        """
        # check what the OS is
        import platform
        if platform.system() == 'Windows':
            os.system(f'start {self.filename}')
        elif platform.system() == 'Darwin':
            os.system(f'open {self.filename}')
        elif platform.system() == 'Linux':
            os.system(f'xdg-open {self.filename}')
        else:
            raise ValueError("OS not supported")
