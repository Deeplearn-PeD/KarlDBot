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
"""

class Report:
    def __init__(self, Problem: DataScienceProblem, model_name: str):
        self.model_name = model_name
        self.problem = Problem
        self.report = None
        self.filename = None

    def render(self):
        template = jinja2.Template(TEMPLATE)
        self.report =  template.render(Problem_name=self.problem.problem_name,
                               model_name=self.model_name,
                               date=datetime.now(),
                               description=self.problem.description
                               )
        return self.report

    def save(self, filename):
        self.filename = filename
        if self.report is None:
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
