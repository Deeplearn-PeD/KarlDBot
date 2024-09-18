import pytest

from karldbot.brain import Koder, QualityReport, CodeReviewer
from karldbot.rle.environment import Environment, DataScienceProblem

class Problem(DataScienceProblem):
    def __init__(self, problem_name, data_source):
        super().__init__(problem_name, data_source)

    def evaluate_problem(self, solution):
        return 5
env = Environment('test_env')
problem = Problem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
problem.set_description('calcule a correlação entre a temperatura mínima(temp_min), e a precipitação média(precip_med) por ano(date) e geocodigo')
problem.load_data()

def test_koder():
    koder = Koder('gpt-4o')
    assert koder.language_model.model == 'gpt-4o'

def test_write_code():
    koder = Koder('gpt-4o')
    koder.set_problem(problem)
    code = koder.write_code(problem.description)
    assert code == "print('Hello, World!')"
