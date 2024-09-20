import pytest

from karldbot.brain import Koder, QualityReport, CodeReviewer, CodeOutput
from karldbot.rle.environment import Environment, DataScienceProblem


problem = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
problem.set_description('calcule a correlação entre a temperatura mínima(temp_min), e a precipitação média(precip_med) por ano(date) e geocodigo')
problem.load_data()
env = Environment('test_env', problem)

def test_koder():
    koder = Koder('gpt-4o')
    assert koder.language_model.model == 'gpt-4o'

@pytest.mark.skip("Already tested")
def test_write_code():
    koder = Koder()
    koder.set_problem(problem)
    code = koder.write_code(problem.description)
    assert isinstance(code, CodeOutput)
    assert code.language == 'python'

def test_code_reviewer():
    code_reviewer = CodeReviewer()
    assert code_reviewer.language_model.model == 'gpt-4o'

def test_review_code():
    code_reviewer = CodeReviewer()
    koder = Koder()
    koder.set_problem(problem)
    code = koder.write_code(problem.description)
    report = code_reviewer.review_code(code)
    assert isinstance(report, QualityReport)
    assert report.correctness >= 0
    assert report.efficiency >= 0
    assert report.correctness <= 10
    assert report.efficiency <= 10

