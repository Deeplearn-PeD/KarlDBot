import pytest
from karldbot.rle.environment import Environment, DataScienceProblem

def test_environment():
    problem = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    env = Environment('test_env', problem = problem)
    assert env.ename == 'test_env'
    assert env.score == {"code_correctness": 0, "code_efficiency": 0, "code_style": 0, "approved": False}
    assert env.state == 0
    assert env.reward == 0
    assert env.done == False
    assert env.info == {"recommendations": "", "solution": ""}

def test_datascienceproblem():
    dsp = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    assert dsp.problem_name == 'Climate analysis'
    dsp.set_description('calculate the correlation between temperature and precipitation by year and geocode')
    assert dsp.description == 'calculate the correlation between temperature and precipitation by year and geocode'

def test_load_data():
    dsp = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    dsp.load_data()
    assert dsp.data_loaded


