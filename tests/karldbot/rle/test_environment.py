import pytest
from karldbot.rle.environment import Environment, DataScienceProblem

def test_environment():
    env = Environment('test_env')
    assert env.ename == 'test_env'
    assert env.state == None
    assert env.reward == None
    assert env.done == None
    assert env.info == None

def test_datascienceproblem():
    dsp = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    assert dsp.problem_name == 'Climate analysis'
    dsp.set_description('calculate the correlation between temperature and precipitation by year and geocode')
    assert dsp.description == 'calculate the correlation between temperature and precipitation by year and geocode'

def test_load_data():
    dsp = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    dsp.load_data()
    assert dsp.data_loaded


