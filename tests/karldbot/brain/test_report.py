import pytest
from karldbot.brain import Koder, CodeReviewer
from karldbot.rle.environment import DataScienceProblem
from karldbot.brain.report import Report

def test_report():
    dsp = DataScienceProblem('Climate analysis', 'karldbot/rle/datasets/clima_PR.csv.gz')
    dsp.set_description('calculate the correlation between temperature and precipitation by year and geocode')
    report = Report(dsp, 'gpt-4o')
    assert report.model_name == 'gpt-4o'
    mdreport = report.render()
    assert mdreport.startswith('# Report for Climate analysis using gpt-4o')
