"""
This is the entry point for the CLI application.

"""
import fire

class KarlInterface:
    def __init__(self):
        pass

    def train(self, model_name: str = 'gpt-4o', data_source: str = 'data.csv'):
        """
        Train a model on a given data source.
        :param model_name: Name of the model to train.
        :param data_source: Path or URL to the data source.
        :return: Trained model.
        """
        print(f"Training model {model_name} on data source {data_source}...")

    def view_report(self):
        pass


def main():
    fire.Fire(KarlInterface)