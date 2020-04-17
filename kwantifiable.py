# Quant trading machine learning platform

import logger
import argparse
import sys
from value_predictor import value_predictor
from quote_data import clear, update_all_symbols

class kwantifiable:

    def __init__(self):
        self.log = logger.setup_logger('kwantifiable')
        self.log.debug('Starting Kwantifiable')

        parser = argparse.ArgumentParser("usage: %prog [options] arg1 arg2")

        parser.add_argument("--clear", dest="should_clear_data", default=False,
                            action="store_true", help="Clear all stored data (quotes, graphs, models)")

        parser.add_argument("-u", "--update", dest="should_update_data", default=False,
                            action="store_true", help="Update quote data")

        parser.add_argument("-t", "--train", dest="train_models", default=False,
                            action="store_true", help="(Re)Train models")

        parser.add_argument("-p", "--predict", dest="predict_values", default=False,
                            action="store_true", help="Use models to predict values")

        parser.add_argument("-v", "--value", dest="value_type", default=None,
                            type=str, choices=['open', 'close'], help="Type of value to train / predict")

        parser.add_argument("-b", "--backpoints", dest="train_points", default=50,
                            type=int, help="Number of historical points to train on")

        parser.add_argument("-e", "--epochs", dest="train_epochs", default=200,
                            type=int, help="Number of epochs to train on")

        options = parser.parse_args()

        if (len(sys.argv) == 1):
            parser.print_help(sys.stderr)

        self.should_clear_data = options.should_clear_data
        self.should_update_data = options.should_update_data
        self.should_train_models = options.train_models
        self.should_predict_values = options.predict_values
        self.value_type = options.value_type
        self.train_epochs = options.train_epochs
        self.train_points = options.train_points

        if self.should_predict_values or self.should_train_models:
            if self.value_type == None:
                print("Must specify type of value when training / predicting")
                exit()

    def train_models(self):
        predictor = value_predictor(self.value_type, self.train_epochs, self.train_points)
        predictor.train()
        
    def predict_values(self):
        predictor = value_predictor(self.value_type, self.train_epochs, self.train_points)
        predictor.predict()

    def run(self):
        if self.should_clear_data:
            clear("quotes/")
            clear("models/")
            clear("graphs/")

        if self.should_update_data:
            update_all_symbols('daily_adj')

        if self.should_train_models:
            self.train_models()

        if self.should_predict_values:
            self.predict_values()

if __name__ == "__main__":
    agent = kwantifiable()
    agent.run()
