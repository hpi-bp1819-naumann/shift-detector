from shift_detector.checks.conditional_probabilities_check import ConditionalProbabilitiesCheck
from shift_detector.detector import Detector
import pandas as pd

if __name__ == "__main__":
    args = {'train': pd.read_csv('/Users/merzljl/Desktop/poke_1.csv').drop('#', axis=1),
            'test': pd.read_csv('/Users/merzljl/Desktop/poke_2.csv').drop('#', axis=1)}


    detector = Detector(args['train'], args['test'])
    detector.run(ConditionalProbabilitiesCheck(min_support=0.01, min_confidence=0.1, rule_limit=10,
                                               min_delta_supports=0.10, min_delta_confidences=0.1))
    detector.evaluate()
