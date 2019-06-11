from shift_detector.Detector import Detector
from shift_detector.checks.ConditionalProbabilitiesCheck import ConditionalProbabilitiesCheck

if __name__ == "__main__":
    args = {'train': '/Users/merzljl/Desktop/tpch_data_1/part.tbl',
            'test': '/Users/merzljl/Desktop/tpch_data_2/part.tbl', 'sep': '|'}

    detector = Detector(args['train'], args['test'], delimiter=args['sep'])
    detector.run(ConditionalProbabilitiesCheck(min_support=0.01, min_confidence=0.1, rule_limit=100,
                                               min_delta_supports=0.01, min_delta_confidences=0.01))
    detector.evaluate()
