from shift_detector.Detector import Detector
from shift_detector.checks.ConditionalProbabilitiesCheck import ConditionalProbabilitiesCheck

if __name__ == "__main__":
    args = {'train': '/Users/merzljl/Desktop/train.csv',
            'test': '/Users/merzljl/Desktop/audits.csv', 'sep': ','}

    detector = Detector(args['train'], args['test'], delimiter=args['sep'])
    detector.run(ConditionalProbabilitiesCheck(min_support=0.01, min_confidence=0.1, rule_limit=10,
                                               min_delta_supports=0.01, min_delta_confidences=0.01))
    detector.evaluate()
