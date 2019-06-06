from shift_detector.Detector import Detector
from shift_detector.checks.ConditionalProbabilitiesCheck import ConditionalProbabilitiesCheck

if __name__ == "__main__":
    args = {'train': '/Users/merzljl/Desktop/train.csv',
            'test': '/Users/merzljl/Desktop/audits.csv', 'sep': ','}

    detector = Detector(args['train'], args['test'], delimiter=args['sep'])
    detector.run(ConditionalProbabilitiesCheck())
    detector.evaluate()
