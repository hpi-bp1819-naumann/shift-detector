from shift_detector.Detector import Detector
from shift_detector.checks.SimpleCheck import SimpleCheck
from shift_detector.checks.ConditionalProbabilitiesCheck import ConditionalProbabilitiesCheck

# Starting via console:
# python3 main.py --train ./train_ascii.csv --test ./audits_ascii.csv --sep  ";"
if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-train", "--train", required=True, help="path of train dataset")
    # ap.add_argument("-test", "--test", required=True, help="path of test dataset")
    # ap.add_argument("-s", "--sep", required=True, help="separator for datasets")
    # args = vars(ap.parse_args())

    args = {'train': '/Users/pzimme/Desktop/Datasets/train_leonard.csv',
            'test': '/Users/pzimme/Desktop/Datasets/audits_leonard.csv', 'sep': ','}
    # train_path = args["train"]
    # audits_path = args["test"]
    # separator = args["sep"]

    detector = Detector(args['train'], args['test'], delimiter=args['sep'])
    detector.add_checks(ConditionalProbabilitiesCheck())
    # detector.add_check(Chi2Check())

    detector.run()
    detector.evaluate()
    print()

    # detector.checks_reports[0].reports.evaluate(significance=0.5)
    # detector.evaluate()

    # modules=[])
    # .add_check(KsChiCheck)
