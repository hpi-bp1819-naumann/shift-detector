import argparse
from shift_detector.Detector import Detector
from shift_detector.checks.BasicCheck import BasicCheck
from shift_detector.checks.KsChiCheck import KsChiCheck
from shift_detector.checks.Chi2Check import Chi2Check
from shift_detector.checks.FrequentItemRulesCheck import FrequentItemsetCheck
from shift_detector.checks.TestCheck import TestCheck

# Starting via console:
# python3 main.py --train ./train_ascii.csv --test ./audits_ascii.csv --sep  ";"
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("-train", "--train", required=True, help="path of train dataset")
    # ap.add_argument("-test", "--test", required=True, help="path of test dataset")
    # ap.add_argument("-s", "--sep", required=True, help="separator for datasets")
    args = vars(ap.parse_args())

    args = {'train': '/Users/pzimme/Desktop/Datasets/audits_leonard.csv',
            'test':'/Users/pzimme/Desktop/Datasets/train_leonard.csv', 'sep': ','}
    train_path = args["train"]
    audits_path = args["test"]
    separator = args["sep"]

    Detector(train_path, audits_path, separator=separator) \
        .add_check(TestCheck) \
        .add_check(FrequentItemsetCheck) \
        .run()

    # modules=[])
    # .add_check(KsChiCheck) \
