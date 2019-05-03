import argparse
from shift_detector.Detector import Detector
from shift_detector.analyzers.BasicAnalyzer import BasicAnalyzer
from shift_detector.analyzers.KsChiAnalyzer import KsChiAnalyzer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-train", "--train", required=True, help="path of train dataset")
    ap.add_argument("-test", "--test", required=True, help="path of test dataset")
    ap.add_argument("-s", "--sep", required=True, help="separator for datasets")
    args = vars(ap.parse_args())

    train_path = args["train"]
    audits_path = args["test"]
    separator = args["sep"]

    Detector(train_path, audits_path, separator=separator) \
    .add_analyzer(KsChiAnalyzer) \
    .run()
    
    #modules=[])
    #.add_analyzer(KsChiAnalyzer) \
    