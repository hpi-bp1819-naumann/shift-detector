from shift_detector.Detector import Detector
from shift_detector.analyzers.BasicAnalyzer import BasicAnalyzer
from shift_detector.analyzers.KsChiAnalyzer import KsChiAnalyzer
from resources.config import Config

if __name__ == "__main__":

    train_path = Config.TRAIN_FILE_PATH
    audits_path = Config.PRODUCTION_FILE_PATH
    
    Detector(train_path, audits_path, seperator=',') \
    .add_analyzer(BasicAnalyzer) \
    .run()
    
    #modules=[])
    #.add_analyzer(KsChiAnalyzer) \