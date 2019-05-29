import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from shift_detector.Detector import Detector
from shift_detector.checks.statistical_checks.TextMetadataStatisticalCheck import TextMetadataStatisticalCheck
from shift_detector.preprocessors.Store import Store
from shift_detector.preprocessors.TextMetadata import NumCharsMetadata, NumWordsMetadata, DistinctWordsRatioMetadata, \
    LanguagePerParagraph, UnknownWordRatioMetadata, StopwordRatioMetadata


class TestTextMetadataStatisticalCheck(unittest.TestCase):

    def setUp(self):
        self.poems = [
            'Tell me not, in mournful numbers,\nLife is but an empty dream!\nFor the soul is dead that slumbers,\nAnd things are not what they seem.',
            'Life is real! Life is earnest!\nAnd the grave is not its goal;\nDust thou art, to dust returnest,\nWas not spoken of the soul.',
            'Not enjoyment, and not sorrow,\nIs our destined end or way;\nBut to act, that each to-morrow\nFind us farther than to-day.',
            'Art is long, and Time is fleeting,\nAnd our hearts, though stout and brave,\nStill, like muffled drums, are beating\nFuneral marches to the grave.',
            'In the world’s broad field of battle,\nIn the bivouac of Life,\nBe not like dumb, driven cattle!\nBe a hero in the strife! ',
            'Trust no Future, howe’er pleasant!\nLet the dead Past bury its dead!\nAct,— act in the living Present!\nHeart within, and God o’erhead! ',
            'LIFE, believe, is not a dream\nSo dark as sages say;\nOft a little morning rain\nForetells a pleasant day.\nSometimes there are clouds of gloom,\nBut these are transient all;\nIf the shower will make the roses bloom,\nO why lament its fall ? ',
            "Rapidly, merrily,\nLife's sunny hours flit by,\nGratefully, cheerily,\nEnjoy them as they fly !",
            "What though Death at times steps in\nAnd calls our Best away ?\nWhat though sorrow seems to win,\nO'er hope, a heavy sway ?\nYet hope again elastic springs,\nUnconquered, though she fell;\nStill buoyant are her golden wings,\nStill strong to bear us well.\nManfully, fearlessly,\nThe day of trial bear,\nFor gloriously, victoriously,\nCan courage quell despair ! ",
            'When sinks my heart in hopeless gloom,\nAnd life can shew no joy for me;\nAnd I behold a yawning tomb,\nWhere bowers and palaces should be;\nIn vain you talk of morbid dreams;\nIn vain you gaily smiling say,\nThat what to me so dreary seems,\nThe healthy mind deems bright and gay.',
            "I too have smiled, and thought like you,\nBut madly smiled, and falsely deemed:\nTruth led me to the present view,\nI'm waking now -- 'twas then I dreamed. ",
            'I lately saw a sunset sky,\nAnd stood enraptured to behold\nIts varied hues of glorious dye:\nFirst, fleecy clouds of shining gold; ',
            'These blushing took a rosy hue;\nBeneath them shone a flood of green;\nNor less divine, the glorious blue\nThat smiled above them and between.',
            'I cannot name each lovely shade;\nI cannot say how bright they shone;\nBut one by one, I saw them fade;\nAnd what remained whey they were gone?',
            "Dull clouds remained, of sombre hue,\nAnd when their borrowed charm was o'er,\nThe azure sky had faded too,\nThat smiled so softly bright before. ",
            'So, gilded by the glow of youth,\nOur varied life looks fair and gay;\nAnd so remains the naked truth,\nWhen that false light is past away. ',
            'Why blame ye, then, my keener sight,\nThat clearly sees a world of woes,\nThrough all the haze of golden light,\nThat flattering Falsehood round it throws? ',
            'When the young mother smiles above\nThe first-born darling of her heart,\nHer bosom glows with earnest love,\nWhile tears of silent transport start. ',
            'Fond dreamer! little does she know\nThe anxious toil, the suffering,\nThe blasted hopes, the burning woe,\nThe object of her joy will bring. ',
            'Her blinded eyes behold not now\nWhat, soon or late, must be his doom;\nThe anguish that will cloud his brow,\nThe bed of death, the dreary tomb. ',
            'As little know the youthful pair,\nIn mutual love supremely blest,\nWhat weariness, and cold despair,\nEre long, will seize the aching breast. ',
            'And, even, should Love and Faith remain,\n(The greatest blessings life can show,)\nAmid adversity and pain,\nTo shine, throughout with cheering glow; ',
            'They do not see how cruel Death\nComes on, their loving hearts to part:\nOne feels not now the gasping breath,\nThe rending of the earth-bound heart, --',
            "The soul's and body's agony,\nEre she may sink to her repose,\nThe sad survivor cannot see\nThe grave above his darling close;",
            'Nor how, despairing and alone,\nHe then must wear his life away;\nAnd linger, feebly toiling on,\nAnd fainting, sink into decay. ',
            'Oh, Youth may listen patiently,\nWhile sad Experience tells her tale;\nBut Doubt sits smiling in his eye,\nFor ardent Hope will still prevail!',
            "He hears how feeble Pleasure dies,\nBy guilt destroyed, and pain and woe;\nHe turns to Hope -\xad and she replies,\n'Believe it not -\xad it is not so!' ",
            "Oh, heed her not!' Experience says,\n'For thus she whispered once to me;\nShe told me, in my youthful days,\nHow glorious manhood's prime would be. ",
            "When, in the time of early Spring,\nToo chill the winds that o'er me pass'd,\nShe said, each coming day would bring\nA fairer heaven, a gentler blast. ",
            "And when the sun too seldom beamed,\nThe sky, o'ercast, too darkly frowned,\nThe soaking rain too constant streamed,\nAnd mists too dreary gathered round;",
            "She told me Summer's glorious ray\nWould chase those vapours all away,\nAnd scatter glories round,\nWith sweetest music fill the trees,\nLoad with rich scent the gentle breeze,\nAnd strew with flowers the ground. ",
            'But when, beneath that scorching ray,\nI languished, weary, through the day,\nWhile birds refused to sing,\nVerdure decayed from field and tree,\nAnd panting Nature mourned with me\nThe freshness of the Spring. ',
            '"Wait but a little while," she said,\n"Till Summer\'s burning days are fled;\nAnd Autumn shall restore,\nWith golden riches of her own,\nAnd Summer\'s glories mellowed down,\nThe freshness you deplore." ',
            'It has neither a beginning nor an end\nYou can never predict where it will bend.',
            'Life is a teacher, it will show you the way\nBut unless you live it...it will run away.',
            'If you have no fear of living, you will find\nNo fear of death and you will not mind.',
            'You have to feel the agonizing sorrow and feel the pain\nOnly then it will heal and you will be whole again.',
            'It is in every leaf, in your smile, in your tears\nIn your toil, in your triumphs and in your fears.',
            'Just enjoy the journey without looking back\nSavour the senses and you will not lack.',
            'Truth is more in the process than the result\nLiberates you from thought and you can exult',
            'To see the truth in the false, thats the key\nTo understand, without changing it...just let it be\nLove life, live it and it will set you free.... ',
            "He wakes, who never thought to wake again,\nWho held the end was Death. He opens eyes\nSlowly, to one long livid oozing plain\nClosed down by the strange eyeless heavens. He lies;\nAnd waits; and once in timeless sick surmise\nThrough the dead air heaves up an unknown hand,\nLike a dry branch. No life is in that land,\nHimself not lives, but is a thing that cries;\nAn unmeaning point upon the mud; a speck\nOf moveless horror; an Immortal One\nCleansed of the world, sentient and dead; a fly\nFast-stuck in grey sweat on a corpse's neck.",
            "I thought when love for you died, I should die.\nIt's dead. Alone, most strangely, I live on. ",
            'Being released from the womb of a woman\nWe walk towards another\nTo be chained again',
            'None is dependable\nNone is lovable ',
            'Life is a coplex road\nWalls, stones, mud, water…',
            'Life is a groaning running through\nGraveyards',
            "Life is too short to be spent\ngriping about the past,\nthings you don't have,\nplaces you haven't seen,\nthings you haven't done.",
            'Life is too short to be spent\nholding grievances against another,\nfinding fault in your brother,\ncounting the wrongs done on you.',
            "Life is just long enough\nto enjoy the beauty of a sunrise\nthe smell of wet earth\nand the sound of laughter\nafter a long day's work.",
            'Life is just long enough\nto practice compassion and generosity,\nto comfort the grieving,\nto lend strength to the fainthearted,\nand direction to the lost. ',
            'A student life is a golden life, truly it is told.\nThe student who has a golden crown of his study.\nWould be a great man in his life with a great hold.\nOtherwise an idle student becomes a wild rowdy.',
            'Every student should use his time in a proper way.\nIf he kills his time, surely time itself always kills him.\nsuch student repents in life every moment of a day.\nA studen must think that his time is a golden rim.',
            'To be a Dr.an engineer, a lawyer or a politician.\nTime makes him all for all to become more of thing.\nGolden time of a golden life even makes a muscian.\nOne who neglects this period of life becomes nothing.',
            'This period has a great charm to make also a magician.\nA life is in the hands of a student to make him something. ',
            "Some say the world will end in fire,\nSome say in ice.\nFrom what I've tasted of desire\nI hold with those who favor fire.\nBut if it had to perish twice,\nI think I know enough of hate\nTo say that for destruction ice\nIs also great\nAnd would suffice. ",
            'I left you in the morning,\nAnd in the morning glow,\nYou walked a way beside me\nTo make me sad to go.\nDo you know me in the gloaming,\nGaunt and dusty gray with roaming?\nAre you dumb because you know me not,\nOr dumb because you know? ',
            "All for me And not a question\nFor the faded flowers gay\nThat could take me from beside you\nFor the ages of a day?\nThey are yours, and be the measure\nOf their worth for you to treasure,\nThe measure of the little while\nThat I've been long away. ",
            "Here come real stars to fill the upper skies,\nAnd here on earth come emulating flies,\nThat though they never equal stars in size,\n(And they were never really stars at heart)\nAchieve at times a very star-like start.\nOnly, of course, they can't sustain the part. ",
            'I stand amid the roar\nOf a surf-tormented shore,\nAnd I hold within my hand\nGrains of the golden sand-\nHow few! yet how they creep\nThrough my fingers to the deep,\nWhile I weep- while I weep!\nO God! can I not grasp\nThem with a tighter clasp?\nO God! can I not save\nOne from the pitiless wave?\nIs all that we see or seem\nBut a dream within a dream? ',
            'If you die before me\nI would jump down into your grave\nand hug you so innocently\nthat angels will become jealous.',
            'I shall kiss you.\nSo intimately shall I kiss you\nthat your breath becomes mine.\nIn one breath of love shall\nwe merge into hugs of true joy.',
            'Your heart shall beat\nin rhythms unheard\nlike the drums of the desert\nand the wild forest in the night.',
            "You shall murmur in my ears:\n'Oh press me to your chest;\nTear open your chest,\nMake way for me\nto enter into your loving heart\nthat beats only for me in resounding colors.",
            'Tell me please Oh my lover\nIs it a rainbow that I see?\nor the glow of a burning pyre?\nWhy is it that I cannot utter it in words?\nTell me glorious angels of love:\nWhat am I experiencing in uncountable\nmoments of indescribable inner comfort? ',
            'Shivering in your presence\nI shall long to dance with you.',
            'If the dark souls lead you to Hades\nI will dance and dance with you\neven in the nether world.',
            'The most constant\nof all the characteristics\nof Illumination is the\nConsciousness\nof the Absolute.',
            'If you take away my breath\nI will fall down and perish.\nMy body shall return to dust.',
            "I feel God's life-giving breath\nwhen in springtime the nature\nwakes up from the winter sleep.",
            'I shall open my mouth\nand gladly praise my God.',
            'Wilt thou forgive that sin where I begun,\nWhich was my sin, though it were done before?\nWilt thou forgive that sin, through which I run,\nAnd do run still, though still I do deplore?\nWhen thou hast done, thou hast not done,\nFor I have more.',
            "Wilt thou forgive that sin which I have won\nOthers to sin, and made my sin their door?\nWilt thou forgive that sin which I did shun\nA year or two, but wallow'd in, a score?\nWhen thou hast done, thou hast not done,\nFor I have more.",
            'I have a sin of fear, that when I have spun\nMy last thread, I shall perish on the shore;\nBut swear by thyself, that at my death thy Son\nShall shine as he shines now, and heretofore;\nAnd, having done that, thou hast done;\nI fear no more.']
        self.phrases = ['Front-line leading edge website',
                        'Upgradable upward-trending software',
                        'Virtual tangible throughput',
                        'Robust secondary open system',
                        'Devolved multimedia knowledge user',
                        'Intuitive encompassing alliance',
                        'Automated 3rd generation benchmark',
                        'Switchable global info-mediaries',
                        'Automated 24/7 alliance',
                        'Down-sized homogeneous software',
                        'Balanced coherent internet solution',
                        'Total intangible groupware',
                        'Implemented zero defect Graphic Interface',
                        'Programmable multi-tasking open system',
                        'Extended non-volatile software',
                        'Organized fresh-thinking initiative',
                        'Public-key demand-driven product',
                        'Visionary asymmetric utilisation',
                        'Horizontal web-enabled structure',
                        'Upgradable intangible paradigm',
                        'Grass-roots background contingency',
                        'User-centric homogeneous ability',
                        'Face to face 5th generation analyzer',
                        'Centralized maximized framework',
                        'Future-proofed client-server internet solution',
                        'Secured mission-critical benchmark',
                        'Virtual zero defect throughput',
                        'Reduced incremental neural-net',
                        'Intuitive real-time help-desk',
                        'Advanced client-server strategy',
                        'Advanced secondary adapter',
                        'Assimilated attitude-oriented hierarchy',
                        'Innovative mobile project',
                        'Synergized tertiary emulation',
                        'Innovative upward-trending framework',
                        'Face to face multi-tasking utilisation',
                        'Multi-layered maximized parallelism',
                        'Versatile 6th generation utilisation',
                        'Automated homogeneous pricing structure',
                        'Ameliorated cohesive model',
                        'Multi-channelled systemic process improvement',
                        'Devolved upward-trending strategy',
                        'Quality-focused secondary Graphical User Interface',
                        'Diverse impactful focus group',
                        'Fundamental modular monitoring',
                        'Cloned exuding hub',
                        'Secured clear-thinking matrix',
                        'Digitized motivating superstructure',
                        'Devolved foreground definition',
                        'Versatile explicit adapter',
                        'Pre-emptive intermediate support',
                        'Business-focused actuating interface',
                        'Compatible empowering internet solution',
                        'Customizable tangible neural-net',
                        'Networked stable methodology',
                        'Networked transitional artificial intelligence',
                        'Function-based secondary definition',
                        'Horizontal 6th generation task-force',
                        'Diverse 3rd generation customer loyalty',
                        'Organic mobile structure',
                        'User-friendly empowering complexity',
                        'Versatile stable frame',
                        'Synchronised directional superstructure',
                        'Enhanced logistical protocol',
                        'Persistent empowering open architecture',
                        'Profit-focused optimal contingency',
                        'User-friendly background migration',
                        'Re-engineered directional array',
                        'Automated upward-trending knowledge base',
                        'Automated tangible attitude',
                        'Multi-channelled mobile core',
                        'Implemented real-time initiative',
                        'Managed homogeneous concept',
                        'Integrated attitude-oriented model']

    def test_not_significant(self):
        df1 = pd.DataFrame.from_dict({'text': self.poems})
        df2 = pd.DataFrame.from_dict({'text': list(reversed(self.poems))})
        store = Store(df1, df2)
        result = TextMetadataStatisticalCheck().run(store)
        self.assertEqual(0, len(result.significant_columns()))

    def test_significant(self):
        df1 = pd.DataFrame.from_dict({'text': self.poems})
        df2 = pd.DataFrame.from_dict({'text': self.phrases})
        store = Store(df1, df2)
        result = TextMetadataStatisticalCheck([NumCharsMetadata(), NumWordsMetadata(),
                                               DistinctWordsRatioMetadata(), LanguagePerParagraph()]
                                              ).run(store)
        self.assertEqual(3, len(result.significant_columns()))

    def test_compliance_with_detector(self):
        df1 = pd.DataFrame.from_dict({'text': ['This is a very important text. It contains information.']})
        df2 = pd.DataFrame.from_dict({'text': ['This is a very important text. It contains information.']})
        detector = Detector(df1=df1, df2=df2)
        detector.add_checks(TextMetadataStatisticalCheck())
        detector.run()
        column_index = pd.MultiIndex.from_product([['text'], ['distinct_words', 'num_chars', 'num_words']],
                                                  names=['column', 'metadata'])
        solution = pd.DataFrame([[1.0, 1.0, 1.0]], columns=column_index, index=['pvalue'])
        assert_frame_equal(solution, detector.check_reports[0].result)

    def test_language_can_be_set(self):
        check = TextMetadataStatisticalCheck([UnknownWordRatioMetadata(), StopwordRatioMetadata()], language='fr')
        md_with_lang = [mdtype for mdtype in check.metadata_preprocessor.text_metadata_types
                        if type(mdtype) in [UnknownWordRatioMetadata, StopwordRatioMetadata]]
        for mdtype in md_with_lang:
            self.assertEqual('fr', mdtype.language)

    def test_infer_language_is_set(self):
        check = TextMetadataStatisticalCheck([UnknownWordRatioMetadata(), StopwordRatioMetadata()], infer_language=True)
        md_with_lang = [mdtype for mdtype in check.metadata_preprocessor.text_metadata_types
                        if type(mdtype) in [UnknownWordRatioMetadata, StopwordRatioMetadata]]
        for mdtype in md_with_lang:
            self.assertTrue(mdtype.infer_language)
