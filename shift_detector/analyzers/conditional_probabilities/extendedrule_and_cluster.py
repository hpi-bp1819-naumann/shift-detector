class ExtendedRule:

    def __init__(self, left_side, right_side, supports_of_left_side, delta_supports_of_left_side, supports,
                 delta_supports, confidences, delta_confidences, length, all_sides, groupkey):

        self.left_side = left_side
        self.right_side = right_side
        self.supports_of_left_side = supports_of_left_side
        self.delta_supports_of_left_side = delta_supports_of_left_side
        self.supports = supports
        self.delta_supports = delta_supports
        self.confidences = confidences
        self.delta_confidences = delta_confidences
        self.length = length
        self.all_sides = all_sides
        self.groupkey = groupkey


class RuleCluster:

    def __init__(self, attributes, rules):
        self.attributes = attributes
        self.rules = rules
        self.max_abs_delta_support_left = None
        self.max_abs_delta_confidence = None
        self.subcluster = []

    def calculate_max_support_and_confidence(self):
        self.max_abs_delta_support_left = 0
        self.max_abs_delta_confidence = 0
        for rule in self.rules:
            if abs(rule.delta_supports_of_left_side) > abs(self.max_abs_delta_support_left):
                self.max_abs_delta_support_left = abs(rule.delta_supports_of_left_side)
            if abs(rule.delta_confidences) > abs(self.max_abs_delta_confidence):
                self.max_abs_delta_confidence = abs(rule.delta_confidences)

    def is_more_specific_cluster_of(self, other_cluster):
        own_attributes = set(self.attributes)
        other_attributes = set(other_cluster.attributes)
        # print('own_attributes', own_attributes)
        # print('other_attributes', other_attributes)
        # print(own_attributes.issubset(other_attributes))
        return other_attributes.issubset(own_attributes)

    def compare_to_cluster(self, other_cluster):
        own_attributes = set(self.attributes)
        other_attributes = set(other_cluster.attributes)

        if other_attributes.issubset(own_attributes):
            print()

        return other_attributes.issubset(own_attributes)

    def compare_to_new_rule(self, new_rule):
        own_attributes = set(self.attributes)
        new_rule_attributes = set(new_rule.left_side + new_rule.right_side)

        if own_attributes.issubset(new_rule_attributes):
            if abs(new_rule.delta_supports_of_left_side) <= abs(self.max_abs_delta_support_left):
                return 'supercluster'
        else:
            return 'not together'

    def print(self):
        attribute_string = ''
        for attribute in self.attributes:
            attribute_string += str(attribute[0]) + ':' + str(attribute[1]) + ' '
        print('[ ', attribute_string, ']')

        print('max_delta_support: ', self.max_abs_delta_support_left, '\t max_delta_confidence:',
              self.max_abs_delta_confidence, '\t number of subrules:', len(self.subcluster))

        subrules_string = ''
        for subrule in self.subcluster:
            subrules_string += str(subrule.all_sides) + '  '
        if len(self.subcluster) > 0:
            print('-->', subrules_string)

        print('\n')

    def attributes_as_string(self):
        result = ''
        for attribute in self.attributes:
            result += str(attribute)
        return result







