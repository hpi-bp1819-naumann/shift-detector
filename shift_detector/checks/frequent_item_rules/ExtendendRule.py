class ExtendedRule:

    def __init__(self, left_side, right_side, supports_of_left_side, delta_supports_of_left_side, supports,
                 delta_supports, confidences, delta_confidences):

        self.left_side = left_side
        self.right_side = right_side
        self.supports_of_left_side = supports_of_left_side
        self.delta_supports_of_left_side = delta_supports_of_left_side
        self.supports = supports
        self.delta_supports = delta_supports
        self.confidences = confidences
        self.delta_confidences = delta_confidences

    def all_sides(self):
        return self.left_side + self.right_side


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

    def is_supercluster(self, new_rule):
        own_attributes = set(self.attributes)
        new_rule_attributes = set(new_rule.left_side + new_rule.right_side)

        if own_attributes.issubset(new_rule_attributes):
            if abs(new_rule.delta_supports_of_left_side) <= abs(self.max_abs_delta_support_left):
                return True
        else:
            return False

    def print(self):
        attribute_string = ''
        for attribute in self.attributes:
            attribute_string += str(attribute[0]) + ':' + str(attribute[1]) + ' '
        print('[ ', attribute_string, ']')

        print('max_delta_support: ', self.max_abs_delta_support_left, '\t max_delta_confidence:',
              self.max_abs_delta_confidence, '\t number of subrules:', len(self.subcluster))
        print('\n')

