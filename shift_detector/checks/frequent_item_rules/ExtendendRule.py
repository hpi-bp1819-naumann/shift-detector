from pprint import pprint

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

    def __str__(self):
        result = ('{left_sides} ==> {right_sides} [SUPPORTS_OF_LEFT_SIDES: {supports_of_left_sides}, '
                  'DELTA_SUPPORTS_OF_LEFT_SIDES: {delta_supports_of_left_sides}, SUPPORTS: {supports}, '
                  'DELTA_SUPPORTS: {delta_supports}, CONFIDENCES: {confidences}, '
                  'DELTA_CONFIDENCES: {delta_confidences}]').format(
            left_sides=', '.join(f'{l[0].upper()}: {l[1]}' for l in self.left_side),
            right_sides='()' if not self.right_side else ', '.join(f'{l[0].upper()}: {l[1]}' for l in self.right_side),
            supports_of_left_sides=self.supports_of_left_side,
            delta_supports_of_left_sides=self.delta_supports_of_left_side,
            supports=self.supports,
            delta_supports=self.delta_supports,
            confidences=self.confidences,
            delta_confidences=self.delta_confidences
        )
        return result


class RuleCluster:

    def __init__(self, attributes, rules):
        self.attributes = attributes
        self.rules = rules
        self.max_abs_delta_supports = None
        self.max_abs_delta_confidence = None
        self.subcluster = []

    def calculate_max_support_and_confidence(self):
        self.max_abs_delta_supports = 0
        self.max_abs_delta_confidence = 0
        for rule in self.rules:
            if abs(rule.delta_supports) > abs(self.max_abs_delta_supports):
                self.max_abs_delta_supports = abs(rule.delta_supports)
            if abs(rule.delta_confidences) > abs(self.max_abs_delta_confidence):
                self.max_abs_delta_confidence = abs(rule.delta_confidences)

    def is_supercluster(self, new_rule):
        own_attributes = set(self.attributes)
        new_rule_attributes = set(new_rule.left_side + new_rule.right_side)

        if own_attributes.issubset(new_rule_attributes):
            if abs(new_rule.delta_supports) <= abs(self.max_abs_delta_supports):
                return True
        else:
            return False

    def ___str__(self):
        return_str = ''
        attribute_string = ''
        for attribute in self.attributes:
            attribute_string += str(attribute[0]) + ':' + str(attribute[1]) + ' '
        return_str += '[ ' + attribute_string + '] \n'
        return_str += 'rule: ' + str(self.rules[0]) + '\n'
        return_str += 'max_delta_support: ' + str(self.max_abs_delta_supports) + '\t max_delta_confidence:' + \
              str(self.max_abs_delta_confidence) + '\t number of subrules:' + str(len(self.subcluster)) + '\n'

        return return_str
