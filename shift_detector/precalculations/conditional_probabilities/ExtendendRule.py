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

    def __str__(self):
        return ('{left_sides} ==> {right_sides} [SUPPORTS_OF_LEFT_SIDES: {supports_of_left_sides}, '
                'DELTA_SUPPORTS_OF_LEFT_SIDES: {delta_supports_of_left_sides}, SUPPORTS: {supports}, '
                'DELTA_SUPPORTS: {delta_supports}, CONFIDENCES: {confidences}, '
                'DELTA_CONFIDENCES: {delta_confidences}]').format(
            left_sides=', '.join('{}: {}'.format(l[0].upper(), l[1]) for l in self.left_side),
            right_sides='()' if not self.right_side else ', '.join(
                '{}: {}'.format(r[0].upper(), r[1]) for r in self.right_side),
            supports_of_left_sides=self.supports_of_left_side,
            delta_supports_of_left_sides=self.delta_supports_of_left_side,
            supports=self.supports,
            delta_supports=self.delta_supports,
            confidences=self.confidences,
            delta_confidences=self.delta_confidences
        )


class RuleCluster:
    def __init__(self, rule):
        self.attributes = set(rule.left_side + rule.right_side)
        self.rule = rule
        self.sub_clusters = []

    def is_super_cluster_of(self, other_rule):
        other_attributes = set(other_rule.left_side + other_rule.right_side)

        return self.attributes.issubset(other_attributes) and abs(other_rule.delta_supports) <= abs(
            self.rule.delta_supports)

    def __str__(self):
        return_str = ''
        attribute_string = ''
        for attribute in self.attributes:
            attribute_string += str(attribute[0]) + ':' + str(attribute[1]) + ' '
        return_str += '[ ' + attribute_string + '] \n'
        return_str += 'rule: ' + str(self.rule) + '\n'
        return_str += 'max_delta_support: ' + str(abs(self.rule.delta_supports)) + '\t max_delta_confidence:' + \
                      str(abs(self.rule.delta_confidences)) + '\t number of subrules:' + str(
            len(self.sub_clusters)) + '\n'

        return return_str
