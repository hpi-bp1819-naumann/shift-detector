from shift_detector.precalculations.conditional_probabilities import fpgrowth


class RuleCluster:
    def __init__(self, rule):
        self.attribute_value_pairs = set(rule.left_side + rule.right_side)
        self.rule = rule
        self.sub_rules = []

    def is_super_cluster_of(self, rule):
        other_attribute_value_pairs = set(rule.left_side + rule.right_side)

        return self.attribute_value_pairs.issubset(other_attribute_value_pairs) and abs(
            rule.delta_supports) <= abs(self.rule.delta_supports)

    def __str__(self):
        attribute_string = ', '.join(f'{attribute[0]}: {attribute[1]}' for attribute in self.attribute_value_pairs)
        return_str = f'[{attribute_string}]\n'
        return_str += fpgrowth.to_string(self.rule) + '\n'
        return_str += (f'delta_support: {abs(self.rule.delta_supports):.0%}, '
                       f'number of sub-rules: {len(self.sub_rules)}\n')
        return return_str
