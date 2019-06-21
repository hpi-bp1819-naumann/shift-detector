from morpheus.precalculations.conditional_probabilities import fpgrowth


class RuleCluster:
    def __init__(self, rule):
        self.attribute_value_pairs = set(rule.left_side + rule.right_side)
        self.main_rule = rule
        self.sub_rules = []

    def is_super_cluster_of(self, rule):
        other_attribute_value_pairs = set(rule.left_side + rule.right_side)

        return self.attribute_value_pairs.issubset(other_attribute_value_pairs) and abs(
            rule.delta_supports) <= abs(self.main_rule.delta_supports)

    def __str__(self):
        attribute_string = ', '.join('{a[0]}: {a[1]}'.format(a=attribute) for attribute in self.attribute_value_pairs)
        return_str = '[{}]\n'.format(attribute_string)
        return_str += fpgrowth.to_string(self.main_rule) + '\n'
        return_str += 'delta_support: {:.0%}, number of sub-rules: {}\n'.format(abs(self.main_rule.delta_supports),
                                                                                len(self.sub_rules))
        return return_str
