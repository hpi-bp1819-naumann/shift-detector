from shift_detector.precalculations.conditional_probabilities import fpgrowth


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
        attribute_string = ', '.join(f'{attribute[0]}: {attribute[1]}' for attribute in self.attributes)
        return_str = f'[{attribute_string}]\n'
        return_str += fpgrowth.to_string(self.rule) + '\n'
        return_str += (f'max_delta_support: {abs(self.rule.delta_supports):.0%}, '
                       f'max_delta_confidence: {abs(self.rule.delta_confidences):.0%}, '
                       f'number of sub-rules: {len(self.sub_clusters)}\n')
        return return_str
