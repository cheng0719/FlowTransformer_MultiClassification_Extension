#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from typing import List

from framework.enumerations import CategoricalFormat


class ModelInputSpecification:
    def __init__(self, feature_names:List[str], n_numeric_features:int, levels_per_categorical_feature:List[int], categorical_format:CategoricalFormat, attack_class_type_list:List[str], all_class_types_list:List[str]):
        self.feature_names = feature_names

        self.numeric_feature_names = feature_names[:n_numeric_features]
        self.categorical_feature_names = feature_names[n_numeric_features:]
        self.categorical_format:CategoricalFormat = categorical_format

        self.n_numeric_features = n_numeric_features
        self.levels_per_categorical_feature = levels_per_categorical_feature

        # add support for multiple attack class types
        self.attack_class_type_list = attack_class_type_list

        # add support for multiple class types
        self.all_class_types_list = all_class_types_list