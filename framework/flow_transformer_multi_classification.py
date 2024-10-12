#  FlowTransformer 2023 by liamdm / liam@riftcs.com
#  Modified by: cheng0719
import os
import time
import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd

import random

from framework.base_classification_head import BaseClassificationHead
from framework.base_input_encoding import BaseInputEncoding
from framework.base_preprocessing import BasePreProcessing
from framework.dataset_specification import DatasetSpecification
from framework.enumerations import EvaluationDatasetSampling, CategoricalFormat
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from framework.model_input_specification import ModelInputSpecification
from framework.utilities import get_identifier, load_feather_plus_metadata, save_feather_plus_metadata

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras import Input, Model
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

class FlowTransformer:
    retain_inmem_cache = False
    inmem_cache = None

    def  __init__(self, pre_processing:BasePreProcessing,
                  input_encoding:BaseInputEncoding,
                  sequential_model:FunctionalComponent,
                  classification_head:BaseClassificationHead,
                  params:FlowTransformerParameters,
                  rs:np.random.RandomState=None):

        self.rs = np.random.RandomState() if rs is None else rs
        self.classification_head = classification_head
        self.sequential_model = sequential_model
        self.input_encoding = input_encoding
        self.pre_processing = pre_processing
        self.parameters = params

        self.dataset_specification: Optional[DatasetSpecification] = None

        self.X = None
        self.y = None
        self.y_encoded = None

        self.training_mask = None
        self.model_input_spec: Optional[ModelInputSpecification] = None

        self.class_map = None
        self.class_num = None

        self.evaluation_percent = None

        self.experiment_key = {}

    def build_model(self, prefix:str=None):
        if prefix is None:
            prefix = ""

        if self.X is None:
            raise Exception("Please call load_dataset before calling build_model()")

        m_inputs = []
        for numeric_feature in self.model_input_spec.numeric_feature_names:
            m_input = Input((self.parameters.window_size, 1), name=f"{prefix}input_{numeric_feature}", dtype="float32")
            m_inputs.append(m_input)

        for categorical_feature_name, categorical_feature_levels in \
            zip(self.model_input_spec.categorical_feature_names, self.model_input_spec.levels_per_categorical_feature):
            m_input = Input(
                (self.parameters.window_size, 1 if self.model_input_spec.categorical_format == CategoricalFormat.Integers else categorical_feature_levels),
                name=f"{prefix}input_{categorical_feature_name}",
                dtype="int32" if self.model_input_spec.categorical_format == CategoricalFormat.Integers else "float32"
            )
            m_inputs.append(m_input)

        self.input_encoding.build(self.parameters.window_size, self.model_input_spec)
        self.sequential_model.build(self.parameters.window_size, self.model_input_spec)
        self.classification_head.build(self.parameters.window_size, self.model_input_spec)

        m_x = self.input_encoding.apply(m_inputs, prefix)

        # in case the classification head needs to add tokens at this stage
        m_x = self.classification_head.apply_before_transformer(m_x, prefix)

        m_x = self.sequential_model.apply(m_x, prefix)
        m_x = self.classification_head.apply(m_x, prefix)

        for layer_i, layer_size in enumerate(self.parameters.mlp_layer_sizes):
            m_x = Dense(layer_size, activation="relu", name=f"{prefix}classification_mlp_{layer_i}_{layer_size}")(m_x)
            m_x = Dropout(self.parameters.mlp_dropout)(m_x) if self.parameters.mlp_dropout > 0 else m_x
        
        m_x = Dense(int(self.class_num), activation="softmax", name=f"{prefix}multiclass_classification_out")(m_x)
        m = Model(m_inputs, m_x)
        #m.summary()
        return m

    def _load_preprocessed_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_folder:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000) -> Tuple[pd.DataFrame, ModelInputSpecification]:

        cache_file_path = None

        if dataset_name is None:
            raise Exception(f"Dataset name must be specified so FlowTransformer can optimise operations between subsequent calls!")

        pp_key = get_identifier(
            {
                "__preprocessing_name": self.pre_processing.name,
                **self.pre_processing.parameters
            }
        )

        local_key = get_identifier({
            "evaluation_percent": evaluation_percent,
            "numerical_filter": numerical_filter,
            "categorical_method": str(self.input_encoding.required_input_format),
            "n_rows": n_rows,
        })

        cache_key = f"{dataset_name}_{n_rows}_{pp_key}_{local_key}"

        # if FlowTransformer.retain_inmem_cache:
        #     if FlowTransformer.inmem_cache is not None and cache_key in FlowTransformer.inmem_cache:
        #         print(f"Using in-memory cached version of this pre-processed dataset. To turn off this functionality set FlowTransformer.retain_inmem_cache = False")
        #         return FlowTransformer.inmem_cache[cache_key]

        # if cache_folder is not None:
        #     cache_file_name = f"{cache_key}.feather"
        #     cache_file_path = os.path.join(cache_folder, cache_file_name)

        #     print(f"Using cache file path: {cache_file_path}")

        #     if os.path.exists(cache_file_path):
        #         print(f"Reading directly from cache {cache_file_path}...")
        #         model_input_spec: ModelInputSpecification
        #         dataset, model_input_spec = load_feather_plus_metadata(cache_file_path)
        #         return dataset, model_input_spec

        if isinstance(dataset, str):
            print(f"Attempting to read dataset from path {dataset}...")
            if dataset.lower().endswith(".feather"):
                # read as a feather file
                dataset = pd.read_feather(dataset, columns=specification.include_fields+[specification.class_column])
            elif dataset.lower().endswith(".csv"):
                dataset = pd.read_csv(dataset, nrows=n_rows if n_rows > 0 else None)
            else:
                raise Exception("Unrecognised dataset filetype!")
        elif not isinstance(dataset, pd.DataFrame):
            raise Exception("Unrecognised dataset input type, should be a path to a CSV or feather file, or a pandas dataframe!")

        assert isinstance(dataset, pd.DataFrame)

        # bulid 2 lists: attack_class_type_list and class_type_list, and exclude the benign label from the attack_class_type_list
        exclude_values = [specification.benign_label]
        unique_class_values = dataset[specification.class_column].unique()
        class_type_list = [v for v in unique_class_values]
        attack_class_type_list = [v for v in unique_class_values if v not in exclude_values]

        if 0 < n_rows < len(dataset):
            dataset = dataset.iloc[:n_rows]

        training_mask = np.ones(len(dataset),  dtype=bool)
        eval_n = int(len(dataset) * evaluation_percent)

        # We only support the EvaluationDatasetSampling.LastRows option for now
        training_mask[-eval_n:] = False

        numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        categorical_columns = specification.categorical_fields

        print(f"Set y to = {specification.class_column}")
        
        # transform the values in the class column to integers using the LabelEncoder
        label_encoder = LabelEncoder()
        y_values = dataset[specification.class_column].values
        y_encoded = label_encoder.fit_transform(y_values)
        # store the class map as a dictionary
        class_map = {index: label for index, label in enumerate(label_encoder.classes_)}
        self.class_map = class_map
        self.class_num = len(self.class_map)

        print(f"Class map: {self.class_map}")
        print(f"Class num: {self.class_num}")


        # create a new dataframe with the columns: __training, __y, __y_encoded(Integer Encoded), and the numerical and categorical columns
        new_df = None
        new_df = {
            "__training": training_mask,
            "__y": y_values,
            "__y_encoded": y_encoded
        }
        new_features = []

        print("Converting numerical columns to floats, and removing out of range values...")
        for col_name in numerical_columns:
            assert col_name in dataset.columns
            new_features.append(col_name)

            col_values = dataset[col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -numerical_filter] = 0
            col_values[col_values > numerical_filter] = 0
            col_values = col_values.astype("float32")

            if not np.all(np.isfinite(col_values)):
                raise Exception("Flow format data had non finite values after float transformation!")

            new_df[col_name] = col_values

        print(f"Applying pre-processing to numerical values")
        for i, col_name in enumerate(numerical_columns):
            print(f"[Numerical {i+1:,} / {len(numerical_columns)}] Processing numerical column {col_name}...")
            all_data = new_df[col_name]
            training_data = all_data[training_mask]

            self.pre_processing.fit_numerical(col_name, training_data)
            new_df[col_name] = self.pre_processing.transform_numerical(col_name, all_data)

        print(f"Applying pre-processing to categorical values")
        levels_per_categorical_feature = []
        for i, col_name in enumerate(categorical_columns):
            new_features.append(col_name)
            if col_name == specification.class_column:
                continue
            print(f"[Categorical {i+1:,} / {len(categorical_columns)}] Processing categorical column {col_name}...")

            all_data = dataset[col_name].values
            training_data = all_data[training_mask]

            self.pre_processing.fit_categorical(col_name, training_data)
            new_values = self.pre_processing.transform_categorical(col_name, all_data, self.input_encoding.required_input_format)

            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                # multiple columns of one hot values
                if isinstance(new_values, pd.DataFrame):
                    levels_per_categorical_feature.append(len(new_values.columns))
                    for c in new_values.columns:
                        new_df[c] = new_values[c]
                else:
                    n_one_hot_levels = new_values.shape[1]
                    levels_per_categorical_feature.append(n_one_hot_levels)
                    for z in range(n_one_hot_levels):
                        new_df[f"{col_name}_{z}"] = new_values[:, z]
            else:
                # single column of integers
                levels_per_categorical_feature.append(len(np.unique(new_values)))
                new_df[col_name] = new_values

        print(f"Generating pre-processed dataframe...")
        new_df = pd.DataFrame(new_df)
        model_input_spec = ModelInputSpecification(new_features, len(numerical_columns), levels_per_categorical_feature, self.input_encoding.required_input_format, attack_class_type_list=attack_class_type_list, all_class_types_list=class_type_list)

        print(f"Input data frame had shape ({len(dataset)},{len(dataset.columns)}), output data frame has shape ({len(new_df)},{len(new_df.columns)}) after pre-processing...")

        if cache_file_path is not None:
            print(f"Writing to cache file path: {cache_file_path}...")
            save_feather_plus_metadata(cache_file_path, new_df, model_input_spec)

        if FlowTransformer.retain_inmem_cache:
            if FlowTransformer.inmem_cache is None:
                FlowTransformer.inmem_cache = {}

            FlowTransformer.inmem_cache.clear()
            FlowTransformer.inmem_cache[cache_key] = (new_df, model_input_spec)

        return new_df, model_input_spec

    def load_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_path:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000) -> pd.DataFrame:
        """
        Load a dataset and prepare it for training

        :param dataset: The path to a CSV dataset to load from, or a dataframe
        :param cache_path: Where to store a cached version of this file
        :param n_rows: The number of rows to ingest from the dataset, or 0 to ingest all
        """

        if cache_path is None:
            cache_path = "cache"

        if not os.path.exists(cache_path):
            warnings.warn(f"Could not find cache folder: {cache_path}, attempting to create")
            os.mkdir(cache_path)

        self.dataset_specification = specification
        df, model_input_spec = self._load_preprocessed_dataset(dataset_name, dataset, specification, cache_path, n_rows, evaluation_percent, numerical_filter)

        print(df.columns)

        training_mask = df["__training"].values
        del df["__training"]

        y = df["__y"].values
        del df["__y"]

        y_encoded = df["__y_encoded"].values
        del df["__y_encoded"]

        self.X = df
        self.y = y
        self.y_encoded = y_encoded
        self.training_mask = training_mask
        self.model_input_spec = model_input_spec
        self.evaluation_percent = evaluation_percent

        return df

    def evaluate(self, m:keras.Model, batch_size, early_stopping_patience:int=5, epochs:int=100, steps_per_epoch:int=128):
        sample_num_per_class = int(batch_size / len(self.model_input_spec.all_class_types_list))
        sample_num_last_class = int(batch_size - (len(self.model_input_spec.all_class_types_list) - 1) * sample_num_per_class)

        all_class_types_list = self.model_input_spec.all_class_types_list
        
        n_malicious_per_batch = int(0.5 * batch_size)
        n_legit_per_batch = batch_size - n_malicious_per_batch

        overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
        overall_y_preserve[:n_malicious_per_batch] = 1.

        selectable_mask = np.ones(len(self.X), dtype=bool)
        training_set_size = int(len(self.X) * (1 - self.evaluation_percent))
        eval_set_start = training_set_size
        selectable_mask[:self.parameters.window_size] = False
        selectable_mask[eval_set_start:eval_set_start + self.parameters.window_size] = False

        train_mask = self.training_mask

        # extract the positive samples
        y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

        indices_train = np.argwhere(train_mask).reshape(-1)

        indices_test:np.ndarray = np.argwhere(~train_mask & selectable_mask).reshape(-1)

        # create the boolean mask for every attack class in training set
        classes_indices_train_list = {}
        for class_name in self.model_input_spec.all_class_types_list:
            mask = self.y.astype('str') == str(class_name)
            classes_indices_train_list[class_name] = np.argwhere(train_mask & mask).reshape(-1)

        # create the boolean mask for every attack class in testing set
        classes_indices_test_list = {}
        for class_name in self.model_input_spec.all_class_types_list:
            mask = self.y.astype('str') == str(class_name)
            classes_indices_test_list[class_name] = np.argwhere(~train_mask & mask).reshape(-1)
        
        def get_windows_for_indices(indices:np.ndarray, ordered) -> List[pd.DataFrame]:
            X: List[pd.DataFrame] = []
            if ordered:
                # we don't really want to include eval samples as part of context, because out of range values might be learned
                # by the model, _but_ we are forced to in the windowed approach, if users haven't just selected the
                # "take last 10%" as eval option. We warn them prior to this though.
                for idx, i1 in enumerate(indices):
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                         replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])
            
            return X

        def get_label_for_indices(indices:np.ndarray) -> np.ndarray:
            return self.y_encoded[indices]

        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)

            featurewise_X = []

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        # this is a one-hot encoded categorical feature
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                combined_values = np.array([combined_values[i:i+sequence_length] for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X

        print(f"Classes in eval set:")
        for class_name in self.model_input_spec.all_class_types_list:
            print(f"\t{class_name}")
        print(f"Building eval dataset...")
        eval_X = get_windows_for_indices(indices_test, True)
        print(f"Splitting dataset to featurewise...")
        eval_featurewise_X = samplewise_to_featurewise(eval_X)

        eval_y_multiclass = self.y_encoded[indices_test]

        print(f"Evaluation dataset is built!")

        # compute the number of each class in the evaluation set
        print(f"Computing and print the number of each class in the evaluation set...")
        eval_class_array_list = {}
        for class_name in self.model_input_spec.all_class_types_list:
            eval_class_array_list[class_name] = self.y[classes_indices_test_list[class_name]]
            print(f"Attack class {class_name} samples in eval set: {len(eval_class_array_list[class_name])}")

        epoch_results = []

        # implement the run_evaluation function of the multi-class classification
        def run_evaluation(epoch):
            pred_y = m.predict(eval_featurewise_X, verbose=True)
            pred_y = np.argmax(pred_y, axis=1)

            report = classification_report(eval_y_multiclass, pred_y, output_dict=True)
            cm = confusion_matrix(eval_y_multiclass, pred_y)

            print("\nRaw report :")
            print(report)

            class_map = self.class_map
            class_map_str = {str(k): v for k, v in class_map.items()}

            report_df = pd.DataFrame.from_dict(report).transpose()
            report_df['support'] = report_df['support'].astype(int)
            report_df = report_df.rename(index=class_map_str)
            print("\nFormatted classification report:")
            print(report_df)

            print("\nRaw confusion Matrix:")
            print(cm)

            class_names = [class_map[i] for i in range(len(class_map))]

            plt.figure(figsize=(14, 12))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()

            # display the number of samples in each cell
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    # if cm[i, j] != 0:  # if you don't want to display zeros
                    plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black", fontsize=16)
                    rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                    plt.gca().add_patch(rect)

            # set the labels
            plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
            plt.yticks(np.arange(len(class_names)), class_names)

            plt.title("Confusion Matrix Visualization")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.tight_layout()

            plt.savefig("confusion_matrix.jpg", format="jpg", dpi=300)
            plt.show()


        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.sample_cursor_per_class = {all_class_types_list[i]: 0 for i in range(len(all_class_types_list))}
                self.rs = rs

            def get_batch(self):
                classes_indices_batch = {}
                for i in range(len(all_class_types_list) - 1): # exclude the last class
                    class_name = all_class_types_list[i]
                    class_indices_train = classes_indices_train_list[class_name]
                    classes_indices_batch[class_name] = self.rs.choice(class_indices_train, size=sample_num_per_class, replace=True) \
                        if self.random else \
                        class_indices_train[self.sample_cursor_per_class[class_name]:self.sample_cursor_per_class[class_name] + sample_num_per_class]
                classes_indices_batch[all_class_types_list[-1]] = self.rs.choice(classes_indices_train_list[all_class_types_list[-1]], size=sample_num_last_class, replace=True) \
                    if self.random else \
                    classes_indices_train_list[all_class_types_list[-1]][self.sample_cursor_last_class:self.sample_cursor_last_class + sample_num_last_class]
                

                indices = np.concatenate([v for v in classes_indices_batch.values()])

                for i in range(len(all_class_types_list)):
                    class_name = all_class_types_list[i]
                    if class_name != all_class_types_list[-1]:
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] + sample_num_per_class
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] % (len(classes_indices_train_list[class_name]) - sample_num_per_class)
                    else:
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] + sample_num_last_class
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] % (len(classes_indices_train_list[class_name]) - sample_num_last_class)

                X = get_windows_for_indices(indices, self.ordered)
                y_integer_encoded = get_label_for_indices(indices)
                # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

                # we have a dataframe containing batch_size x (window_size, features)
                # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
                featurewise_X = samplewise_to_featurewise(X)

                return featurewise_X, y_integer_encoded  # batch_y should be the values in self.y_encoded (results after Integer Encoding)

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows, not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0

        train_results = []
        final_epoch = 0

        last_print = time.time()
        elapsed_time = 0

        for epoch in range(epochs):
            final_epoch = epoch

            has_reduced_loss = False
            for step in range(steps_per_epoch):
                batch_X, batch_y = batch_yielder.get_batch()

                t0 = time.time()
                batch_results = m.train_on_batch(batch_X, batch_y)
                t1 = time.time()

                if epoch > 0 or step > 0:
                    elapsed_time += (t1 - t0)
                    if epoch == 0 and step == 1:
                        # include time for last "step" that we skipped with step > 0 for epoch == 0
                        elapsed_time *= 2

                train_results.append(batch_results + [elapsed_time, epoch])

                batch_loss = batch_results[0] if isinstance(batch_results, list) else batch_results

                if time.time() - last_print > 3:
                    last_print = time.time()
                    early_stop_phrase = "" if early_stopping_patience <= 0 else f" (early stop in {early_stopping_patience - iters_since_loss_decrease:,})"
                    print(f"Epoch = {epoch:,} / {epochs:,}{early_stop_phrase}, step = {step}, loss = {batch_loss:.5f}, results = {batch_results} -- elapsed (train): {elapsed_time:.2f}s")

                if batch_loss < min_loss:
                    has_reduced_loss = True
                    min_loss = batch_loss

            if has_reduced_loss:
                iters_since_loss_decrease = 0
            else:
                iters_since_loss_decrease += 1

            do_early_stop = early_stopping_patience > 0 and iters_since_loss_decrease > early_stopping_patience
            is_last_epoch = epoch == epochs - 1
            run_eval = epoch in [6] or is_last_epoch or do_early_stop

            if run_eval:
                run_evaluation(epoch)

            if do_early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        eval_results = pd.DataFrame(epoch_results)

        return (train_results, eval_results, final_epoch)


    def time(self, m:keras.Model, batch_size, n_steps=128, n_repeats=4):
        sample_num_per_class = int(batch_size / len(self.model_input_spec.all_class_types_list))
        sample_num_last_class = int(batch_size - (len(self.model_input_spec.all_class_types_list) - 1) * sample_num_per_class)

        all_class_types_list = self.model_input_spec.all_class_types_list
        
        selectable_mask = np.ones(len(self.X), dtype=bool)
        training_set_size = int(len(self.X) * (1 - self.evaluation_percent))
        eval_set_start = training_set_size
        selectable_mask[:self.parameters.window_size] = False
        selectable_mask[eval_set_start:eval_set_start + self.parameters.window_size] = False

        train_mask = self.training_mask

        y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

        indices_train = np.argwhere(train_mask).reshape(-1)
        classes_indices_train_list = {}
        for attack_class in self.model_input_spec.all_class_types_list:
            mask = self.y.astype('str') == str(attack_class)
            classes_indices_train_list[attack_class] = np.argwhere(train_mask & mask).reshape(-1)
        

        indices_test:np.ndarray = np.argwhere(~train_mask).reshape(-1)

        def get_windows_for_indices(indices:np.ndarray, ordered) -> List[pd.DataFrame]:
            X: List[pd.DataFrame] = []

            if ordered:
                # we don't really want to include eval samples as part of context, because out of range values might be learned
                # by the model, _but_ we are forced to in the windowed approach, if users haven't just selected the
                # "take last 10%" as eval option. We warn them prior to this though.
                for i1 in indices:
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                         replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])

            return X

        def get_label_for_indices(indices:np.ndarray) -> np.ndarray:
            return self.y_encoded[indices]

        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)

            featurewise_X = []

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        # this is a one-hot encoded categorical feature
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                # maybe this can be faster with a reshape but I couldn't get it to work
                combined_values = np.array([combined_values[i:i+sequence_length] for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X


        epoch_results = []


        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.sample_cursor_per_class = {all_class_types_list[i]: 0 for i in range(len(all_class_types_list))}
                self.rs = rs

            def get_batch(self):
                classes_indices_batch = {}
                for i in range(len(all_class_types_list) - 1): # exclude the last class
                    class_name = all_class_types_list[i]
                    class_indices_train = classes_indices_train_list[class_name]
                    classes_indices_batch[class_name] = self.rs.choice(class_indices_train, size=sample_num_per_class, replace=True) \
                        if self.random else \
                        class_indices_train[self.sample_cursor_per_class[class_name]:self.sample_cursor_per_class[class_name] + sample_num_per_class]
                classes_indices_batch[all_class_types_list[-1]] = self.rs.choice(classes_indices_train_list[all_class_types_list[-1]], size=sample_num_last_class, replace=True) \
                    if self.random else \
                    classes_indices_train_list[all_class_types_list[-1]][self.sample_cursor_last_class:self.sample_cursor_last_class + sample_num_last_class]
                

                indices = np.concatenate([v for v in classes_indices_batch.values()])
                
                for i in range(len(all_class_types_list)):
                    class_name = all_class_types_list[i]
                    if class_name != all_class_types_list[-1]:
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] + sample_num_per_class
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] % (len(classes_indices_train_list[class_name]) - sample_num_per_class)
                    else:
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] + sample_num_last_class
                        self.sample_cursor_per_class[class_name] = self.sample_cursor_per_class[class_name] % (len(classes_indices_train_list[class_name]) - sample_num_last_class)

                X = get_windows_for_indices(indices, self.ordered)
                y_integer_encoded = get_label_for_indices(indices)
                # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

                # we have a dataframe containing batch_size x (window_size, features)
                # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
                featurewise_X = samplewise_to_featurewise(X)

                return featurewise_X, y_integer_encoded  # batch_y should be the values in self.y_encoded(results after Integer Encoding)

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows, not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0

        final_epoch = 0

        last_print = time.time()
        elapsed_time = 0

        batch_times = []


        for step in range(n_steps):
            batch_X, batch_y = batch_yielder.get_batch()

            local_batch_times = []
            for i in range(n_repeats):
                t0 = time.time()
                batch_results = m.predict_on_batch(batch_X)
                t1 = time.time()
                local_batch_times.append(t1 - t0)

            batch_times.append(local_batch_times)

            if time.time() - last_print > 3:
                last_print = time.time()
                print(f"Step = {step}, running model evaluation... Average times = {np.mean(np.array(batch_times).reshape(-1))}")

        return batch_times


