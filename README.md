# FlowTransformer for Multi-Classification Task

## About
This is a **FlowTransformer**-based framework with the following additional features:  

1. Display the detection performance for each class in the evaluation set.  
2. Support training and testing on multi-classification tasks.  

Before diving in, please check out the [original project](https://github.com/liamdm/FlowTransformer) for an overview of how the framework operates.

## Reference Paper Information
- **Title** : `FlowTransformer: A Transformer Framework for Flow-based Network Intrusion Detection Systems`
- **Link to Paper** : [FlowTransformer Paper](https://www.sciencedirect.com/science/article/pii/S095741742303066X)

## Use
- **For binary classification** : `binary_classification_demo.ipynb`
- **For multi-classification** : `multi_classification_demo.ipynb`

## Directory Hierarchy
The list below highlights the modifications made compared to the original project :
```
|—— binary_classification_demo.ipynb  [updated]
|—— multi_classification_demo.ipynb  [updated]
|—— framework
|    |—— __init__.py
|    |—— base_classification_head.py
|    |—— base_input_encoding.py
|    |—— base_preprocessing.py
|    |—— base_sequential.py
|    |—— dataset_specification.py  [modified]
|    |—— enumerations.py
|    |—— flow_transformer_binary_classification.py  [updated]
|    |—— flow_transformer_multi_classification.py  [updated]
|    |—— flow_transformer_parameters.py
|    |—— framework_component.py
|    |—— model_input_specification.py  [modified]
|    |—— sequential_input_encoding.py
|    |—— utilities.py
|—— implementations
|    |—— __init__.py
|    |—— classification_heads.py
|    |—— input_encodings.py
|    |—— pre_processings.py
|    |—— transformers
|        |—— basic
|            |—— decoder_block.py
|            |—— encoder_block.py
|        |—— basic_transformers.py  [modified]
|        |—— named_transformers.py  [modified]
|—— main.py
|—— .gitignore
```

## License
This project is based on [FlowTransformer](https://github.com/liamdm/FlowTransformer) and licensed under the AGPL-3.0 License.
