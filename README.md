<img src="https://img.shields.io/pypi/v/projecttalos?style=flat-square"> <img src="https://img.shields.io/pypi/pyversions/projecttalos?style=flat-square"> <img src="https://img.shields.io/pypi/l/projecttalos?style=flat-square"> <img src="https://img.shields.io/badge/status-pre--alpha-<COLOR>.svg?style=flat-square">

<p align="center">
<img src="https://user-images.githubusercontent.com/46282429/110646268-cba51200-81c7-11eb-9807-71a8509a2ec8.png">

</p>

## About ProjectTalos

* ProjectTalos is an open source Machine Learning library that still on its very early phase.
* Name of the project is inspired by the automaton named Talos in greek mythology.
* The source code and documentation are all public all help is appreciated.


## Table of contents

 * [Title / Repository Name](#title--repository-name)
   * [About](#about-projecttalos)
   * [Features](#features)
   * [Installation](#installation)
   * [Usage](#example-usage)
   * [Documentation](#documentation)
   * [License](#license)

## Installation

```
pip install projecttalos
```

## Example Usage

Let's start with importing necessary functions

![image](https://user-images.githubusercontent.com/46282429/110657317-cb117900-81d1-11eb-9b81-2e686b566a80.png)

Next step is converting images to numpy arrays and labeling them with ImagePreprocessing() (We are using very small dataset for the sake of computing power :D )

![image](https://user-images.githubusercontent.com/46282429/110658728-fe083c80-81d2-11eb-92ad-16ea309d4a2c.png)

Last step is initializing NeuralNetwork with necessary arguments and training the model then making predictions with trained parameters.

![image](https://user-images.githubusercontent.com/46282429/110659577-c4840100-81d3-11eb-89d1-54c6f044e553.png)


## Features
```
Models
├───NeuralNetwork
│   ├───train()
│   └───predict()
│   └───score()
│
├───Multiple/Linear Regression
│   ├───train()
│   └───predict()
│   └───score()
.
.
.
├───KNN,SVM,....(Coming Soon)

Data Processing
├───ImagePreprocessing()
│
.
.
.
├───Imputation,...(Coming Soon)




```

## Documentation
 (It's on the way!)
## License

[MIT License](https://github.com/Xessen/ProjectTalos/blob/main/LICENSE)
