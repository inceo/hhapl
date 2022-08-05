**[Requirements](#requirements)** |
**[Running](#running)** |
**[Inspiration](#inspiration)** |
**[License](#license)** 


<img align="right" width="10%" height="10%" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Andrew_Fielding_Huxley_nobel.jpg" alt="Andrew Huxley">
<img align="right" width="10%" height="10%" src="https://upload.wikimedia.org/wikipedia/commons/0/07/Alan_Lloyd_Hodgkin_nobel.jpg" alt="Andrew Huxley">

# [Hodgkin-Huxley action potential lab](https://github.com/inceo/hhapl)

Educational web application for the purpose of learning how action potentials are initiated based on the Hodgkins and Huxley model.

The following plots are generated 'on the fly' based on the chosen inputs. The calculations behind this follow the famous Hodgkins and Huxley model. In 1952 __Alan Hodgkin__ and __Andrew Huxley__ formulated the model to explain the ionic mechanisms underlying the initiation of action potentials in neurons. They received the Nobel Prize in Physiology or Medicine for this work (1963).

This application was developed as a term project for the Scientific Python course (summer '22) of the Cognitive Science program at Osnabrück University by Christian Rohde.

## Features

- standalone web application
- interactive figures controlled via user input through widgets
- efficient background calculations
- topics
  - sodium channel activation and inactivation
  - voltage dependence potassium channel
  - phases of action potential
  - chemical manipulation of ion channels
  - refractory periods
  - spike trains
  - parameter dependence of the Hodgkins and Huxley model

## Requirements

- This project was developed and tested with Python 3.10 on a linux machine
- It should however also work on other operation systems with the requirements installed as described
- Installation via pip:
  ```shell
  pip install -r requirements.txt
  ```
- Installation in a conda environment:
  ```
  conda env create -f environment.yml
  ```

## Running

### As a standalone tornado application

Option 1: [Voila](https://github.com/inceo/hhapl) (recommended)
```
voila notebook.ipynb
```
For more command line options (e.g., to specify an alternate port number),
run `voila --help`.

### Jupyter `notebook` or `jupyterlab`

Option 2: The project notebook can also opened in a Jupyter [notebook](https://github.com/jupyter/notebook) (version >=4.2) or in [JupyterLab](https://github.com/jupyterlab/jupyterlab). 

## Inspiration

A good source that also provided an inspiration to this project can be found [here](https://nba.uth.tmc.edu/neuroscience/m/s1/chapter02.html).

## License

This project is publisted under the MIT License.

A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.
