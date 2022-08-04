**[Requirements](#requirements)** |
**[Running](#running)** |
**[Inspiration](#inspiration)** |
**[License](#license)** 


<img align="right" width="10%" height="10%" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Andrew_Fielding_Huxley_nobel.jpg" alt="Andrew Huxley">
<img align="right" width="10%" height="10%" src="https://upload.wikimedia.org/wikipedia/commons/0/07/Alan_Lloyd_Hodgkin_nobel.jpg" alt="Andrew Huxley">

# [Hodgkin-Huxley action potential lab](http://inceo.github.com/hhapl)

Educational web application for the purpose of learning how action potentials are initiated based on the Hodgkins and Huxley model.

## Requirements

- Virtual environment via conda:
  ```shell
  conda env create -f environment.yml
  ```

- Installation via pip:
  ```shell
  pip install -r requirements.txt
  ```

## Running

### As a standalone tornado application

- Option 1: [Voila](https://github.com/inceo/hhapl)
```
voila notebook.ipynb
```

For more command line options (e.g., to specify an alternate port number),
run `voila --help`.

### As a server extension to `notebook` or `jupyter_server`

Voilà can also be used as a Jupyter server extension, both with the
[notebook](https://github.com/jupyter/notebook) server or with
[jupyter_server](https://github.com/jupyter/jupyter_server).

To install the Jupyter server extension, run

```
jupyter serverextension enable voila
jupyter server extension enable voila
```

## Inspiration

https://nba.uth.tmc.edu/neuroscience/m/s1/chapter02.html

## License

TODO
