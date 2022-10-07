# active_extrinsic

Active Extrinsic Contact Sensing & Insertion using GTSAM + Python wrapping

## Directories

- codes_real_experiments: contains the source codes to run the real insertion experiments on ABB 120 robot.
- codes_training_n_visualization: contains the source codes for the training the tactile module and the reinforcement learning policy. It also contains the codes that generates the 3D visualization with the collected data.
- gtsam-project-python: is the package that contains the custom factors required for the project. Is should be installed with GTSAM to be used.
- TD3_model: contains the trained TD3 reinforcement learning insertion policy.
- weights: contains the trained tactile module convolutional neural network.

## PREREQUISITES

- Python 3.6+ is required.
- GTSAM is required.
  - To install the wrap package via `GTSAM`:

    - Set the CMake flag `GTSAM_BUILD_PYTHON` to `ON` to enable building the Pybind11 wrapper.
    - Set the CMake flag `GTSAM_PYTHON_VERSION` to `3.x` (e.g. `3.7`), otherwise the default interpreter will be used.
    - You can do this on the command line as follows:

      ```sh
      cmake -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3.7 ..
      ```
  - Alternatively, you can install the wrap package directly from the [repo](https://github.com/borglab/wrap), but you will still need to install `GTSAM`.
- To install the custom GTSAM python package required for the project:
  - In the 'gtsam-project-python' directory, create the `build` directory and `cd` into it.
  - Run `cmake ..`.
  - Run `make`, and the wrapped module will be installed to the `python` directory in the top-level.
  - To install the wrapped module, simply run `make python-install`.
