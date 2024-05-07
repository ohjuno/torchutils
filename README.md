# Torchutils

The [torchutils](torchutils) package contains features that are useful but not officially supported by [pytorch](https://pytorch.org/).

## Installation

### Binaries

This package is not yet supported for binary installation.

### Building from Source

You could use [flit](https://flit.pypa.io/en/stable) and [pyproject.toml](pyproject.toml) to build and install package.

1. First, make sure `flit` is installed in your environment.
    ```shell
    $ pip install flit
    ```

2. Clone this repository.
    ```shell
    $ git clone https://github.com/ohjuno/torchutils.git
    ```

3. Dive into the directory and then build with the following command:
    ```shell
    $ cd torchutils && flit build --format wheel
    ```

4. Install package.
    ```shell
    $ pip install dist/torchutils-[version]-py3-non-any.whl
    ```

## Ops

- Softargmax (1D/2D/3D)

## License

Unless noted inside the file, all code is released under the MIT license (see [LICENSE](LICENSE) for details).
