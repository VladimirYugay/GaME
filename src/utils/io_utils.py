import json
from pathlib import Path
from typing import Union

import torch
import yaml


def read_yaml_file(file_path: Union[str, Path]) -> dict:
    """Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path: The path to the YAML file to be read.

    Returns:
        A dictionary containing the contents of the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def mkdir_decorator(func):
    """Decorator that creates the ``directory`` kwarg path before calling the function.

    Args:
        func: The function to decorate. Must accept a ``directory`` keyword
            argument that is a path-like value.

    Returns:
        The wrapped function, with ``directory`` coerced to a ``Path`` and
        created (including any missing parents) before the inner call.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        kwargs["directory"] = output_path
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_dict_to_json(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """Save a dictionary to a JSON file, creating the directory if needed.

    Args:
        dictionary: The dictionary to serialise.
        file_name: Name of the JSON file (e.g. ``"configs.json"``).
        directory: Directory in which the file will be written. Created
            automatically if it does not exist.
    """
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f)


@mkdir_decorator
def save_dict_to_ckpt(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """Save a dictionary to a PyTorch checkpoint file, creating the directory if needed.

    Args:
        dictionary: The dictionary to save (e.g. model state and metadata).
        file_name: Name of the checkpoint file (e.g. ``"checkpoint.pth"``).
        directory: Directory in which the file will be written. Created
            automatically if it does not exist.
    """
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)


def setup_output_path(output_path) -> Path:
    """Create a directory at ``output_path`` and return it as a ``Path``.

    Args:
        output_path: Target directory path (str or path-like).

    Returns:
        The resolved ``Path`` object after the directory has been created.
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    return output_path


def setup_output_paths(output_path: str, folder_names: list) -> None:
    """Create ``output_path`` and a set of named subdirectories inside it.

    Args:
        output_path: Root directory to create (including any missing parents).
        folder_names: Names of subdirectories to create under ``output_path``.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for folder_name in folder_names:
        (output_path / folder_name).mkdir(exist_ok=True)
