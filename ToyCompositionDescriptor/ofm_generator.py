"""
The original code is from Prof. Dam.

python poscar2ofm.py number_cores is_ofm1 is_including_d dir_1 dir_2 ...
"""

from typing import Union
from pymatgen.core import Element, Structure

# import matplotlib.pyplot as plt
# from pymatgen.io.cif import CifParser
import copy
import re
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from numpy.typing import ArrayLike
import pandas as pd


GENERAL_ELECTRON_SUBSHELLS = [
    "s1",
    "s2",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "d8",
    "d9",
    "d10",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
]

ROW = ["row1", "row2", "row3", "row4", "row5", "row6", "row7"]


IS_OFM1 = True
IS_ADDING_ROW = False
IS_INCLUDING_D = True


class ofmColumns:
    """
    use ofmColumns to get columns of OFM.
    """

    def __init__(self, is_adding_row=IS_ADDING_ROW):
        _general_electron_subshells = GENERAL_ELECTRON_SUBSHELLS.copy()
        if is_adding_row:
            _general_electron_subshells.extend(ROW)
        self._general_electron_subshells = _general_electron_subshells

    @property
    def general_electron_subshells(self):
        return self._general_electron_subshells


import warnings
import functools


def deprecated(func):
    """Decorator to mark functions as deprecated."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated and will be removed in a future version.", category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper


def _obtain_ofm_1d_columns(is_ofm1=IS_OFM1, is_adding_row=IS_ADDING_ROW):
    if is_ofm1:
        v = [
            "C",
        ]
    else:
        v = []
    general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells
    v.extend(general_electron_subshells)
    h = copy.deepcopy(general_electron_subshells)
    vh = []

    for v1 in v:
        for h1 in h:
            vh.append("_".join(["ofm", v1, h1]))
    return vh


@deprecated
def obtain_ofm_1d_columns(is_ofm1=IS_OFM1, is_adding_row=IS_ADDING_ROW):
    return _obtain_ofm_1d_columns(is_ofm1, is_adding_row)


def obtain_df_ofm_1d(v: ArrayLike = None, is_ofm1=IS_OFM1, is_adding_row=IS_ADDING_ROW):
    if v is None:
        general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells
        n = len(general_electron_subshells)
        if is_ofm1:
            v = np.zeros(n * (n + 1))
        else:
            v = np.zeros(n * n)
    _obtain_df_ofm_1d = pd.DataFrame(v, index=_obtain_ofm_1d_columns(is_ofm1, is_adding_row)).T
    return _obtain_df_ofm_1d


def obtain_df_ofm_2d(v: ArrayLike = None, is_ofm1=IS_OFM1, is_adding_row=IS_ADDING_ROW):
    if is_ofm1:
        index = ["C"]
    else:
        index = []
    general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells
    index.extend(general_electron_subshells)
    n1 = len(index)
    n2 = len(general_electron_subshells)
    if v is None:
        v = np.zeros((n1, n2))
    else:
        v = v.reshape((n1, n2))
    _df = pd.DataFrame(v, columns=general_electron_subshells, index=index)
    return _df


def get_element_representation(name="Si", is_adding_row=IS_ADDING_ROW):
    """
    generate one-hot representation for a element, e.g, si = [0.0, 1.0, 0.0, 0.0, ...]
    :param name: element symbol
    """
    element = Element(name)

    general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells

    general_element_electronic = {}
    for _name in general_electron_subshells:
        general_element_electronic[_name] = 0.0

    if name == "H":
        element_electronic_structure = ["s1"]
    elif name == "He":
        element_electronic_structure = ["s2"]
    else:
        # element_electronic_structure = [''.join(pair) for pair in re.findall("\.\d(\w+)<sup>(\d+)</sup>",
        #                                                                     element.electronic_structure)]
        element_electronic_structure = ["".join(pair) for pair in re.findall(r"\.\d(\w+)(\d+)", element.electronic_structure)]

    for eletron_subshell in element_electronic_structure:
        general_element_electronic[eletron_subshell] = 1.0
    if is_adding_row:
        irow = element.row
        general_element_electronic[ROW[irow - 1]] = 1.0

    v = np.array([general_element_electronic[key] for key in general_electron_subshells])
    return v
    # return pd.DataFrame([v], columns=general_electron_subshells)


def ofm(struct: Structure, is_ofm1=IS_OFM1, is_including_d=True, is_adding_row=IS_ADDING_ROW):
    atoms = np.array([site.species_string for site in struct])

    # local_xyz = []
    local_orbital_field_matrices = []
    general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells
    N = len(general_electron_subshells)
    for i_atom, atom in enumerate(atoms):

        coordinator_finder = VoronoiNN(cutoff=10.0)
        neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

        site = struct[i_atom]
        center_vector = get_element_representation(atom, is_adding_row=is_adding_row)
        env_vector = np.zeros(N)

        # atom_xyz = [atom]
        # coords_xyz = [site.coords]

        for nn in neighbors:

            site_x = nn["site"]
            w = nn["weight"]
            site_x_label = site_x.species_string
            # atom_xyz += [site_x_label]
            # coords_xyz += [site_x.coords]
            neigh_vector = get_element_representation(site_x_label, is_adding_row=is_adding_row)

            if is_including_d:
                d = np.sqrt(np.sum((site.coords - site_x.coords) ** 2))
                env_vector += neigh_vector * w / d
            else:
                env_vector += neigh_vector * w

        if is_ofm1:
            env_vector = np.concatenate(([1.0], env_vector))

        local_matrix = center_vector[None, :] * env_vector[:, None]

        local_matrix = np.ravel(local_matrix)  # ravel to make 2024-Dimensional vector
        local_orbital_field_matrices.append(local_matrix)
        # local_xyz.append({"atoms": np.array(atom_xyz), "coords": np.array(coords_xyz)})

    local_orbital_field_matrices = np.array(local_orbital_field_matrices)
    material_descriptor = np.mean(local_orbital_field_matrices, axis=0)

    return {
        "mean": material_descriptor,
        "locals": local_orbital_field_matrices,
        "atoms": atoms,
        # "local_xyz": local_xyz,
        "cif": struct.to(fmt="cif", filename=None),
    }


def ofm_alloy(struct: Structure, is_ofm1=IS_OFM1, is_including_d=True, is_adding_row=IS_ADDING_ROW):

    # local_xyz = []
    local_orbital_field_matrices = []
    general_electron_subshells = ofmColumns(is_adding_row=is_adding_row).general_electron_subshells
    N = len(general_electron_subshells)
    # for i_atom, atom in enumerate(atoms):
    for i_atom, site in enumerate(struct):

        coordinator_finder = VoronoiNN(cutoff=10.0)
        neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

        center_vector = np.zeros(N)
        for specie, frac in site.species.as_dict().items():
            v = get_element_representation(specie, is_adding_row=is_adding_row)
            center_vector += v * frac

        env_vector = np.zeros(N)

        # atom_xyz = [atom]
        # coords_xyz = [site.coords]

        for nn in neighbors:

            site_x = nn["site"]
            w = nn["weight"]
            # site_x_label = site_x.species_string
            # atom_xyz += [site_x_label]
            # coords_xyz += [site_x.coords]
            neigh_vector = np.zeros(N)
            for nn_specie, nn_frac in site_x.species.as_dict().items():
                neigh_vector += get_element_representation(nn_specie, is_adding_row=is_adding_row) * nn_frac

            d = np.sqrt(np.sum((site.coords - site_x.coords) ** 2))
            if is_including_d:
                env_vector += neigh_vector * w / d
            else:
                env_vector += neigh_vector * w

        if is_ofm1:
            env_vector = np.concatenate(([1.0], env_vector))

        local_matrix = center_vector[None, :] * env_vector[:, None]

        local_matrix = np.ravel(local_matrix)  # ravel to make N*N- or N*(N+1)-Dimensional vector
        local_orbital_field_matrices.append(local_matrix)
        # local_xyz.append({"atoms": np.array(atom_xyz), "coords": np.array(coords_xyz)})

    local_orbital_field_matrices = np.array(local_orbital_field_matrices)
    material_descriptor = np.mean(local_orbital_field_matrices, axis=0)

    atoms = [site.species.as_dict() for site in struct.sites]
    return {
        "mean": material_descriptor,
        "locals": local_orbital_field_matrices,
        "atoms": atoms,
        # "local_xyz": local_xyz,
        "cif": struct.to(fmt="cif", filename=None),
        "is_ofm1": is_ofm1,
        "is_including_d": is_including_d,
        "is_adding_row": is_adding_row,
    }


class OFMFeatureRepresentation:
    """
    Class for converting OFM descriptor results into various representations.

    Attributes:
        KEYS (list of str): Allowed keys for accessing different types of descriptor results.
        result (dict): Dictionary containing OFM descriptor results.
    """

    KEYS = ["mean", "locals"]

    def __init__(self, result: dict, is_ofm1, is_adding_row, is_including_d, check_strictly=True):
        """
        Initialize an OFMFeatureRepresentation object.

        Args:
            result (dict): Dictionary containing OFM descriptor results.
            is_ofm1 (bool): OFM parameter
            is_adding_row (bool): OFM parameter
            is_including_d (bool): OFM parameter
            check_strictly (bool, optional):  check the keys of `result` stricutly.

        Raises:
            RuntimeError: If the result does not have the required keys.
        """
        self.result = None
        self.is_ofm1 = is_ofm1
        self.is_adding_row = is_adding_row
        self.is_including_d = is_including_d
        if check_strictly:
            dict_keys = list(result.keys())
            flags = ["mean" in dict_keys, "locals" in dict_keys, "cif" in dict_keys]
            if np.all(flags):
                self.result = result
            else:
                raise RuntimeError("result must be dict type with mean, locals, atoms and cif keys")

    @property
    def columns_1d(self):
        is_ofm1 = self.is_ofm1
        is_adding_row = self.is_adding_row
        return _obtain_ofm_1d_columns(is_ofm1=is_ofm1, is_adding_row=is_adding_row)

    def validate_key(self, key):
        """
        Validate the provided key.

        Args:
            key (str): Key to be validated.

        Returns:
            bool: True if the key is valid, False otherwise.

        Raises:
            ValueError: If the provided key is not in the allowed KEYS.
        """
        if key in self.KEYS:
            return True
        else:
            raise ValueError("key must be mean or locals.")

    def as_1d_array(self, key: str):
        """
        Convert the descriptor result to a 1D array.

        Args:
            key (str): The type of descriptor result to be converted.

        Returns:
            np.ndarray: 1D array representation of the descriptor result.

        Raises:
            ValueError: If the provided key is not in the allowed KEYS.
        """
        self.validate_key(key)
        return self.result[key]

    def as_2d_array(self, key: str):
        """
        Convert the descriptor result to a 2D array.

        Args:
            key (str): The type of descriptor result to be converted.

        Returns:
            np.ndarray: 2D array representation of the descriptor result.

        Raises:
            ValueError: If the provided key is not in the allowed KEYS.
        """
        self.validate_key(key)
        general_electron_subshells = ofmColumns(is_adding_row=self.is_adding_row).general_electron_subshells
        n = len(general_electron_subshells)
        v = self.result[key]
        if self.is_ofm1:
            shape = (n + 1, n)
        else:
            shape = (n, n)
        if key == "mean":
            v = v.reshape(shape[0], shape[1])
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v1 = v1.reshape(shape[0], shape[1])
                v.append(v1)
        return v

    def as_1d_df(self, key: str):
        """
        Converts the stored OFM data of a specific representation type to a 1D pandas DataFrame.

        Parameters:
            key (str): The type of OFM representation, such as "mean", "locals", etc.

        Returns:
            DataFrame: A 1D pandas DataFrame with the chosen representation.

        Raises:
            ValueError: If the provided key is not in the allowed KEYS.
        """
        self.validate_key(key)
        if key == "mean":
            return obtain_df_ofm_1d(self.result[key], self.is_ofm1, self.is_adding_row)
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v.append(obtain_df_ofm_1d(v1, self.is_ofm1, self.is_adding_row))
            return v

    def as_2d_df(self, key: str):
        """
        Converts the stored OFM data of a specific representation type to a 2D pandas DataFrame.

        Parameters:
            key (str): The type of OFM representation, such as "mean", "locals"

        Returns:
            DataFrame: A 2D pandas DataFrame with the chosen representation.

        Raises:
            ValueError: If the provided key is not in the allowed KEYS.
        """
        self.validate_key(key)
        if key == "mean":
            return obtain_df_ofm_2d(self.result[key], self.is_ofm1, self.is_adding_row)
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v.append(obtain_df_ofm_2d(v1, self.is_ofm1, self.is_adding_row))
            return v


class OFMGenerator:
    """
    Initializes the OFMGenerator.

    Parameters:
        representation (str, optional): Type of data representation desired. It can be either 'mean',
                                        'locals', or None. Default is None.
        is_ofm1 (bool, optional): Determines if the OFM1 variant of the descriptor should be used.
                                    Default is True.
        is_including_d (bool, optional): Determines if distances between atoms should be included in
                                            the descriptor. Default is True.
    """

    def __init__(
        self,
        representation: Union[str, None] = None,
        is_ofm1=IS_OFM1,
        is_including_d=IS_INCLUDING_D,
        is_adding_row=IS_ADDING_ROW,
    ):
        """
        Initializes the OFMGenerator.

        Parameters:
            representation (str, optional): Type of data representation desired. It can be either 'mean',
                                            'locals', or None. Default is None.
            is_ofm1 (bool, optional): Determines if the OFM1 variant of the descriptor should be used.
                                       Default is True.
            is_including_d (bool, optional): Determines if distances between atoms should be included in
                                               the descriptor. Default is True.
            is_adding_row (bool, optiona): Determines if the row variables are in the descriptor. Default is True.
            is_ofm1 (bool, optional): OFM parameter. Default is True.
            is_including_d (bool, optional): OFM parameter. Default is True.
            is_adding_row (bool, optional): OFM parameter. Default is False.
        """
        self.is_ofm1 = is_ofm1
        self.is_including_d = is_including_d
        self.is_adding_row = is_adding_row
        self.representation = representation

    def transform(self, struct: Structure):
        """
        Transforms a pymatgen Structure object into its OFM representation.

        Parameters:
            struct (Structure): The pymatgen Structure object representing the material structure.

        Returns:
            OFMFeatureRepresentation: An object encapsulating the OFM representation of the material.
        """
        result = ofm_alloy(
            struct,
            is_ofm1=self.is_ofm1,
            is_including_d=self.is_including_d,
            is_adding_row=self.is_adding_row,
        )
        return OFMFeatureRepresentation(
            result,
            is_ofm1=self.is_ofm1,
            is_including_d=self.is_including_d,
            is_adding_row=self.is_adding_row,
        )
