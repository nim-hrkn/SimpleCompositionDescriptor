
"""
The original code is from Prof. Dam.

python poscar2ofm.py number_cores is_ofm1 is_including_d dir_1 dir_2 ...
"""

from typing import Union
from pymatgen.core import Element, Structure
# import matplotlib.pyplot as plt
# from pymatgen.io.cif import CifParser
from itertools import product
import copy
import re
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from numpy.typing import ArrayLike
import pandas as pd


GENERAL_ELECTRON_SUBSHELLS = ['s1', 's2', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                              'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10',
                              'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                              'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']


def obtain_ofm_1d_columns(is_ofm1=True):
    if is_ofm1:
        v = ["C",]
    else:
        v = []
    v.extend(GENERAL_ELECTRON_SUBSHELLS)
    h = copy.deepcopy(GENERAL_ELECTRON_SUBSHELLS)
    vh = []

    for v1 in v:
        for h1 in h:
            vh.append("_".join([v1, h1]))
    return vh


def obtain_df_ofm_1d(v: ArrayLike):
    _obtain_df_ofm_1d = pd.DataFrame(v, index=obtain_ofm_1d_columns()).T
    return _obtain_df_ofm_1d


def obtain_df_ofm_2d(v: ArrayLike):
    n = len(GENERAL_ELECTRON_SUBSHELLS)
    columns = ["C"]
    columns.extend(GENERAL_ELECTRON_SUBSHELLS)
    v = v.reshape(n + 1, n)
    _df = pd.DataFrame(v, columns=GENERAL_ELECTRON_SUBSHELLS, index=columns)
    return _df


def get_element_representation(name='Si'):
    """
    generate one-hot representation for a element, e.g, si = [0.0, 1.0, 0.0, 0.0, ...]
    :param name: element symbol
    """
    element = Element(name)
    general_electron_subshells = GENERAL_ELECTRON_SUBSHELLS
    general_element_electronic = {}
    for _name in general_electron_subshells:
        general_element_electronic[_name] = 0.0

    if name == 'H':
        element_electronic_structure = ['s1']
    elif name == 'He':
        element_electronic_structure = ['s2']
    else:
        # element_electronic_structure = [''.join(pair) for pair in re.findall("\.\d(\w+)<sup>(\d+)</sup>",
        #                                                                     element.electronic_structure)]
        element_electronic_structure = [''.join(pair) for pair in re.findall(r"\.\d(\w+)(\d+)",
                                                                             element.electronic_structure)]

    for eletron_subshell in element_electronic_structure:
        general_element_electronic[eletron_subshell] = 1.0
    return np.array([general_element_electronic[key] for key in general_electron_subshells])


def ofm(struct: Structure, is_ofm1=True, is_including_d=True,):
    atoms = np.array([site.species_string for site in struct])

    # local_xyz = []
    local_orbital_field_matrices = []
    N = len(GENERAL_ELECTRON_SUBSHELLS)
    for i_atom, atom in enumerate(atoms):

        coordinator_finder = VoronoiNN(cutoff=10.0)
        neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

        site = struct[i_atom]
        center_vector = get_element_representation(atom)
        env_vector = np.zeros(N)

        # atom_xyz = [atom]
        # coords_xyz = [site.coords]

        for nn in neighbors:

            site_x = nn['site']
            w = nn['weight']
            site_x_label = site_x.species_string
            # atom_xyz += [site_x_label]
            # coords_xyz += [site_x.coords]
            neigh_vector = get_element_representation(site_x_label)
            d = np.sqrt(np.sum((site.coords - site_x.coords)**2))
            if is_including_d:
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

    return {'mean': material_descriptor,
            'locals': local_orbital_field_matrices,
            'atoms': atoms,
            # "local_xyz": local_xyz,
            'cif': struct.to(fmt='cif', filename=None)}


def ofm_alloy(struct: Structure, is_ofm1=True, is_including_d=True,):

    # local_xyz = []
    local_orbital_field_matrices = []
    N = len(GENERAL_ELECTRON_SUBSHELLS)
    # for i_atom, atom in enumerate(atoms):
    for i_atom, site in enumerate(struct):

        coordinator_finder = VoronoiNN(cutoff=10.0)
        neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

        center_vector = np.zeros(N)
        for specie, frac in site.species.as_dict().items():
            v = get_element_representation(specie)
            center_vector += v * frac

        env_vector = np.zeros(N)

        # atom_xyz = [atom]
        # coords_xyz = [site.coords]

        for nn in neighbors:

            site_x = nn['site']
            w = nn['weight']
            # site_x_label = site_x.species_string
            # atom_xyz += [site_x_label]
            # coords_xyz += [site_x.coords]
            neigh_vector = np.zeros(N)
            for nn_specie, nn_frac in site_x.species.as_dict().items():
                neigh_vector += get_element_representation(nn_specie) * nn_frac

            d = np.sqrt(np.sum((site.coords - site_x.coords)**2))
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
    return {'mean': material_descriptor,
            'locals': local_orbital_field_matrices,
            'atoms': atoms,
            # "local_xyz": local_xyz,
            'cif': struct.to(fmt='cif', filename=None)}


if False:
    def _obtain_sub_structures(structure):

        lattice = structure.lattice
        frac_coords_list = [site.frac_coords for site in structure.sites]

        species_dicts = [site.species.as_dict() for site in structure.sites]

        combinations = product(*[d.items() for d in species_dicts])
        for combo in combinations:
            elements = [key[0] for key in combo]
            values = [key[1] for key in combo]
            fraction = np.product(values)
            sub_structure = Structure(lattice, elements, frac_coords_list)
            yield (fraction, sub_structure)

if False:
    def ofm_alloy(struct: Structure, is_ofm1=True, is_including_d=True):
        structure = struct
        print("ofm_alloy", structure)

        ofm_mean_weighted_sum = None
        ofm_locals_weighted_sum = None

        N = 0
        for _ in _obtain_sub_structures(structure):
            N += 1

        for index, (weight, sub_structure) in enumerate(_obtain_sub_structures(structure)):
            print(f"sub structure {index+1}/{N}")
            test = ofm(struct=sub_structure, is_ofm1=is_ofm1, is_including_d=is_including_d)
            # result_list.append((weight, test))
            if ofm_mean_weighted_sum is None:
                ofm_mean_weighted_sum = weight * test["mean"]
            else:
                ofm_mean_weighted_sum += weight * test["mean"]

            if ofm_locals_weighted_sum is None:
                ofm_locals_weighted_sum = weight * test["locals"]
            else:
                ofm_locals_weighted_sum += weight * test["locals"]

        atoms = [site.species.as_dict() for site in structure.sites]
        cif_file = struct.to(fmt='cif', filename=None)
        result = {'mean': ofm_mean_weighted_sum, 'locals': ofm_locals_weighted_sum,
                  'atoms': atoms, 'cif': cif_file}
        return result


class OFMFeatureRepresentation:
    """
    Class for converting OFM descriptor results into various representations.

    Attributes:
        KEYS (list of str): Allowed keys for accessing different types of descriptor results.
        result (dict): Dictionary containing OFM descriptor results.
    """

    KEYS = ["mean", "locals"]

    def __init__(self, result: dict):
        """
        Initialize an OFMFeatureRepresentation object.

        Args:
            result (dict): Dictionary containing OFM descriptor results.

        Raises:
            RuntimeError: If the result does not have the required keys.
        """
        dict_keys = list(result.keys())
        flags = ['mean' in dict_keys, 'locals' in dict_keys, 'cif' in dict_keys]
        if np.all(flags):
            self.result = result
        else:
            raise RuntimeError('result must be dict type with mean, locals, atoms and cif keys')

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
            raise ValueError('key must be mean or locals.')

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

        n = len(GENERAL_ELECTRON_SUBSHELLS)
        v = self.result[key]
        if key == "mean":
            v = v.reshape(n + 1, n)
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v1 = v1.reshape(n + 1, n)
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
            return obtain_df_ofm_1d(self.result[key])
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v.append(obtain_df_ofm_1d(v1))
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
            return obtain_df_ofm_2d(self.result[key])
        elif key == "locals":
            v = []
            for v1 in self.result[key]:
                v.append(obtain_df_ofm_2d(v1))
            return v


class OFMGenerator:
    """
    Initializes the OFMGenerator.

    Parameters:
        representation (str, optional): Type of data representation desired. It can be either 'mean',
                                        'locals', or None. Default is None.
        use_ofm1 (bool, optional): Determines if the OFM1 variant of the descriptor should be used.
                                    Default is True.
        include_distance (bool, optional): Determines if distances between atoms should be included in
                                            the descriptor. Default is True.
    """

    def __init__(self, representation: Union[str, None] = None, use_ofm1=True, include_distance=True):
        """
        Initializes the OFMGenerator.

        Parameters:
            representation (str, optional): Type of data representation desired. It can be either 'mean',
                                            'locals', or None. Default is None.
            use_ofm1 (bool, optional): Determines if the OFM1 variant of the descriptor should be used.
                                       Default is True.
            include_distance (bool, optional): Determines if distances between atoms should be included in
                                               the descriptor. Default is True.
        """
        self.use_ofm1 = use_ofm1
        self.include_distance = include_distance
        self.representation = representation

    def transform(self, struct: Structure):
        """
        Transforms a pymatgen Structure object into its OFM representation.

        Parameters:
            struct (Structure): The pymatgen Structure object representing the material structure.

        Returns:
            OFMFeatureRepresentation: An object encapsulating the OFM representation of the material.
        """
        result = ofm_alloy(struct, self.use_ofm1, self.include_distance)
        return OFMFeatureRepresentation(result)
