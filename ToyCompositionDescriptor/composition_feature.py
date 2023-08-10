from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition
import numpy as np
import os
import pandas as pd
from typing import Union
from pymatgen.core import FloatWithUnit

# from composition_defs import COMPOSITION_ATTRIBUTES, FILENAME
COMPOSITION_ATTRIBUTES = ["Z", "row", "group", "atomic_radius_calculated",
                          "mendeleev_no", "electrical_resistivity", "reflectivity", "refractive_index",
                          "poissons_ratio", "molar_volume", "thermal_conductivity", "boiling_point",
                          "melting_point", "critical_temperature", "superconduction_temperature",
                          "liquid_range", "bulk_modulus", "youngs_modulus", "brinell_hardness",
                          "rigidity_modulus", "mineral_hardness", "vickers_hardness", "density_of_solid",
                          "coefficient_of_linear_thermal_expansion", "ionization_energies",
                          "atomic_mass", "atomic_radius", "average_anionic_radius", "average_cationic_radius",
                          "average_ionic_radius", "electron_affinity"]

if len(COMPOSITION_ATTRIBUTES) != len(set(COMPOSITION_ATTRIBUTES)):
    raise ValueError("length inconsistent.")

FILENAME = "element_descriptor.csv"


def make_elmnames(N=100):
    """make element names from Z=1 to Z=N"""
    elm_names = []
    for z in range(1, N + 1):
        elm = Element.from_Z(z)
        elm_names.append(elm.symbol)
    return elm_names


def make_element_descriptor_from_Z(elements: list, attributes=COMPOSITION_ATTRIBUTES):
    """make descriptor from Composition

    if accept_none_comp, it returns a dict having None values.

    Args:
        comp (str|Composition): chemical formula.
        accept_none_comp (bool): accept comp==None or not. Defautls to False.

    Returns:
        dict: descriptor list.
    """
    attributes = COMPOSITION_ATTRIBUTES

    each_prop = {}
    for index, name in enumerate(elements):
        elm = Element(name)
        each_prop[name] = {"name": name}
        for attrib in attributes:
            # print(attrib)
            v1 = getattr(elm, attrib)
            if v1 is None:
                v1 = None
            elif isinstance(v1, list):  # for the first, second, ... excited states and so on.
                v1 = v1[0]
            elif isinstance(v1, FloatWithUnit):
                v1 = v1.real
            elif isinstance(v1, float):
                v1 = v1
            elif isinstance(v1, int):
                v1 = float(v1)
            elif isinstance(v1, str):  # 数値と補助説明がある場合
                print(f"try convering {elm} {attrib} from", v1)
                s = v1.split()
                try:
                    v1 = float(s[0])
                    print(f"converted value {v1}\n")
                except ValueError as e:
                    print(e)
                    v1 = None
                    print("failed to convert to a numeric value, string ", type(v1), "value", v1)
                    print(f'set it as {v1} for {elm} {attrib}.')
            else:
                print("unknown v1 type", type(v1), "value", v1)
                raise ValueError("unknown v1 type")
            try:
                each_prop[str(elm)][attrib] = v1
            except TypeError as e:
                print(e)
                print(index, name, attrib)
    return pd.DataFrame(each_prop).T


def make_Composition_descriptor(composition: Union[str, Composition],
                                df_element: pd.DataFrame,
                                accept_none_comp: bool = False):
    """make descriptor from Composition

    if accept_none_comp, it returns a dict having None values.

    Args:
        composition (str|Composition): chemical formula.
        df_element (pd.DataFrame): element feature values.
        accept_none_comp (bool): accept comp==None or not. Defautls to False.

    Returns:
        dict: descriptor list.
    """
    comp = composition
    attributes = COMPOSITION_ATTRIBUTES

    if accept_none_comp and comp is None:
        prop = {}
        for desc in attributes:
            for prefix in ["mean_", "var_"]:
                prop[prefix + desc] = None
        return prop
    else:
        if isinstance(comp, str):
            _comp = Composition(comp)
        elif isinstance(comp, Composition):
            _comp = comp
        else:
            raise TypeError(f"unknown type for comp {type(comp)}")

    each_prop = {}
    each_prop["fraction"] = []
    for attrib in attributes:
        each_prop[attrib] = []

    for name in _comp.as_dict().keys():
        try:
            elm = Element(name)
        except ValueError:  # E.g., element name is D0+
            elm = None
        if elm is not None:
            for attrib in attributes:
                value = df_element.loc[str(elm), attrib]
                each_prop[attrib].append(value)
            each_prop["fraction"].append(_comp.get_atomic_fraction(name))
        else:
            each_prop["fraction"].append(None)

    # print(each_prop)
    prop = {}
    # print(attributes)
    for desc in attributes:
        # print(desc)
        name = "mean_" + desc
        v = each_prop[desc]
        # print(desc,v)
        if np.any(np.isnan(v)):
            mean_value = None
        else:
            f = np.array(each_prop["fraction"])
            v = np.array(v)
            mean_value = np.sum(f * v)

        prop[name] = mean_value
        # print(name,prop[name])

        name = "stddev_" + desc
        if mean_value is None:
            stddev_value = None
        else:
            f = each_prop["fraction"]
            d = v - mean_value
            stddev_value = np.sqrt(np.sum(f * d * d))
        prop[name] = stddev_value
        # print(name,prop[name])
    return prop


if False:
    def load_df_element(filepath: str = None) -> pd.DataFrame:
        """Load elemental data from the provided CSV file.

        Args:
            filepath (str): Path to the CSV file containing element data.

        Returns:
            pd.DataFrame: DataFrame containing the loaded elemental data.
        """
        if filepath is None:
            current_file_path = os.path.abspath(__file__)
            current_file_path_splitted = os.path.split(current_file_path)
            filepath = os.path.join(current_file_path_splitted[0], FILENAME)
        df_element = pd.read_csv(filepath).set_index("name")
        return df_element


class CompositionFeatureRepresentation:
    """
    Represents the features extracted from a chemical composition.

    This class provides a simple wrapper around a dictionary that stores the computed features
    from a given chemical composition. It offers utility methods to retrieve the data as a dictionary
    or as a pandas DataFrame.

    Attributes:
        representation (dict): Dictionary storing the computed features of a composition.

    Methods:
        as_dict() -> dict:
            Returns the stored features as a dictionary.

        as_df() -> pd.DataFrame:
            Returns the stored features as a single-row pandas DataFrame.

        df -> pd.DataFrame
            Returns the stored features as a single-row pandas DataFrame.
    """
    def __init__(self, dic):
        self.representation = dic

    def as_dict(self):
        return self.representation

    def as_df(self):
        return pd.DataFrame([self.representation])

    @property
    def df(self):
        return pd.DataFrame([self.representation])


class CompositionFeatureGenerator:
    """
    Generates feature representations for chemical compositions.

    This class provides a mechanism to compute various features for a given chemical composition
    by utilizing provided elemental data. It initializes with elemental data either by reading
    from a CSV file or by generating it. The class then offers a transform method to extract
    features from a given composition.

    Attributes:
        df_element (pd.DataFrame): DataFrame containing elemental data.

    Methods:
        fit(N: int = 100, filepath: Union[str, None] = None, generation: bool = False) -> pd.DataFrame:
            Fits the generator by loading or generating the elemental data.

        _extract_features(composition) -> dict:
            Extracts features from the given composition.

        transform(composition: Union[str, Composition]) -> CompositionFeatureRepresentation:
            Computes the features for the given composition and returns them as a CompositionFeatureRepresentation.
    """
    def __init__(self, N: int = 100, filepath: Union[str, None] = None, generation: bool = False):
        self.df_element = self.fit(N=N, filepath=filepath, generation=generation)

    def fit(self, N: int = 100, filepath: Union[str, None] = None, generation: bool = False):
        if filepath is None:
            current_file_path = os.path.abspath(__file__)
            current_file_path_splitted = os.path.split(current_file_path)
            filepath = os.path.join(current_file_path_splitted[0], FILENAME)

        elm_names = make_elmnames()
        if not os.path.isfile(filepath):
            if generation:
                df = make_element_descriptor_from_Z(elm_names)
                df.to_csv(filepath, index=False)
            else:
                raise RuntimeError(f'failed to find file {filepath}')
        else:
            df = pd.read_csv(filepath)
        return pd.read_csv(filepath).set_index("name")

    def _extract_features(self, composition) -> dict:
        return make_Composition_descriptor(composition, self.df_element)

    def transform(self, composition: Union[str, Composition]) -> CompositionFeatureRepresentation:
        return CompositionFeatureRepresentation(self._extract_features(composition))


if __name__ == "__main__":
    from pymatgen.core import Composition

    cfg = CompositionFeatureGenerator()

    # cfg = CompositionFeatureGenerator(generation=True), generation = True if you make the elmenet feature value csv file.

    chemical_formula = "SrTiO3"
    rep = cfg.transform(chemical_formula)
    print(rep.as_dict())
    print(rep.df)

    composition = Composition(chemical_formula)
    rep = cfg.transform(composition)
    print(rep.as_dict())
    print(rep.df)

    chemical_formula = "H0.2He0.7O0.1"
    composition = Composition(chemical_formula)
    print(rep.as_dict())
    print(rep.df)
