from pymatgen.core import Element
from collections import OrderedDict
import numpy as np
from typing import Union, List
from pymatgen.core.composition import Composition
import pandas as pd
from numpy.typing import ArrayLike


class ElementOneHotRepresentation:
    def __init__(self, one_hot: ArrayLike, element_dict: OrderedDict, elements: list):
        self.element_dict = element_dict
        self.keys = elements
        self.one_hot = one_hot

    def as_array(self) -> np.ndarray:
        return self.one_hot

    def as_df(self) -> pd.DataFrame:
        df_one_hot = pd.DataFrame(self.one_hot, index=self.element_dict).T
        return df_one_hot

    @property
    def array(self) -> np.ndarray:
        return self.one_hot

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.one_hot, index=self.element_dict).T


class ElementOneHotGenerator:
    """
    A generator class that converts chemical formulas into one-hot vector representations.

    Attributes:
        element_dict (OrderedDict): A dictionary mapping element symbols to their atomic numbers or custom indices.
        keys (list): List of element symbols present in the `element_dict`.

    Methods:
        fit(N=100, elements=None): Initializes the `element_dict` and `keys`.
        _transform(chemical_formula): Internal method to generate one-hot vector representation for a given chemical formula.
        _generate_one_hot(chemical_formula): Helper method to generate one-hot vector or None-filled vector based on the chemical formula.
        transform_as_array(chemical_formula): Returns the one-hot vector representation as a numpy array for a given chemical formula.
        transform_as_df(chemical_formula): Returns the one-hot vector representation as a pandas DataFrame for a given chemical formula.


    Example:

        formula = "SrTiO3H2"
        gen = ElementOneHotGenerator()
        array = gen.transform_as_df(formula)
        print(array)
        print("-"*30)

        gen = ElementOneHotGenerator()
        gen.fit(N=80)
        array = gen.transform_as_df(formula)
        print(array)
        print("-"*30)

        formula = "SrTiO3H2"
        gen = ElementOneHotGenerator()
        gen.fit(elements=["O","Ti","Sr","H"])
        array = gen.transform_as_df(formula)
        print(array)
        print("-"*30)

        formula = "H0.2He0.7O0.1"
        gen = ElementOneHotGenerator()
        array = gen.transform_as_df(formula)
        print(array)

    """

    def __init__(self, N: int = 100):
        """
        Initializes the ElementOneHotGenerator with a default atomic number range.

        fit(N=100) is automatically done.
        if you want to change the representation, you can overwirte it by fit().

        Args:
            N (int): Maximum atomic number to consider for one-hot encoding (default is 100).
        """
        self.fit(N=N)

    def fit(self, N: int = 100, elements: Union[List[str], None] = None):
        """
        Initializes the `element_dict` and `keys` based on the given atomic number range or element list.

        Args:
            N (int): Maximum atomic number to consider for one-hot encoding.
            elements (List[str], optional): List of element symbols to be used for one-hot encoding.

        Raises:
            ValueError: If the provided `elements` have duplicates or if neither `N` nor `elements` are provided.
        """
        # Z=1からZ=100までの元素名をOrdered辞書に格納
        element_dict = OrderedDict()
        if elements is not None:
            if isinstance(elements, list):
                _elements = list(set(elements))
                if len(elements) != len(_elements):
                    raise ValueError('elements must not be duplicated.')
                for i, element_name in enumerate(elements):
                    element_dict[element_name] = i
                self.element_dict = element_dict
                self.keys = elements
            else:
                raise ValueError('elements must be list.')
        elif N is not None:
            if isinstance(N, int):
                for z in range(1, N + 1):
                    element = Element.from_Z(z)
                    element_dict[element.symbol] = z
                self.element_dict = element_dict
                self.keys = list(element_dict.keys())
            else:
                raise ValueError('N must be int.')
        else:
            raise ValueError('N or elements must be set.')

    def _transform(self, chemical_formula: str) -> np.ndarray:
        """
        Generates the one-hot vector representation for a given chemical formula.

        Args:
            chemical_formula (str): The chemical formula to transform.

        Returns:
            np.ndarray: One-hot vector representation of the chemical formula.
        """
        # 実数が入ったone hot vector を返す。
        chemical_formula = chemical_formula.replace("[", "(")
        chemical_formula = chemical_formula.replace("]", ")")
        composition = Composition(chemical_formula)
        element_dict = self.element_dict
        N = len(element_dict)
        one_hot = np.zeros(N)
        for n in composition.fractional_composition:
            n = str(n)
            z = element_dict[n]
            value = composition.fractional_composition[n]
            z = z - 1
            one_hot[z] = value
        return one_hot

    def _generate_one_hot(self, chemical_formula: str) -> np.ndarray:
        """
        Generates a one-hot vector (filled with real numbers or None) for a given chemical formula.

        Args:
            chemical_formula (str): The chemical formula to transform.

        Returns:
            np.ndarray: One-hot vector filled with real numbers or None values.
        """

        # 実数が入ったone hot vector もしくはNoneが入った one hot vectorを返す。

        try:
            one_hot = self._transform(chemical_formula)
        except (AttributeError, KeyError, ValueError) as e:
            print(e, "for", chemical_formula)
            N = len(self.keys)
            one_hot = np.full(N, None)

        return one_hot

    def transform(self, chemical_formula: str) -> ElementOneHotRepresentation:
        return ElementOneHotRepresentation(self._generate_one_hot(chemical_formula), self.element_dict, self.keys)


if __name__ == "__main__":
    formula = "SrTiO3H2"
    gen = ElementOneHotGenerator()
    rep = gen.transform(formula)
    print(rep.as_array())
    print("-" * 30)

    gen = ElementOneHotGenerator()
    gen.fit(N=80)
    rep = gen.transform(formula)
    print(rep.as_df())
    print("-" * 30)

    formula = "SrTiO3H2"
    gen = ElementOneHotGenerator()
    gen.fit(elements=["O", "Ti", "Sr", "H"])
    rep = gen.transform(formula)
    print(rep.as_df())
    print("-" * 30)

    formula = "H0.2He0.7O0.1"
    gen = ElementOneHotGenerator()
    array = gen.transform(formula)
    print(array.as_df())
