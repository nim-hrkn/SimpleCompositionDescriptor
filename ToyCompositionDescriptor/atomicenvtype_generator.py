

"""
This module provides tools to identify coordination environments in crystal structures. 
It is primarily based on the work from D. Waroquiers et al. described in the given references.

References:
    - D. Waroquiers, J. George, M. Horton, S. Schenk, K. A. Persson, G.-M. Rignanese, X. Gonze, G. Hautier, 
      Acta Cryst B 2020, 76, 683–695.
    - D. Waroquiers et al., Chem Mater., 2017, 29, 8346
"""
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
import pandas as pd
from pymatgen.core import Structure


def _extract_aet_data(struct, coordination_environments) -> pd.DataFrame:
    """
    Convert atomic environment type data into a DataFrame.

    Args:
        struct (Structure): Pymatgen Structure object.
        coordination_environments (list): List of coordination environments.

    Returns:
        pd.DataFrame: Dataframe representation of the coordination environments.
    """    
    nsite = len(struct.sites)
    result = []
    for isite in range(nsite):

        for ce in coordination_environments[isite]:
            v = {"site": isite}
            v.update(ce)
            result.append(v)

    return pd.DataFrame(result)


class AtomicEnvTypeRepresentation:
    """
    A class to represent the atomic environment type of atoms in a structure.

    Attributes:
        struct (Structure): Pymatgen Structure object.
        se (StructureEnvironments): Pymatgen StructureEnvironments object.
    """    
    def __init__(self, struct: Structure, se: LightStructureEnvironments):
        self.struct = struct
        self.se = se

    def validate_strategy(self, strategy: str):
        """
        Validates the given strategy to check if it is either 'simple' or 'multi'.

        Args:
            strategy (str): The strategy to be validated.

        Raises:
            ValueError: If the strategy is neither 'simple' nor 'multi'.
        """        
        valid = strategy in ["simple", "multi"]
        if not valid:
            raise ValueError("strategy must be simple or multi.")

    def _transform_simple_strategy(self, cutoff: float = 5.0, angle_cutoff: float = 0.3) -> dict:
        """
        Transforms the atomic environment using the simple strategy.

        Args:
            cutoff (float): Distance cutoff for the simple strategy. Defaults to 5.0.
            angle_cutoff (float): Angle cutoff for the simple strategy. Defaults to 0.3.

        Returns:
            dict: Coordination environments identified using the simple strategy.
        """        
        strategy = "simple"
        se = self.se
        strategy = SimplestChemenvStrategy(distance_cutoff=cutoff, angle_cutoff=angle_cutoff)
        lse_simple = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        return lse_simple.coordination_environments

    def _transform_multi_strategy(self) -> dict:
        """
        Transforms the atomic environment using the multi strategy as defined in 
        D. Waroquiers et al., Chem Mater., 2017, 29, 8346.

        Returns:
            dict: Coordination environments identified using the multi strategy.
        """        
        strategy = "multi"
        se = self.se
        strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
        lse_multi = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        return lse_multi.coordination_environments

    def _transform(self, strategy: str) -> dict:
        """
        General method to transform atomic environment based on the given strategy.

        Args:
            strategy (str): The strategy to be used ('simple' or 'multi').

        Returns:
            dict: Coordination environments identified using the specified strategy.
        """        
        self.validate_strategy(strategy)
        if strategy == "simple":
            return self._transform_simple_strategy()
        elif strategy == "multi":
            return self._transform_multi_strategy()

    def as_dict(self, strategy: str) -> dict:
        """
        Represents the atomic environment as a dictionary based on the given strategy.

        Args:
            strategy (str): The strategy to be used ('simple' or 'multi').

        Returns:
            dict: Coordination environments in dictionary format.
        """        
        return self._transform(strategy)

    def as_df(self, strategy: str) -> pd.DataFrame:
        """
        Represents the atomic environment as a DataFrame based on the given strategy.

        Args:
            strategy (str): The strategy to be used ('simple' or 'multi').

        Returns:
            pd.DataFrame: Coordination environments in dataframe format.
        """        
        result = self._transform(strategy)
        return _extract_aet_data(self.struct, result)


class AtomicEnvTypeGenerator:
    """
    A class to detect atomic environment type in a given crystal structure.

    Attributes:
        only_cations (bool): Whether to consider only cationic sites. Note: Currently not working.
    """
    def __init__(self, only_cations=False):
        """
        Initializes the AtomicEnvTypeGenerator.

        Args:
            only_cations (bool): If True, only consider cationic sites. Default is False.
                                 Note: As of Aug. 6, 2023, this feature doesn't work.

        Attributes:
            only_cations (bool): Stored preference on whether to consider only cationic sites.
        """        
        self.only_cations = only_cations

    def transform(self, struct: Structure) -> AtomicEnvTypeRepresentation:
        """
        Computes atomic environment types for the given structure.

        Args:
            struct (Structure): A pymatgen Structure object representing the crystal structure.

        Returns:
            AtomicEnvTypeRepresentation: A representation of the atomic environment types 
                                         in the given structure.
        """        
        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=struct)
        se = lgf.compute_structure_environments(maximum_distance_factor=3, only_cations=self.only_cations)
        self.lgf = lgf
        self.se = se
        return AtomicEnvTypeRepresentation(struct, self.se)


if __name__ == "__main__":
    from pymatgen.io.cif import CifParser

    def _test_chemenv(cif_file_path, strategy="simple"):
        """
        Utility function to test the atomic environment extraction for a given CIF file and strategy.

        Args:
            cif_file_path (str): Path to the CIF file to be analyzed.
            strategy (str): The strategy to be used ('simple' or 'multi'). Defaults to 'simple'.

        Prints:
            CIF file path, structure information, and atomic environments as a DataFrame.
        """        
        print(cif_file_path)
        parser = CifParser(cif_file_path)
        struct = parser.get_structures()[0]
        print(struct)

        gen = AtomicEnvTypeGenerator(only_cations=False)
        rep = gen.transform(struct)

        print(rep.as_df(strategy))

    strategy = "multi"

    cif_file_path = "materials_project_database/BaTiO3.cif"
    _test_chemenv(cif_file_path, strategy)

    cif_file_path = "../cif/4364517462409499058_4296803739.cif"
    _test_chemenv(cif_file_path, strategy)
