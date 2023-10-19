from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
import logging
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
import pandas as pd


class LGMFeatureRepresentation:
    """
    Represents a feature representation for local geometric motifs (LGMs) in a structure.

    Attributes:
        structure: A pymatgen Structure object representing the crystal structure.
        lse: LightStructureEnvironments object representing the local structure environments.

    Methods:
        as_1d_array: Returns the LGM features as a 1D dataframe.
        as_dict: Returns the coordination environments as a dictionary.

    Properties:
        df: Returns the LGM features as a 1D dataframe.
        dict: Returns the coordination environments as a dictionary.
    """

    def __init__(self, structure, lse):
        self.structure = structure
        self.lse = lse

    def as_1d_array(self):
        """
        Generates the LGM features as a 1D array.

        Returns:
            A pandas DataFrame where each row represents a specific local geometry motif of a site.
        """
        structure = self.structure
        lse = self.lse
        nsite = len(structure.atomic_numbers)
        result = []
        for isite in range(nsite):
            for ce in lse.coordination_environments[isite]:
                v = {"site": isite}
                v.update(ce)
                result.append(v)
        return pd.DataFrame(result)

    @property
    def df(self):
        """
        A property to get the LGM features as a 1D dataframe.

        Returns:
            pandas DataFrame: The LGM features in a 1D dataframe format.
        """
        return self.as_1d_array()

    def as_dict(self):
        """
        Converts the LGM features into a dictionary format.

        Returns:
            dict: A dictionary where keys are site indices and values are their respective coordination environments.
        """
        return self.lse.coordination_environments

    @property
    def dict(self):
        """
        A property to get the coordination environments as a dictionary.

        Returns:
            dict: A dictionary representation of the coordination environments.
        """
        return self.as_dict()


class LGMGenerator:
    """
    A class to generate local geometric motifs (LGMs) for a given crystal structure using pymatgen's chemenv module.

    Attributes:
        lgf: An instance of LocalGeometryFinder used to find the local geometry motifs.

    Methods:
        transform_simple: Uses the SimplestChemenvStrategy to generate the LGM features for the given structure.
        transform_multi: Uses the MultiWeightsChemenvStrategy to generate the LGM features for the given structure.
        transform: Transforms a structure into LGM features using the specified strategy (either 'simple' or 'multi').

    Note:
        When creating an instance, logging for the pymatgen.analysis.chemenv.coordination_environments is set to INFO level.
    """

    def __init__(self):
        lgf = LocalGeometryFinder()
        if False:
            logging.basicConfig(filename='chemenv_structure_environments.log',
                                format='%(levelname)s:%(module)s:%(funcName)s:%(message)s',
                                level=logging.INFO)
        else:
            logger = logging.getLogger('pymatgen.analysis.chemenv.coordination_environments')
            logger.setLevel(logging.INFO)
        self.lgf = lgf

    def transform_simple(self, structure, maximum_distance_factor=3, only_cations=False, distance_cutoff=1.4, angle_cutoff=0.3):
        """
        Transforms the provided structure into LGM features using the SimplestChemenvStrategy.

        Parameters:
            structure (Structure): A pymatgen Structure object.
            maximum_distance_factor (float, optional): The factor to compute the maximum distance to consider for neighbors.
            only_cations (bool, optional): If True, only cations will be considered.
            distance_cutoff (float, optional): Distance cutoff for the strategy.
            angle_cutoff (float, optional): Angle cutoff for the strategy.

        Returns:
            LGMFeatureRepresentation: An instance with the LGM features of the structure.
        """
        lgf = self.lgf
        lgf.setup_structure(structure=structure)
        se = lgf.compute_structure_environments(maximum_distance_factor=maximum_distance_factor, only_cations=only_cations)
        strategy = SimplestChemenvStrategy(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff)
        lse_simple = LightStructureEnvironments.from_structure_environments(strategy=strategy,
                                                                            structure_environments=se)
        self.lse = lse_simple
        # result = lse_simple.coordination_environments # dict
        return LGMFeatureRepresentation(structure, lse_simple)

    def transform_multi(self, structure, maximum_distance_factor=3, only_cations=False):
        """
        Transforms the provided structure into LGM features using the MultiWeightsChemenvStrategy.

        Parameters:
            structure (Structure): A pymatgen Structure object.
            maximum_distance_factor (float, optional): The factor to compute the maximum distance to consider for neighbors.
            only_cations (bool, optional): If True, only cations will be considered.
            distance_cutoff (float, optional): Distance cutoff for the strategy.
            angle_cutoff (float, optional): Angle cutoff for the strategy.

        Returns:
            LGMFeatureRepresentation: An instance with the LGM features of the structure.
        """
        lgf = self.lgf
        lgf.setup_structure(structure=structure)
        se = lgf.compute_structure_environments(maximum_distance_factor=maximum_distance_factor, only_cations=only_cations)
        # Get the strategy from D. Waroquiers et al., Chem Mater., 2017, 29, 8346.
        strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
        lse_multi = LightStructureEnvironments.from_structure_environments(strategy=strategy, structure_environments=se)
        self.lse = lse_multi
        # result = lse_multi.coordination_environments # dict
        return LGMFeatureRepresentation(structure, lse_multi)

    def transform(self, structure, operation="simple", maximum_distance_factor=3, only_cations=False, distance_cutoff=1.4, angle_cutoff=0.3, **kwargs):
        """
        Transforms the provided structure into LGM features using the specified strategy.

        Parameters:
            structure (Structure): A pymatgen Structure object.
            operation (str, optional): The strategy to use, either 'simple' or 'multi'.
            distance_cutoff (float, optional): Distance cutoff for the strategy.
            angle_cutoff (float, optional): Angle cutoff for the strategy.
            **kwargs: Additional keyword arguments to be passed to the chosen transform method.

        Returns:
            LGMFeatureRepresentation: An instance with the LGM features of the structure.

        Raises:
            ValueError: If an unknown operation is specified.
        """
        if operation == "simple":
            return self.transform_simple(structure, maximum_distance_factor=3, only_cations=False, distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff, **kwargs)
        elif operation == "multi":
            return self.transform_multi(structure, maximum_distance_factor=3, only_cations=False, **kwargs)
        else:
            raise ValueError(f"unknown operation={operation}")
