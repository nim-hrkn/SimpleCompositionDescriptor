from ToyCompositionDescriptor import CompositionFeatureRepresentation  # noqa: F401
from ToyCompositionDescriptor import CompositionFeatureGenerator

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
