
from ToyCompositionDescriptor import AtomicEnvTypeGenerator

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

    cif_file_path = "cif/4364517462409499058_4296803739.cif"
    _test_chemenv(cif_file_path, strategy)
