from ToyCompositionDescriptor import OFMFeatureRepresentation, OFMGenerator

if __name__ == "__main__":
    import seaborn as sns
    from pymatgen.io.cif import CifParser
    import numpy as np
    import matplotlib.pyplot as plt

    def _compare_mean(test: OFMFeatureRepresentation, show_fig: bool):
        """
        Compare the 'mean' value of the provided test object in various formats.

        Parameters:
        - test: Object
            The object that provides as_2d_df, as_2d_array, as_1d_df, and as_1d_array methods.
        - show_fig: bool
            If True, display a heatmap of the 2D DataFrame for the 'mean' value.

        Raises:
        - RuntimeError:
            If there's a discrepancy between any of the data representations.
        """
        key = 'mean'
        df_2d = test.as_2d_df(key)
        if show_fig:
            sns.heatmap(df_2d)
            plt.title(key)
            plt.show()

        v_2d = test.as_2d_array(key)
        if not np.all(v_2d == df_2d.values):
            raise RuntimeError(f'error in {key} 2d test.')

        df_1d = test.as_1d_df(key)

        if not np.all(df_2d.values.ravel() == df_1d.values):
            raise RuntimeError(f'error in {key} df 1d vs 2d test.')

        v_1d = test.as_1d_array(key)
        if not np.all(v_1d == df_1d.values):
            raise RuntimeError(f'error in {key} 1d test.')

    def _compare_locals(test: OFMFeatureRepresentation, show_fig: bool):
        """
        Compare the 'locals' value of the provided test object in various formats.

        Parameters:
        - test: Object
            The object that provides as_2d_df, as_2d_array, as_1d_df, and as_1d_array methods.
        - show_fig: bool
            If True, display a heatmap of the 2D DataFrame for each 'local' value.

        Raises:
        - RuntimeError:
            If there's a discrepancy between any of the data representations.
        """
        key = 'locals'
        df_2d = test.as_2d_df(key)
        if show_fig:
            for i, v in enumerate(df_2d):
                sns.heatmap(v)
                plt.title(key + "/" + str(i))
                plt.show()

        v_2d = test.as_2d_array(key)
        for v1, v2 in zip(df_2d, v_2d):
            if not np.all(v2 == v1.values):
                raise RuntimeError(f'error in {key} 2d test.')

        df_1d = test.as_1d_df(key)
        if not np.all(df_2d.values.ravel() == df_1d.values):
            raise RuntimeError(f'error in {key} df 1d vs 2d test.')

        v_1d = test.as_1d_array(key)
        for v1, v2 in zip(df_1d, v_1d):
            if not np.all(v2 == v1.values):
                raise RuntimeError(f'error in {key} 1d test.')

    def _test_cif(cif_file_path: str, show_fig: bool = False):
        """
        Test the provided CIF file for compatibility and consistency.

        The function will load a structure from a CIF file, then transform it using
        an OFMGenerator. It will then compare the 'mean' and 'locals' values of the
        resulting object in various formats.

        Parameters:
        - cif_file_path: str
            Path to the CIF file to be tested.
        - show_fig: bool, optional (default is False)
            If True, display heatmaps for the 'mean' and 'locals' values.

        Returns:
        - Object
            The transformed test object (an OFMFeatureRepresentation).

        Raises:
        - RuntimeError:
            If there's a discrepancy between any of the data representations.
        """
        print(cif_file_path)
        parser = CifParser(cif_file_path)
        structure = parser.get_structures()[0]
        print(structure)

        ofmconv = OFMGenerator()

        test = ofmconv.transform(structure)
        # OFMFeatureRepresentation is made.

        _compare_mean(test, show_fig)
        _compare_locals(test, show_fig)

        return test

    cif_file_path = "materials_project_database/Nd2Fe14B.cif"
    show_fig = True
    rep = _test_cif(cif_file_path, show_fig=show_fig)

    cif_file_path = "materials_project_database/BaTiO3.cif"
    show_fig = True
    rep = _test_cif(cif_file_path, show_fig=show_fig)
