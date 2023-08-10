#!/usr/bin/env python
# coding: utf-8


from composition_feature import CompositionFeatureGenerator

if __name__ == "__main__":

    try:
        gen = CompositionFeatureGenerator()
    except RuntimeError:
        print("make features and save to the csv.")
        gen = CompositionFeatureGenerator(generation=True)

    df = gen.df_element

    print("shape", df.shape)
    print("columns", df.columns)
    print("melting point", df["melting_point"].values)
    print('electron_affinity', df['electron_affinity'].values)
