import pandas as pd
from ToyCompositionDescriptor import make_descriptor_from_Composition

if __name__ == "__main__":

    compostions = ["Pt56.0Pb44.0",
                   "Pt55.5Ni44.5",
                   "Pt19.7Fe15.2Co13.8Ni26.6Cu24.7",
                   "Pt38.0Cu25.0Pd37.0",
                   "Pt45.8Cu16.0Ru19.2Os11.6Ir7.4"]

    df_composition = pd.DataFrame(compostions, columns=["Composition"])

    descriptor = []
    for composition in df_composition["Composition"]:
        print("make descriptor from composition", composition)
        prop = make_descriptor_from_Composition(composition)
        descriptor.append(prop)

    print(descriptor[0])
    
    df_desc = pd.DataFrame(descriptor)
    print(df_desc.shape)
    print("---descriptors")
    print(df_desc.columns.tolist())
    print("---descriptors")
    print(df_desc)
