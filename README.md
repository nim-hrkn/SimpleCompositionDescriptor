# ToyCompositionDescriptor

This software is in the public domain with no warranty.

This function generates descriptor from composition using pymatgen.core.Element

# How to install and uninstall
Steps 1 to 3 can be followed.

Step 1. This package requires pymatgen. 
If you want to install pymatgen with conda, please first install pymatgen following https://pymatgen.org/installation.html before installing this package 

## install

Step 2. download this package

Step 3. install into Python directory.
```
pip install .
````

If pymatgen isn't installed at step 3, pip will attempt to install pymatgen.

## uninstall
```
pip uninstall ToyCompositionDescriptor
```

# How to use

```python
import pandas as pd
from ToyCompositionDescriptor import make_descriptor_from_Composition

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
```
```
make descriptor from composition Pt56.0Pb44.0
make descriptor from composition Pt55.5Ni44.5
make descriptor from composition Pt19.7Fe15.2Co13.8Ni26.6Cu24.7
make descriptor from composition Pt38.0Cu25.0Pd37.0
make descriptor from composition Pt45.8Cu16.0Ru19.2Os11.6Ir7.4
```
```python
print(descriptor[0])
```
```
{'mean_Z': 79.76, 'var_Z': 3.9424, 'mean_row': 6.0, 'var_row': 0.0, 'mean_group': 11.760000000000002, ...}
```
```python
df_desc = pd.DataFrame(descriptor)
print("---descriptors")
print(df_desc.columns.tolist())
print("---descriptors")
print(df_desc)
```
```
---descriptors
['mean_Z', 'var_Z', 'mean_row', 'var_row', 'mean_group', 'var_group', 'mean_atomic_radius_calculated', ...]
---descriptors
   mean_Z       var_Z  mean_row  ...  var_average_ionic_radius  mean_electron_affinity  var_electron_affinity
0  79.760    3.942400     6.000  ...                  0.024839                1.347016               0.770537
1  55.750  617.437500     5.110  ...                  0.001043                1.694370               0.231395
2  37.655  400.273975     4.394  ...                  0.001577                1.146374               0.372978
3  53.910  398.801900     5.130  ...                  0.000324                1.324478               0.460576
4  63.326  391.323724     5.488  ...                  0.004263                1.612717               0.237922

[5 rows x 62 columns]

```
df_desc may contain cells with NaN because some attributes of pymatgen.core.Element contain None for some material element. 
