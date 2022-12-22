
from pymatgen.core.units import FloatWithUnit
from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition


def make_descriptor_from_Composition(comp:  tuple[str, Composition]):
    """make descriptor from Composition

    Args:
        comp (str|Composition): chemical formula.

    Returns:
        dict: descriptor list.
    """
    if isinstance(comp, str):
        _comp = Composition(comp)
    elif isinstance(comp, Composition):
        _comp = comp
    else:
        raise TypeError(f"unknown type for comp {type(comp)}")

    attributes = ["Z", "row", "group", "atomic_radius_calculated", "atomic_radius_calculated",
                  "mendeleev_no", "electrical_resistivity", "reflectivity", "refractive_index",
                  "poissons_ratio", "molar_volume", "thermal_conductivity", "boiling_point",
                  "melting_point", "critical_temperature", "superconduction_temperature",
                  "liquid_range", "bulk_modulus", "youngs_modulus", "brinell_hardness",
                  "rigidity_modulus", "mineral_hardness", "vickers_hardness", "density_of_solid",
                  "coefficient_of_linear_thermal_expansion", "ionization_energies",
                  "atomic_mass", "atomic_radius", "average_anionic_radius", "average_cationic_radius",
                  "average_ionic_radius", "electron_affinity"]

    each_prop = {}
    each_prop["fraction"] = []
    for attrib in attributes:
        each_prop[attrib] = []

    for name in _comp.as_dict().keys():
        elm = Element(name)
        for attrib in attributes:
            each_prop[attrib].append(getattr(elm, attrib))
        each_prop["fraction"].append(_comp.get_atomic_fraction(name))

    prop = {}
    for desc in attributes:
        name = "mean_"+desc
        mean_value = 0.0
        for v1, f in zip(each_prop[desc], each_prop["fraction"]):
            if mean_value is not None:
                try:
                    if v1 is None:
                        mean_value = None
                    elif isinstance(v1, list):
                        v1 = v1[0]
                    else:
                        if isinstance(v1, FloatWithUnit):
                            v1 = v1.real
                        mean_value += v1*f
                except TypeError:
                    print(desc, v1, type(v1), f, mean_value)
                    raise TypeError()
        prop[name] = mean_value

        name = "var_"+desc
        if mean_value is None:
            var_value = None
        else:
            var_value = 0.0
            for v1, f in zip(each_prop[desc], each_prop["fraction"]):
                if v1 is not None and mean_value is not None:
                    if isinstance(v1, list):
                        v1 = v1[0]
                    var_value += f*(v1-mean_value)*(v1-mean_value)
                else:
                    v1 = None
        prop[name] = var_value
    return prop


if __name__ == "__main__":
    import pandas as pd

    compostion_list = ["Pt56.0Pb44.0",
                       "Pt55.5Ni44.5",
                       "Pt19.7Fe15.2Co13.8Ni26.6Cu24.7",
                       "Pt38.0Cu25.0Pd37.0",
                       "Pt45.8Cu16.0Ru19.2Os11.6Ir7.4"]

    df_composition = pd.DataFrame(compostion_list, columns=["Composition"])

    descriptor = []
    for i in range(df_composition.shape[0]):
        comp = df_composition.loc[i, "Composition"]
        print("make descriptor from composition", comp)
        prop = make_descriptor_from_Composition(comp)
        descriptor.append(prop)

    df_desc = pd.DataFrame(descriptor)

    print("---descriptors made")
    print(df_desc.columns.tolist())
    print("---descriptors")
    print(df_desc)
