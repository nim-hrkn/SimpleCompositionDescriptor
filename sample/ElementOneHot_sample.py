from ToyCompositionDescriptor import ElementOneHotRepresentation  # noqa: F401
from ToyCompositionDescriptor import ElementOneHotGenerator

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
