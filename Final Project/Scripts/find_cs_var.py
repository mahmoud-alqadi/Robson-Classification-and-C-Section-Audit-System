import pandas as pd

it = pd.read_stata("EGIQ73FL.DTA", convert_categoricals=False, iterator=True)
labels = it.variable_labels()

keys = ["cesarean", "caesarean", "c-section", "delivery", "deliver",
        "birth", "born", "pregnan"]

print(f"{len(labels)} variables total\n")
for var, lab in labels.items():
    if any(k in str(lab).lower() for k in keys):
        print(f"{var:<12} {lab}")