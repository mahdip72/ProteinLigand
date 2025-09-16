from openbabel import openbabel as ob

def fix_smiles_with_openbabel(smiles):
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInAndOutFormats("smi", "smi")

    mol = ob.OBMol()
    success = ob_conversion.ReadString(mol, smiles)
    if not success:
        print("[!] Failed to parse SMILES.")
        return None

    # Optional cleanup
    builder = ob.OBBuilder()
    builder.Build(mol)            # Generate 3D coords if needed
    mol.StripSalts()
    mol.AddHydrogens()

    # Write back to SMILES
    fixed_smiles = ob_conversion.WriteString(mol).strip()
    return fixed_smiles

# Example
smiles_string = "Cc1cc2c(cc1C)n(cn2)[C@@H]1[C@@H]([C@@H]([C@H](O1)CO)O[P@@](=O)(O)O[C@H](C)CNC(=O)CC[C@@]1([C@H](C2=[N-]3C1=C(C1=[N]4[Co+2]53([N]3=C(C=C4C([C@@H]1CCC(=O)N)(C)C)[C@H]([C@](C3=C(C1=[N]5[C@@]2([C@@]([C@@H]1CCC(=O)N)(C)CC(=O)N)C)C)(C)CC(=O)N)CCC(=O)N)C)C)CC(=O)N)C)O"
fixed = fix_smiles_with_openbabel(smiles_string)
if fixed:
    print(f"Fixed SMILES: {fixed}")
