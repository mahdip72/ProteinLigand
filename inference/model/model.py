import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F

# Ligand dictionary and mapping
LIGANDS = ['2BA', '5GP', '6OU', '8CT', 'A3P', 'A86', 'AC1', 'ACO', 'ACP', 'ACT', 'ADE', 'ADN', 'ADP', 'AF3', 'AGS', 'AJP', 'AKG', 'ALA', 'ALF', 'AMP', 'ANP', 'APC', 'APR', 'AR6', 'ARG', 'ASP', 'ATP', 'AYE', 'AZI', 'B12', 'BCL', 'BCT', 'BDP', 'BEF', 'BGC', 'BLA', 'BMA', 'BPH', 'BTI', 'BTN', 'BU3', 'C2E', 'C5P', 'C8E', 'CA', 'CDP', 'CHL', 'CL', 'CLA', 'CLR', 'CMP', 'CO', 'CO3', 'COA', 'COM', 'CTP', 'CU', 'CU1', 'CYC', 'CYS', 'DD6', 'DMU', 'DTP', 'F3S', 'FAD', 'FBP', 'FDA', 'FE', 'FE2', 'FES', 'FLC', 'FMN', 'FRU', 'FUC', 'G4P', 'G6P', 'GAL', 'GDP', 'GLA', 'GLC', 'GLN', 'GLU', 'GLY', 'GNP', 'GOL', 'GSH', 'GSP', 'GTP', 'HEB', 'HEC', 'HEM', 'HIS', 'I3P', 'IHP', 'IHT', 'II0', 'IMP', 'K', 'KC1', 'KC2', 'LBN', 'LEU', 'LMN', 'LYS', 'MAN', 'MET', 'MG', 'MGD', 'MN', 'MTA', 'NAD', 'NAI', 'NAP', 'NDP', 'NI', 'O', 'OXL', 'OXY', 'P5S', 'PEB', 'PEP', 'PGW', 'PHE', 'PHO', 'PID', 'PIO', 'PLP', 'PLX', 'PMP', 'PNS', 'PO4', 'POP', 'PQN', 'PQQ', 'PRO', 'PYR', 'RBF', 'RET', 'SAH', 'SAM', 'SER', 'SF4', 'SFG', 'SIA', 'SO3', 'SO4', 'STU', 'TPP', 'TRP', 'TTP', 'TYD', 'TYR', 'U10', 'U5P', 'UD1', 'UDP', 'UIX', 'UMP', 'UPG', 'UQ8', 'UTP', 'WO4', 'XYP', 'XYS', 'Y01', 'ZN']

LIGAND_TO_SMILES = {'2BA': 'c1nc(c2c(n1)n(cn2)C3C(C4C(O3)COP(=O)(OC5C(COP(=O)(O4)O)OC(C5O)n6cnc7c6ncnc7N)O)O)N', '5GP': 'NC1=Nc2n(cnc2C(=O)N1)[C@@H]3O[C@H](CO[P](O)(O)=O)[C@@H](O)[C@H]3O', '6OU': 'CCCCCCCCCCCCCCCC(=O)OC[C@H](CO[P](O)(=O)OCCN)OC(=O)CCCCCCC\\C=C/CCCCCCCC', '8CT': 'CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CCCC2(C)C)C)C)C', 'A3P': 'Nc1ncnc2n(cnc12)[CH]3O[CH](CO[P](O)(O)=O)[CH](O[P](O)(O)=O)[CH]3O', 'A86': 'CC(=CC=CC=C(C)C=CC=C(C)C(=O)CC12C(CC(CC1(O2)C)O)(C)C)C=CC=C(C)C=C=C3C(CC(CC3(C)O)OC(=O)C)(C)C', 'AC1': 'CC1C(C(C(C(O1)O)O)O)NC2C=C(C(C(C2O)O)O)CO', 'ACO': 'CC(=O)SCCNC(=O)CCNC(=O)[C@@H](C(C)(C)CO[P@](=O)(O)O[P@@](=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)n2cnc3c2ncnc3N)O)OP(=O)(O)O)O', 'ACP': 'Nc1ncnc2n(cnc12)[CH]3O[CH](CO[P](O)(=O)O[P](O)(=O)C[P](O)(O)=O)[CH](O)[CH]3O', 'ACT': '[O-]C(=O)C', 'ADE': 'Nc1ncnc2[nH]cnc12', 'ADN': 'Nc1ncnc2n(cnc12)[CH]3O[CH](CO)[CH](O)[CH]3O', 'ADP': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)OP(=O)(O)O)O)O)N', 'AF3': 'F[Al](F)F', 'AGS': 'c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=S)(O)O)O)O)N', 'AJP': 'CC1C2C(OC31CCC(C)CO3)C(O)C1C3CCC4CC(OC5OC(CO)C(OC6OC(CO)C(O)C(OC7OCC(O)C(O)C7O)C6OC6OC(CO)C(O)C(OC7OC(CO)C(O)C(O)C7O)C6O)C(O)C5O)C(O)CC4(C)C3CCC21C', 'AKG': 'O=C(O)C(=O)CCC(=O)O', 'ALA': 'CC(C(=O)O)N', 'ALF': 'F[Al-](F)(F)F', 'AMP': 'Nc1ncnc2n(cnc12)[CH]3O[CH](CO[P](O)(O)=O)[CH](O)[CH]3O', 'ANP': 'c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(NP(=O)(O)O)O)O)O)N', 'APC': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(C[P@](=O)(O)OP(=O)(O)O)O)O)O)N', 'APR': 'O=P(O)(OCC3OC(n1c2ncnc(N)c2nc1)C(O)C3O)OP(=O)(O)OCC4OC(O)C(O)C4O', 'AR6': 'c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OCC4C(C(C(O4)O)O)O)O)O)N', 'ARG': 'C(CC(C(=O)O)N)CNC(=[NH2+])N', 'ASP': 'C(C(C(=O)O)N)C(=O)O', 'ATP': 'c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N', 'AYE': 'NCC=C', 'AZI': '[N-]=[N+]=[N-]', 'B12': 'Cc1cc2c(cc1C)n(cn2)C3C(C(C(O3)CO)OP(=O)(O)OC(C)CNC(=O)CCC4(C(C5C6(C(C(C7=[N]6[Co+2]89[N]5=C4C(=C1[NH]8C(=CC2=[N]9C(=C7C)C(C2CCC(=O)N)(C)CC(=O)N)C(C1CCC(=O)N)(C)C)C)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O', 'BCL': 'CC[C@@H]1[C@H](C2=CC3=C(C(=C4[N-]3[Mg+2]56[N]2=C1C=C7[N-]5C8=C([C@H](C(=O)C8=C7C)C(=O)OC)C9=[N]6C(=C4)[C@H]([C@@H]9CCC(=O)OC/C=C(\\C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C)C)C(=O)C)C', 'BCT': 'C(=O)(O)[O-]', 'BDP': '[C@@H]1([C@@H]([C@H](O[C@H]([C@@H]1O)O)C(=O)O)O)O', 'BEF': '[Be-](F)(F)F', 'BGC': 'C(C1C(C(C(C(O1)O)O)O)O)O', 'BLA': 'Cc1c(c([nH]c1\\C=C/2\\C(=C(C(=O)N2)C=C)C)\\C=C/3\\C(=C(C(=N3)\\C=C/4\\C(=C(C(=O)N4)C)C=C)C)CCC(=O)O)CCC(=O)O', 'BMA': 'OC[C@H]1O[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O', 'BPH': 'O=C(OC\\C=C(/C)CCCC(C)CCCC(C)CCCC(C)C)CCC6c4nc(cc1c(c(C(=O)C)c(n1)cc5nc(cc3c(c2C(=O)C(c4c2n3)C(=O)OC)C)C(CC)C5C)C)C6C', 'BTI': 'O=CCCCC[CH]1SC[CH]2NC(=O)N[CH]12', 'BTN': 'OC(=O)CCCC[CH]1SC[CH]2NC(=O)N[CH]12', 'BU3': 'C[C@H]([C@@H](C)O)O', 'C2E': 'O=C7NC(=Nc1c7ncn1C6OC5COP(=O)(OC4C(OC(n2c3N=C(N)NC(=O)c3nc2)C4O)COP(=O)(O)OC5C6O)O)N', 'C5P': 'NC1=NC(=O)N(C=C1)[CH]2O[CH](CO[P](O)(O)=O)[CH](O)[CH]2O', 'C8E': 'O(CCCCCCCC)CCOCCOCCOCCO', 'CA': '[Ca++]', 'CDP': 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)O)O)O', 'CHL': 'CCC1=C(c2cc3c(c(c4n3[Mg]56[n+]2c1cc7n5c8c(c9[n+]6c(c4)C(C9CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C)C(C(=O)c8c7C)C(=O)OC)C)C=C)C=O', 'CL': '[Cl-]', 'CLA': 'CCC1=C(C2=Cc3c(c(c4n3[Mg]56[N]2=C1C=C7N5C8=C(C(C(=O)C8=C7C)C(=O)OC)C9=[N]6C(=C4)C(C9CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C)C)C=C)C', 'CLR': 'CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C', 'CMP': 'c1nc(c2c(n1)n(cn2)C3C(C4C(O3)COP(=O)(O4)O)O)N', 'CO': '[Co+2]', 'CO3': 'C(=O)([O-])[O-]', 'COA': 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)n2cnc3c2ncnc3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCS)O', 'COM': 'O[S](=O)(=O)CCS', 'CTP': 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O', 'CU': '[Cu+2]', 'CU1': '[Cu+]', 'CYC': 'CC[C@@H]1[C@@H](C)C(=O)N\\C1=C/c2[nH]c(\\C=C3/N=C(\\C=C4/NC(=O)C(=C4C)CC)C(=C3CCC(O)=O)C)c(CCC(O)=O)c2C', 'CYS': 'N[CH](CS)C(O)=O', 'DD6': 'CC([C@H]=C[C@H]=C(C#CC=1C(C)(C)CC(CC=1C)O)C)=[C@H]C=[C@H]\\C=C(/C)\\C=C\\C=C(/C)\\C=C/C32OC2(CC(CC3(C)C)O)C', 'DMU': 'CCCCCCCCCCO[CH]1O[CH](CO)[CH](O[CH]2O[CH](CO)[CH](O)[CH](O)[CH]2O)[CH](O)[CH]1O', 'DTP': 'c1nc(c2c(n1)n(cn2)[C@H]3C[C@@H]([C@H](O3)CO[P@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O)N', 'F3S': 'S1[Fe]S[Fe]2S[Fe]1S2', 'FAD': 'Cc1cc2N=C3C(=O)NC(=O)N=C3N(C[C@H](O)[C@H](O)[C@H](O)CO[P@](O)(=O)O[P@@](O)(=O)OC[C@H]4O[C@H]([C@H](O)[C@@H]4O)n5cnc6c(N)ncnc56)c2cc1C', 'FBP': 'O[C@H]1[C@H](O)[C@@](O)(CO[P](O)(O)=O)O[C@@H]1CO[P](O)(O)=O', 'FDA': 'Cc1cc2c(cc1C)N(C3=C(N2)C(=O)NC(=O)N3)CC(C(C(COP(=O)(O)OP(=O)(O)OCC4C(C(C(O4)n5cnc6c5ncnc6N)O)O)O)O)O', 'FE': '[Fe+3]', 'FE2': '[Fe+2]', 'FES': '[Fe]1S[Fe]S1', 'FLC': 'OC(CC([O-])=O)(CC([O-])=O)C([O-])=O', 'FMN': 'Cc1cc2c(cc1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(COP(=O)(O)O)O)O)O', 'FRU': 'OC1C(O)C(OC1(O)CO)CO', 'FUC': 'C[C@@H]1O[C@@H](O)[C@@H](O)[C@H](O)[C@@H]1O', 'G4P': 'c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)O)OP(=O)(O)OP(=O)(O)O)O)N=C(NC2=O)N', 'G6P': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)OP(=O)(O)O', 'GAL': 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', 'GDP': 'c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N', 'GLA': 'OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O', 'GLC': 'C(C1C(C(C(C(O1)O)O)O)O)O', 'GLN': 'C(CC(=O)N)C(C(=O)O)N', 'GLU': 'O=C(O)C(N)CCC(=O)O', 'GLY': 'C(C(=O)O)N', 'GNP': 'O=P(O)(O)NP(=O)(O)OP(=O)(O)OCC3OC(n2cnc1c2N=C(N)NC1=O)C(O)C3O', 'GOL': 'C(C(CO)O)O', 'GSH': 'O=C(NCC(=O)O)C(NC(=O)CCC(C(=O)O)N)CS', 'GSP': 'c1nc2c(n1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=S)(O)O)O)O)N=C(NC2=O)N', 'GTP': 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N', 'HEB': 'CCC1=C(C)C2=NC1=Cc3n4[Fe][N]5C(=C2)C(=C(CCC(O)=O)C5=CC6=NC(=Cc4c(C=C)c3C)C(=C6CCC(O)=O)C)C', 'HEC': 'O=C(O)CCC1=C(C2=CC6=C(C(=C/C)\\C5=CC4=C(C(\\C3=Cc7c(c(c8C=C1N2[Fe](N34)(N56)n78)CCC(=O)O)C)=C/C)C)C)C', 'HEM': 'Cc1c2n3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe]36[N]7=C(C=C8N6C(=C5)C(=C8C)C=C)C(=C(C7=C2)C)C=C)C)CCC(=O)O', 'HIS': 'c1c([nH+]c[nH]1)CC(C(=O)O)N', 'I3P': 'O=P(OC1C(O)C(O)C(OP(=O)(O)O)C(O)C1OP(=O)(O)O)(O)O', 'IHP': 'O[P](O)(=O)O[CH]1[CH](O[P](O)(O)=O)[CH](O[P](O)(O)=O)[CH](O[P](O)(O)=O)[CH](O[P](O)(O)=O)[CH]1O[P](O)(O)=O', 'IHT': 'CC(=CC=CC=C(C)C=CC=C(C)C#CC1=C(C)C[CH](O)CC1(C)C)C=CC=C(C)C=CC2=C(C)CCCC2(C)C', 'II0': 'CC1=C(C(C[C@@H](C1)O)(C)C)C#C/C(=C/C=C/C(=C/C=C/C=C(/C=C/C=C(/C#CC2=C(C[C@H](CC2(C)C)O)C)\\C)\\C)/C)/C', 'IMP': 'c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N=CNC2=O', 'K': '[K+]', 'KC1': 'N16C=3C(=C(C1=CC=7C(=C(C(=Cc2n(c5c(c2C)C(C(C(=O)OC)C5=C4C(=C(C(C=3)=N4)C)\\C=C\\C(=O)O)=O)[Mg]6)N=7)CC)C)\\C=C)C', 'KC2': 'Cc1c2cc3nc(c4c5c(c(c6n5[Mg]n2c(c1C=C)cc7nc(c6)C(=C7C)C=C)C)C(=O)C4C(=O)OC)C(=C3C)C=CC(=O)O', 'LBN': 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCC=CCCCCCCCC', 'LEU': 'CC(C)C[C@@H](C(=O)O)N', 'LMN': 'O(CC(CCCCCCCCCC)(CCCCCCCCCC)COC2OC(CO)C(OC1OC(CO)C(O)C(O)C1O)C(O)C2O)C4OC(C(OC3OC(CO)C(O)C(O)C3O)C(O)C4O)CO', 'LYS': 'N[CH](CCCC[NH3+])C(O)=O', 'MAN': 'C(C1C(C(C(C(O1)O)O)O)O)O', 'MET': 'CSCC[CH](N)C(O)=O', 'MG': '[Mg+2]', 'MGD': 'NC1=NC2=C(N[CH]3[CH](N2)O[CH](CO[P](O)(=O)O[P](O)(=O)OC[CH]4O[CH]([CH](O)[CH]4O)n5cnc6C(=O)NC(=Nc56)N)C(=C3S)S)C(=O)N1', 'MN': '[Mn+2]', 'MTA': 'CSCC1C(C(C(O1)n2cnc3c2ncnc3N)O)O', 'NAD': 'NC(=O)c1ccc[n+](c1)[C@@H]2O[C@H](CO[P]([O-])(=O)O[P@](O)(=O)OC[C@H]3O[C@H]([C@H](O)[C@@H]3O)n4cnc5c(N)ncnc45)[C@@H](O)[C@H]2O', 'NAI': 'c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OCC4C(C(C(O4)N5C=CCC(=C5)C(=O)N)O)O)O)O)N', 'NAP': 'c1cc(c[n+](c1)C2C(C(C(O2)COP(=O)([O-])OP(=O)(O)OCC3C(C(C(O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N', 'NDP': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@@](=O)(O)OC[C@@H]4[C@H]([C@H]([C@@H](O4)N5C=CCC(=C5)C(=O)N)O)O)O)OP(=O)(O)O)N', 'NI': '[Ni++]', 'O': 'O', 'OXL': '[O-]C(=O)C([O-])=O', 'OXY': 'O=O', 'P5S': 'CCCCCCCCCCCCCCCCCC(=O)OC[CH](CO[P](O)(=O)OC[CH](N)C(O)=O)OC(=O)CCCCCCCCCCCCCCCCC', 'PEB': 'CCC1C(C(=O)NC1=CC2=NC(=Cc3c(c(c([nH]3)CC4C(=C(C(=O)N4)C=C)C)C)CCC(=O)O)C(=C2C)CCC(=O)O)C', 'PEP': 'C=C(C(=O)O)OP(=O)(O)O', 'PGW': 'CCCCCCCCCCCCCCCC(=O)OC[CH](CO[P](O)(=O)OC[CH](O)CO)OC(=O)CCCCCCCC=CCCCCCCCC', 'PHE': 'N[CH](Cc1ccccc1)C(O)=O', 'PHO': 'CCC1=C(c2cc3c(c(c([nH]3)cc4nc(c5c6c(c(c([nH]6)cc1n2)C)C(=O)C5C(=O)OC)C(C4C)CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C)C=C)C', 'PID': 'CC(=O)O[CH]1CC(C)(C)[C](=[C]=[CH]C(C)=CC=CC=CC=C(C)C=C2OC(=O)C(=C2)C=C[C]34O[C]3(C)C[CH](O)CC4(C)C)[C](C)(O)C1', 'PIO': 'CCCCCCCC(=O)OC[C@H](CO[P](O)(=O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O[P](O)(O)=O)[C@H](O[P](O)(O)=O)[C@H]1O)OC(=O)CCCCCCC', 'PLP': 'Cc1ncc(CO[P](O)(O)=O)c(C=O)c1O', 'PLX': 'CCCCCCCCCCCCCCCCC[C@@H](O)O[C@H](CO[C@H](O)CCCCCCCCCCCCCCC)CO[P@@](O)(=O)OCC[N+](C)(C)C', 'PMP': 'O=P(O)(O)OCc1cnc(c(O)c1CN)C', 'PNS': 'CC(C)(CO[P](O)(O)=O)[C@@H](O)C(=O)NCCC(=O)NCCS', 'PO4': '[O-][P]([O-])([O-])=O', 'POP': 'O[P@@](=O)([O-])O[P@@](=O)(O)[O-]', 'PQN': 'CC1=C(C(=O)c2ccccc2C1=O)CC=C(C)CCCC(C)CCCC(C)CCCC(C)C', 'PQQ': 'c1c2c([nH]c1C(=O)O)-c3c(cc(nc3C(=O)C2=O)C(=O)O)C(=O)O', 'PRO': 'C1C[C@H](NC1)C(=O)O', 'PYR': 'CC(=O)C(O)=O', 'RBF': 'Cc1cc2N=C3C(=O)NC(=O)N=C3N(C[CH](O)[CH](O)[CH](O)CO)c2cc1C', 'RET': 'CC(=C\\C=O)/C=C/C=C(C)/C=C/C1=C(C)CCCC1(C)C', 'SAH': 'N[CH](CCSC[CH]1O[CH]([CH](O)[CH]1O)n2cnc3c(N)ncnc23)C(O)=O', 'SAM': 'C[S@@+](CC[C@H](N)C([O-])=O)C[C@H]1O[C@H]([C@H](O)[C@@H]1O)n2cnc3c(N)ncnc23', 'SER': 'N[CH](CO)C(O)=O', 'SF4': '[S]12[Fe]3[S]4[Fe]1[S]5[Fe]2[S]3[Fe]45', 'SFG': 'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)C[C@H](CC[C@@H](C(=O)O)N)N)O)O)N', 'SIA': 'CC(=O)N[CH]1[CH](O)C[C](O)(O[CH]1[CH](O)[CH](O)CO)C(O)=O', 'SO3': '[O-][S]([O-])=O', 'SO4': '[O-]S(=O)(=O)[O-]', 'STU': 'CC12C(C(CC(O1)n3c4ccccc4c5c3c6n2c7ccccc7c6c8c5C(=O)NC8)NC)OC', 'TPP': 'Cc1ncc(C[n+]2csc(CCO[P@@](O)(=O)O[P](O)(O)=O)c2C)c(N)n1', 'TRP': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N', 'TTP': 'CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO[P@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O', 'TYD': 'CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)CO[P@](=O)(O)OP(=O)(O)O)O', 'TYR': 'O=C(O)C(N)Cc1ccc(O)cc1', 'U10': 'CC1=C(C(=O)C(=C(C1=O)OC)OC)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C', 'U5P': 'O=C1NC(=O)N(C=C1)C2OC(C(O)C2O)COP(=O)(O)O', 'UD1': 'CC(=O)NC1C(C(C(OC1OP(=O)(O)OP(=O)(O)OCC2C(C(C(O2)N3C=CC(=O)NC3=O)O)O)CO)O)O', 'UDP': 'C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)(O)OP(=O)(O)O)O)O', 'UIX': 'CC(=CC=CC=C(C)C=CC=C(C)C=C=C1C(CC(CC1(C)O)OC(=O)C)(C)C)C=CC=C(C)C=CC23C(CC(CC2(O3)C)O)(C)C', 'UMP': 'O=P(O)(O)OCC2OC(N1C(=O)NC(=O)C=C1)CC2O', 'UPG': 'OC[C@H]1O[C@H](O[P](O)(=O)O[P](O)(=O)OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)N3C=CC(=O)NC3=O)[C@H](O)[C@@H](O)[C@@H]1O', 'UQ8': 'CC1=C(C(=O)C(=C(C1=O)OC)OC)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C', 'UTP': 'C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O)O', 'WO4': '[O-][W](=O)(=O)[O-]', 'XYP': 'C1C(C(C(C(O1)O)O)O)O', 'XYS': 'C1C(C(C(C(O1)O)O)O)O', 'Y01': 'CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)OC(=O)CCC(O)=O', 'ZN': '[Zn++]'}

def idx2ligand(ligand_names):
    """
    Takes a list of ligand names and returns {index: ligand_name} dictionary.
    """
    mapping = {idx: ligand for idx, ligand in enumerate(sorted(set(ligand_names)))}
    mapping[-1] = "UNKNOWN"
    return mapping

class LigandPredictionModel(nn.Module):
    def __init__(self, configs, num_labels=1):
        """
        Fine-tuning model for post-translational modification prediction.

        Args:
            configs: Contains model configurations like
                     - model.model_name (str)
                     - model.hidden_size (int)
                     - model.freeze_backbone (bool, optional)
                     - model.dropout_rate (float, optional)
            num_labels (int): The number of output labels (default: 2 for binary classification).
        """
        super().__init__()

        # 1. Read from configs
        base_model_name = configs.model.model_name
        hidden_size = configs.model.hidden_size
        use_mp = configs.model.use_mixed_precision if hasattr(configs.model, "use_mixed_precision") else False
        dtype = configs.mixed_precision_dtype
        if use_mp and dtype in ("bf16", "bfloat16", "bfloat"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None
        freeze_backbone = configs.model.freeze_backbone
        freeze_embeddings = configs.model.freeze_embeddings
        num_unfrozen_layers = configs.model.num_unfrozen_layers
        classifier_dropout_rate = configs.model.classifier_dropout_rate
        backbone_dropout_rate = configs.model.backbone_dropout_rate
        esm_to_decoder_dropout_rate = configs.model.last_state_dropout_rate
        num_ligands = configs.num_ligands
        ligand_names = LIGANDS
        # If true, use chemical encoder for ligand representation, else use embedding table
        self.use_chemical_encoder = configs.stage_3

        # 2. Load the pretrained transformer
        config = AutoConfig.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        cache_dir = getattr(configs.model, "cache_dir", None)
        if cache_dir and os.path.isdir(cache_dir):
            self.base_model = AutoModel.from_pretrained(base_model_name, config=config, cache_dir=cache_dir)
        else:
            self.base_model = AutoModel.from_pretrained(base_model_name, config=config)
        num_total_layers = len(self.base_model.encoder.layer)
        # option to load structure-aware model
        structure_aware = configs.model.structure_aware
        if structure_aware:
            checkpoint_path = "/home/dc57y/data/checkpoint_60.pth"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')["model_state_dict"]

            # remove '_orig_mod.protein_encoder.model.' from the keys
            checkpoint = {
                k.replace('_orig_mod.protein_encoder.model.', ''): v
                for k, v in checkpoint.items()
            }

            load_report = self.base_model.load_state_dict(checkpoint, strict=False)
            print("Loaded structure-aware weights")
            # print("Load report:", load_report)

        # 3. Freeze backbone if requested
        if freeze_backbone:
            if freeze_embeddings:
                # Freeze all layers (including embeddings)
                for param in self.base_model.parameters():
                    param.requires_grad = False
                # Unfreeze layers
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i >= num_total_layers - num_unfrozen_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                        # add dropout to unfrozen backbone layers
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)
            else:
                # Freeze requested layers and leave embeddings
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i < num_total_layers - num_unfrozen_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)

        # Final dropout layer for the last hidden state
        self.encoder_to_decoder_dropout = nn.Dropout(esm_to_decoder_dropout_rate)

        # 4. Ligand Embedding Table // Chemical Encoder
        if not self.use_chemical_encoder: # Use embedding table
            # Potentially could experiment with variable embedding_dim size
            print("Using embedding table for ligand representation")
            self.ligand_embedding = nn.Embedding(num_embeddings=num_ligands, embedding_dim=hidden_size)
        else: # Use chemical encoder
            print("Using chemical encoder for ligand representation")
            clm_source = configs.model.chemical_encoder_source
            self.ligand_to_smiles = LIGAND_TO_SMILES
            self.idx_to_ligand = idx2ligand(ligand_names)

            if clm_source == "huggingface":
                huggingface_model_name = configs.model.huggingface_model_name
                print(f"Loading {huggingface_model_name} for chemical encoder")
                if cache_dir and os.path.isdir(cache_dir):
                    self.smiles_tokenizer = AutoTokenizer.from_pretrained(
                        huggingface_model_name, trust_remote_code=True, cache_dir=cache_dir)
                else:
                    self.smiles_tokenizer = AutoTokenizer.from_pretrained(
                        huggingface_model_name, trust_remote_code=True)
                self.smiles_tokenizer.pad_token = "<pad>"
                self.smiles_tokenizer.bos_token = "<s>"
                self.smiles_tokenizer.eos_token = "</s>"
                self.smiles_tokenizer.mask_token = "<unk>"
                clm_config = AutoConfig.from_pretrained(huggingface_model_name, trust_remote_code=True, torch_dtype=torch_dtype)
                clm_hidden_dropout = configs.model.clm_hidden_dropout_rate if configs.model.clm_hidden_dropout_rate else 0.0
                clm_embedding_dropout = configs.model.clm_embedding_dropout_rate if configs.model.clm_embedding_dropout_rate else 0.0
                clm_config.hidden_dropout_prob = clm_hidden_dropout
                clm_config.embedding_dropout_prob = clm_embedding_dropout
                if cache_dir and os.path.isdir(cache_dir):
                    self.smiles_model = AutoModel.from_pretrained(
                        huggingface_model_name,
                        config=clm_config,
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                else:
                    self.smiles_model = AutoModel.from_pretrained(
                        huggingface_model_name,
                        config=clm_config,
                        trust_remote_code=True
                    )

                # Freeze everything by default (including embeddings)
                for param in self.smiles_model.parameters():
                    param.requires_grad = False

                # Unfreeze last n layers
                last_n = configs.model.chemical_encoder_num_unfrozen_layers if hasattr(configs.model, "chemical_encoder_num_unfrozen_layers") else 0
                if last_n > 0:
                    for param in self.smiles_model.encoder.layer[-last_n:].parameters():
                        param.requires_grad = True

                # Unfreeze embeddings if requested
                unfreeze_clm_embeddings = configs.model.unfreeze_clm_embeddings if hasattr(configs.model, "unfreeze_clm_embeddings") else False
                if unfreeze_clm_embeddings:
                    print("Unfreezing CLM embeddings")
                    for param in self.smiles_model.embeddings.parameters():
                        param.requires_grad = True

                self.clm_max_length = configs.model.clm_max_length


            elif clm_source == "unimol":
                print("Loading UniMol2 for chemical encoder")
                from unimol.utils import load_model as load_unimol_model
                from unimol.utils import featurize as unimol_featurize
                model_size = configs.model.unimol_model_size
                if cache_dir and os.path.isdir(cache_dir):
                    self.smiles_model = load_unimol_model(model_size, torch.device("cpu"), cache_dir=cache_dir)
                    print("Using cache directory for UniMol2 model:", cache_dir)
                else:
                    self.smiles_model = load_unimol_model(model_size, torch.device("cpu"))  # Might remove device logic later
                    print("Using default cache directory for UniMol2 model")
                self.smiles_model.eval()
                # Freeze all parameters first
                for param in self.smiles_model.parameters():
                    param.requires_grad = False

                # Unfreeze last N layers
                last_n = getattr(configs.model, "chemical_encoder_num_unfrozen_layers", 0)
                if last_n > 0 and hasattr(self.smiles_model, "encoder"):
                    encoder_layers = self.smiles_model.encoder.layers
                    for layer in encoder_layers[-last_n:]:
                        for param in layer.parameters():
                            param.requires_grad = True

                self.unimol_featurize = unimol_featurize

            else:
                raise ValueError(f"Unsupported chemical encoder source: {clm_source}")

            clm_output_dim = configs.model.clm_output_dim
            self.projector = nn.Linear(clm_output_dim, hidden_size)
            self.proj_layernorm = nn.LayerNorm(hidden_size)

        # 5. Transformer Head with Cross-Attention
        num_heads = configs.transformer_head.num_heads
        num_layers = configs.transformer_head.num_layers
        dim_feedforward = configs.transformer_head.dim_feedforward
        dropout = configs.transformer_head.dropout

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 6. Add classifier on top
        # self.classifier = nn.Linear(encoder_output_size, num_labels)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask,ligand_idx, ligand_smiles=None):
        """
        Forward pass for the Ligand prediction model.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask for padding.
            ligand_idx (Tensor): Index of the ligand type.
            ligand_smiles (Tensor, optional): SMILES representation of the ligand (if using PLINDER dataset)

        Returns:
            Tensor: Logits for each residue in the input sequence.
        """
        # 1. Get protein representation
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # protein_repr = outputs.last_hidden_state
        protein_repr = self.encoder_to_decoder_dropout(outputs.last_hidden_state)


        # 2. Retrieve ligand representation
        if self.use_chemical_encoder:
            if ligand_smiles is None:
                # Use dictionary to map ligand_idx to SMILES
                ligand_names = [self.idx_to_ligand[i.item()] for i in ligand_idx]
                smiles_batch = [self.ligand_to_smiles[name] for name in ligand_names]
            else:
                # Use ligand_smiles directly
                smiles_batch = ligand_smiles

            if hasattr(self, 'smiles_tokenizer'):
                # using HuggingFace tokenizer
                encoded = self.smiles_tokenizer(smiles_batch,
                                                max_length=self.clm_max_length,
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors="pt",
                                                add_special_tokens=True).to(input_ids.device)
                ligand_hidden = self.smiles_model(**encoded).last_hidden_state
                ligand_repr = self.proj_layernorm(self.projector(ligand_hidden))
                memory_key_padding_mask = (encoded["attention_mask"] == 0).to(input_ids.device)
            else:
                # using UniMol2 featurization
                featurized = self.unimol_featurize(smiles_batch, input_ids.device)
                ligand_hidden, _ = self.smiles_model(featurized)
                ligand_repr = self.proj_layernorm(self.projector(ligand_hidden))
                mask = (featurized["mask"] == 0).to(input_ids.device)
                memory_key_padding_mask = F.pad(mask, (0, 1), value=True)

        else:
            ligand_repr = self.ligand_embedding(ligand_idx).unsqueeze(1)
            memory_key_padding_mask = None
        # 3. Pass through transformer
        transformer_output = self.transformer_decoder(
            tgt=protein_repr,
            memory=ligand_repr,
            memory_key_padding_mask=memory_key_padding_mask
        )
        normalized = self.norm(transformer_output)
        dropout_output = self.dropout(normalized)
        logits = self.classifier(dropout_output)
        return logits

    def num_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def prepare_model(configs):
    """
    Prepares the ESM2 model and tokenizer based on given configurations.

    Args:
        configs (dict): A dictionary containing model configurations.
            Example keys:
                - "model_name" (str): The name of the ESM model to load (e.g., "facebook/esm2_t12_35M_UR50D").
                - "hidden_size" (int): The hidden size of the model

    Returns:
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model loaded with the specified configurations.
    """
    # Extract configurations
    model_name = configs.model.model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = LigandPredictionModel(configs)

    print(f"Loaded model: {model_name}")
    # print(f"Model has {model.num_parameters():,} trainable parameters")

    return tokenizer, model

if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")

    from box import Box
    import yaml

    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_configs = Box(config_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = prepare_model(test_configs)
    model.to(device)

    # Define a sample amino acid sequence
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEY"

    labels = torch.tensor([
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    ], dtype=torch.long)

    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=False)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = labels.to(device)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        ligand_idx = torch.tensor([6]).to(device)

        # Step 1: Input IDs and Tokens
        print("\n[1] Tokenized Input IDs:")
        print(inputs["input_ids"])
        print("\n[1.1] Tokens:")
        print(tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).tolist()))

        # Step 2: Get ESM2 hidden states
        outputs = model.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        protein_repr = outputs.last_hidden_state
        print(f"\n[2] ESM2 Protein Representation Shape: {protein_repr.shape}")

        # Step 3: Ligand Representation
        if model.use_chemical_encoder:
            # Map ligand_idx to SMILES
            ligand_name = model.idx_to_ligand[ligand_idx.item()]
            try:
                smiles = model.ligand_to_smiles[ligand_name]
            except KeyError:
                print(f"Ligand '{ligand_name}' not found in ligand_to_smiles.")
                print("Falling back to a dummy SMILES: 'CCO'.")
                smiles = "CCO"

            print(f"\n[3] Ligand Name: {ligand_name}")
            print(f"[3.1] SMILES: {smiles}")

            if hasattr(model, "smiles_tokenizer"):
                # HuggingFace chemical encoder
                encoded = model.smiles_tokenizer([smiles], max_length=model.clm_max_length,
                                                 padding="max_length", truncation=True,
                                                 return_tensors="pt").to(device)
                print(f"[3.1.1] Tokenized SMILES Input IDs: {encoded['input_ids']}")
                ligand_hidden = model.smiles_model(**encoded).last_hidden_state
                print(f"[3.1.2] Ligand Hidden States Tensor First 10 Tokens: {ligand_hidden.squeeze(0)[:10]}")
                ligand_repr = model.projector(ligand_hidden)
                print(f"[3.2] CLM Embedding Shape: {ligand_repr.shape}")
                memory_key_padding_mask = encoded["attention_mask"] == 0

            else:
                # UniMol chemical encoder

                if next(model.smiles_model.parameters()).device != torch.device(device):
                    model.smiles_model.to(device)
                featurized = model.unimol_featurize([smiles], torch.device(device)) # Unimol has a different featurization method for single SMILES so "[smiles]" is different from "smiles"
                ligand_hidden, _ = model.smiles_model(featurized)
                print(f"[3.1.2] UniMol Ligand Hidden States Tensor First 10 Tokens: {ligand_hidden.squeeze(0)[:10]}")
                ligand_repr = model.projector(ligand_hidden)
                print(f"[3.2] CLM Embedding Shape: {ligand_repr.shape}")
                memory_key_padding_mask = (featurized["mask"] == 0).to(device)
                B, N, _ = ligand_repr.shape
                mask = featurized["mask"]

                if mask.shape[1] != N:
                    # Pad mask to match ligand_repr length
                    pad_len = N - mask.shape[1]
                    import torch.nn.functional as F
                    mask = F.pad(mask, (0, pad_len), value=0)

                memory_key_padding_mask = (mask == 0)

            transformer_output = model.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr,
                memory_key_padding_mask=memory_key_padding_mask
            )

        else:
            ligand_repr = model.ligand_embedding(ligand_idx).unsqueeze(1)
            print(f"\n[3] Ligand Embedding Shape: {ligand_repr.shape}")
            print(f"[3.1] Ligand Embedding Vector: {ligand_repr.squeeze(1).cpu().numpy()}")

            transformer_output = model.transformer_decoder(
                tgt=protein_repr,
                memory=ligand_repr
            )

        # Step 4: Transformer Decoder (cross-attention)
        print(f"\n[4] Transformer Decoder Output Shape: {transformer_output.shape}")

        # Step 5: Final Classification
        logits = model.classifier(transformer_output)
        print(f"\n[5] Final Logits Shape: {logits.shape}")
        print(f"[5.1] Final Logits (Sample):\n{logits.squeeze(0)[:10]}")  # Show first 10 tokens for readability
