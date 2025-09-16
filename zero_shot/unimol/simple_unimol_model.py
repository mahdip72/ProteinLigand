import torch
from torch.cuda.amp import autocast
import utils

if __name__ == "__main__":
    # specify which UniMol2 model to load
    MODEL_SIZE = "84M"
    # set device and SMILES for testing (batch of two)
    device = torch.device("cuda:1")
    smiles = ["CC(C)C1=CC=C(C=C1)C(C2=CN=C(C=C2)C3CCCCC3)C(=O)OCCN4CCN(CC4)C",
              "c1ccccc1",
              "CC[C@H]1C[C@H]([C@H](O[C@@H]1C(=O)OC)[C@@H]2[C@H]([C@H]([C@H]([C@@H](OC2=O)C)OC)OC)N(C)C)OC3=CC=CC=C3"]
    # load model and featurize via utils
    model = utils.load_model(MODEL_SIZE, device)
    data = utils.featurize(smiles, device)
    # run inference with mixed precision on CUDA
    with torch.inference_mode():
        with autocast():
            reps, pair = model(data)
    print("Model:", MODEL_SIZE)
    print("SMILES:", smiles)
    print("Batch representation shape:", reps.shape)
    print(reps)


