import torch
import utils

if __name__ == "__main__":
    # specify which UniMol2 model to load
    MODEL_SIZE = "310M"
    # set device and SMILES for testing (batch of two)
    device = torch.device("cpu")
    smiles = ["CCO", "CCO"]
    # load model and featurize via utils
    model = utils.load_model(MODEL_SIZE, device)
    data = utils.featurize(smiles, device)
    # run inference
    with torch.no_grad():
        reps, pair = model(data)
    print("Model:", MODEL_SIZE)
    print("SMILES:", smiles)
    print("Batch representation shape:", reps.shape)
    print(reps)

