import sys
import os
import glob
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from model import CNN
from dataset import BuildingDataset

def find_model_file(preferred_name = "cnn_buildings_epoch_30.pth"):

    if os.path.exists(preferred_name):
        return preferred_name
    
    # Just In Case:
    pths = glob.glob("*.pth")
    if not pths:
        return None
    pths = sorted(pths, key = lambda p: os.path.getmtime(p), reverse = True)
    return pths[0]

def main():
    if len(sys.argv) != 3:
        print("python test.py [test_dir] [labels_file]")
        sys.exit(1)

    test_dir = sys.argv[1]
    labels_file = sys.argv[2]

    if not os.path.isdir(test_dir):
        print(f"{test_dir} DOESN'T EXIST.")
        sys.exit(1)
    if not os.path.exists(labels_file):
        print(f"{labels_file} DOESN'T EXIST.")
        sys.exit(1)

    df = pd.read_csv(labels_file)
    num_classes = len(df.iloc[:, 1].unique())

    if num_classes == 0:
        print("Error: labels file appears empty.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])

    test_dataset = BuildingDataset(annotations_file=labels_file, img_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 4)

    model = CNN(in_dim = 3, out_dim = num_classes)

    model_file = find_model_file()
    if model_file is None:
        print("NO PTH MODEL")
        sys.exit(1)

    print(f"Loading model weights from: {model_file}", flush = True)
    state = torch.load(model_file, map_location = device)

    if isinstance(state, dict) and any(k.startswith("state_dict") for k in state.keys()):
        if 'state_dict' in state:
            state = state['state_dict']
    try:
        model.load_state_dict(state)
    except Exception as e:
        try:
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                nk = k.replace("module.", "")
                new_state[nk] = v
            model.load_state_dict(new_state)
        except Exception as e2:
            print("CAN'T LOAD STATE_DICT INTO MODEL.", e)
            print("CAN'T STRIP MODULE" , e2)
            sys.exit(1)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        print("NO IMAGES.")
        sys.exit(1)

    accuracy = correct / total * 100.0
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
