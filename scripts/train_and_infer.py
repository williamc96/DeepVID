import os
import sys
import pathlib
from dipn2v_multi_train import run_training
from dipn2v_inference import run_inference
baseDir = pathlib.Path(__file__).absolute().parent.parent
# print(baseDir)
sys.path.append(str(baseDir))
sys.path.append(str(os.path.join(baseDir, 'predict')))
sys.path.append(str(os.path.join(baseDir, 'source')))

def train_and_infer():
    tif_files_directory = os.path.abspath(os.path.join(baseDir, 'datasets', 'training'))
    tif_files = [file for file in os.listdir(tif_files_directory) if file.endswith('.tif')]
    
    for tif_file in tif_files:
        print(f"Training on {tif_file}")
        model_name = run_training([tif_file])
        print(f"Running inference on {tif_file}")
        run_inference(model_name, tif_file)

if __name__ == "__main__":
    train_and_infer()