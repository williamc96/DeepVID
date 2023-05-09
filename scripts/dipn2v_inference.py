import os
import sys
import pathlib

baseDir = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(baseDir))
sys.path.append(str(os.path.join(baseDir, 'predict')))
sys.path.append(str(os.path.join(baseDir, 'source')))

import logging
import multiprocessing
import tensorflow as tf
from source.utils import JsonSaver, ClassLoader

def run_inference(model_name: str, input_fname: str):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    input_fname_NO_EXT = os.path.splitext(input_fname)[0]

    generator_param = {}
    inferrence_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "DIPN2VGenerator"
    generator_param["pre_post_frame"] = 3
    generator_param["pre_post_omission"] = 0
    generator_param[
        "steps_per_epoch"
    ] = -1  # No steps necessary for inference as epochs are not relevant. -1 deactivate it.

    if __name__ == "__main__":
        generator_param["train_path"] = os.path.join(
            baseDir,
            "datasets", 
            input_fname,
        )
    else:
        generator_param["train_path"] = os.path.join(
            baseDir,
            "datasets", 
            'training', 
            input_fname,
        )


    generator_param["batch_size"] = 1
    generator_param[
        "randomize"
    ] = False  # This is important to keep the order and avoid the randomization used during training
    generator_param["blind_pixel_ratio"] = 0.1
    generator_param["blind_pixel_method"] = "replace"
    generator_param["cell_mask_path"] = None

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"
    inferrence_param["rescale"] = True
    inferrence_param["nb_workers"] = multiprocessing.cpu_count()
    print("num_workers:", inferrence_param["nb_workers"])

    # Replace this path to where you stored your model
    base_path = os.path.join(
        baseDir,
        "results",
        model_name,
    )

    inferrence_param["model_path"] = os.path.join(
        base_path,
        model_name + "_model.h5"
    )

    # Replace this path to where you want to store your output file
    inferrence_param["output_file"] = os.path.join(
        base_path,
        "_".join((model_name, "result", input_fname)),
    )

    jobdir = os.path.join(
        base_path,
        "_".join(("inference", input_fname_NO_EXT)),  # suggest to append the name of image in the end (i.e. 2b here)
    )

    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_network = os.path.join(base_path, "network.json")

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    network_obj = ClassLoader(path_network)
    network_callback = network_obj.find_and_build()(path_network)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator, network_callback)

    # Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
    inferrence_class.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print("Please provide the model name as the first command-line argument.")
        sys.exit(1)

    datasets_dir = os.path.join(baseDir, "datasets")
    if len(sys.argv) > 2:
        tif_file = sys.argv[2]
    else:
        tif_file = glob.glob(os.path.join(datasets_dir, "*.tif"))[0]    

    input_fname = os.path.basename(tif_file)

    run_inference(model_name, input_fname)