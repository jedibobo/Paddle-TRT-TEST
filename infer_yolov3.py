import numpy as np
import argparse
import cv2
from PIL import Image
import os
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import time
from utils import preprocess, draw_bbox


def create_predictor(args):
    if args.model_dir is not "":
        config = AnalysisConfig(args.model_dir)
    else:
        config = AnalysisConfig(args.model_file, args.params_file)

    config.switch_use_feed_fetch_ops(False)
    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(100, 0)
        config.enable_tensorrt_engine(
	    workspace_size=1<<20,
 	    max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=AnalysisConfig.Precision.Half,
            use_static=True,
            use_calib_mode=True)  #tensorrt_engine
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        #config.enable_mkldnn()

    predictor = create_paddle_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_tensor(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    #print('start')
    import time
    time_start=time.time()
    predictor.zero_copy_run()
    time_end=time.time()
    time_cost=time_end-time_start
    #print('end')
    #print('time cost:{}'.format(time_cost))

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_tensor(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./yolov3-darknet-416/__model__",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./yolov3-darknet-416/__params__",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument("--use_gpu",
                        type=int,
                        default=0,
                        help="Whether use gpu.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path='./imgs'
    pred = create_predictor(args)
    
    start=time.time()
    num=0
    for root, dirs, files in os.walk(path):
        for file in files:
            num+=1
            img_name = root + '/' + file
            img_saving_path = img_name.replace('.jpg', 'result_'+str(round(time.time()*1000)) + '.jpg')
            save_img_name='result/'+img_saving_path.split('/')[-1]
            im_size = 416
            
            img = cv2.imread(img_name)
            data = preprocess(img, im_size)
            im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.int32)
            result = run(pred, [data, im_shape])
            img = Image.open(img_name).convert('RGB').resize((im_size, im_size))
            draw_bbox(img, result[0], save_name=save_img_name)
    end=time.time()
    print('fps:{}'.format(num/(end-start)))
