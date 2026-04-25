# Overview
  Implementation of the model and SO-TAD described in the paper "SO-TAD: A Surveillance-Oriented Benchmark for Traffic Accident Detection". The warehouse is being improved. 
# Dataset
  ## How to obtain the dataset:
    We have placed the dataset in **Baidu Cloud Disk**, which can be accessed and downloaded by yourself. 
    Link https://pan.baidu.com/s/1b3PAGQzAiltb3EiOzwzFJQ?pwd=d133. 
    Extraction code：d133. File size 25.48GB 
    
    Or 
    **Terabox**
    Link https://1024terabox.com/s/1d1QDtsLZGqTNNRrQF3PrAw
  ## Other
  We have also updated the annotation information, please download the latest **Appendix.txt** in this repository
# Training
  Use this command line to start the code to start training："CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=7 train.py -batch_size 8 -num_works 7 -vae_ep 50 -gan_ep 60 -val_ep 5 -lr_d 0.002 -lr_g 0.002 -path 1"

# Simple CNN Prototype
  If you only want to validate that the dataset can be loaded and a lightweight baseline can train on a single Colab GPU, use `train_cnn.py`.

  Example:
  `python train_cnn.py --root /content/so-tad-dataset --epochs 5 --batch-size 16 --image-size 128 --frame-stride 30 --max-frames-per-video 16 --train-video-limit 100 --test-video-limit 40`

  Notes:
  - This prototype trains a small 2D CNN on sampled video frames, not the original VAE/GAN pipeline.
  - Videos with index `< 400` are treated as positive samples and videos `>= 400` as negative samples, following the repository's existing evaluation logic.
  - `Appendix.txt` is optional for the prototype, but if it exists it is used to avoid sampling positive training frames after the annotated accident point.
