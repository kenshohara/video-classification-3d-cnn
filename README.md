# Video Classification Using 3D ResNet
This is a torch code for video (action) classification using 3D ResNet trained by [this code](https://github.com/kenshohara/3D-ResNets).  
The 3D ResNet is trained on the Kinetics dataset, which includes 400 action classes.  
This code uses videos as inputs and outputs class names and predicted class scores for each 16 frames.  

**PyTorch (Python) version of this code is available [here](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).**

## Requirements
* [Torch](http://torch.ch/)
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```
* [json package](https://github.com/clementfarabet/lua---json)
```
luarocks install json
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* Download this code.
* Download the [pretrained model](https://github.com/kenshohara/3D-ResNets/releases).  
  * We recommend ResNet-34.

## Usage
```
th main.lua --input ./input --output ./output.json --model ./resnet-34-kinetics.t7
```

To visualize the classification results, use ```generate_result_video/generate_result_video.py```.

## Citation
If you use this code, please cite the following:
```
@article{hara3dresnets
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh}
  title={Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition}
  journal={arXiv preprint}
  volume={arXiv:1708.07632}
  year={2017}
}
```
