# Emotic 

PONITOR에서 보이스피싱 피해자를 탐지하는데 감정 인식은 중요한 기술이다. 
하지만 코로나19로 인해 마스크 착용이 일상화된 지금, 표정만으로 감정을 탐지하는 것은 정확도가 낮았다.
따라서 표정뿐만아니라 몸짓, 장면 맥락까지 고려한 emotion recognition 모델을 사용하게 되었다. 


## Pipeline

다음의 논문에서 사용하는 CNN 모델을 사용하였다. 
그 구조와 해당 논문은 다음과 같다. 
![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Model Pipeline") 
###### Fig 2: Model Pipeline ([Image source](https://arxiv.org/pdf/2003.13401.pdf))

첫번째 모듈에서는 YOLO를 이용하여 Body를 detect하고 여기서 body feature를 추출한다.
두번째 모듈에서는 이미지 전체의 image(context) feature를 추출한다.
Fusion network에서는 앞서 추출한 두가지의 feature를 combine하여 최종 결과값인 vad값과 카테고리를 예측한다.  

## Emotic Dataset 
다음의 EMOTIC dataset을 사용하였다.  
r *['Context based emotion recognition using EMOTIC dataset'](https://arxiv.org/pdf/2003.13401.pdf)*.

## Usage
위의 pre 파일에는 EMOTIC dataset을 다운로드받아 전처리한 데이터셋들이 npy형식으로 저장되어있어 
별도로 데이터를 다운로드 받고 전처리할 필요가 없다. 
또한 학습이 완료된 모델도 /model/models에 저장되어 있기 때문에 별도로 학습시키지 않아도 바로 test할 수 있다. 

## To perform inference: 

```
> python main.py --mode inference --inference_file proj/debug_exp/inference_file.txt --experiment_path proj/debug_exp
```
* mode: Mode to run the main file.
* inference_file: Text file specifying images to perform inference. A row is: 'full_path_of_image x1 y1 x2 y2', where (x1,y1) and (x2,y2) specify the bounding box. Refer [sample_inference_list.txt](https://github.com/Tandon-A/emotic/blob/master/sample_inference_list.txt).
* experiment_path: Path of the experiment directory. Models stored in the the directory are used for inference.     
  
  
You can also train and test models on Emotic dataset by using the [Colab_train_emotic notebook](https://github.com/Tandon-A/emotic/blob/master/Colab_train_emotic.ipynb). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tandon-A/emotic/blob/master/Colab_train_emotic.ipynb)

The **trained models and thresholds** to use for inference purposes are availble [here](https://drive.google.com/drive/folders/1e-JLA7V73CQD5pjTFCSWnKCmB0gCpV1D?usp=sharing). 

## Results 

![Result GIF 1](https://github.com/Tandon-A/emotic/blob/master/assets/eld11_gif2.gif "Result GIF 1")

## Acknowledgements

* [Places365-CNN](https://github.com/CSAILVision/places365) 
* [Pytorch-Yolo](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Context Based Emotion Recognition using Emotic Dataset 
_Ronak Kosti, Jose Alvarez, Adria Recasens, Agata Lapedriza_ <br>
[[Paper]](https://arxiv.org/pdf/2003.13401.pdf) [[Project Webpage]](http://sunai.uoc.edu/emotic/) [[Authors' Implementation]](https://github.com/rkosti/emotic)

```
@article{kosti2020context,
  title={Context based emotion recognition using emotic dataset},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  journal={arXiv preprint arXiv:2003.13401},
  year={2020}
}
```

## Reference
[Context Based Emotion Recognition using EMOTIC Dataset]([https://github.com/Tandon-A](https://paperswithcode.com/paper/context-based-emotion-recognition-using))


