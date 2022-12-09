# Emotic 

PONITOR에서 보이스피싱 피해자를 탐지하는데 감정 인식은 중요한 기술이다. 
하지만 코로나19로 인해 마스크 착용이 일상화된 지금, 표정만으로 감정을 탐지하는 것은 정확도가 낮았다.
따라서 표정뿐만아니라 몸짓, 장면 맥락까지 고려한 emotion recognition 모델을 사용하게 되었다. 

<br></br>
## Pipeline

다음의 논문에서 사용하는 CNN 모델을 사용하였다. 
그 구조와 해당 논문은 다음과 같다. 
![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Model Pipeline") 
###### Model Pipeline ([Image source](https://arxiv.org/pdf/2003.13401.pdf))

첫번째 모듈에서는 YOLO를 이용하여 Body를 detect하고 여기서 body feature를 추출한다.
두번째 모듈에서는 이미지 전체의 image(context) feature를 추출한다.
Fusion network에서는 앞서 추출한 두가지의 feature를 combine하여 최종 결과값인 vad값과 카테고리를 예측한다.  
<br></br>
## Emotic Dataset 
다음의 EMOTIC dataset을 사용하였다.  
*[EMOTIC dataset'](https://paperswithcode.com/dataset/emotic)*
<br></br>
## Usage
pre 파일에는 EMOTIC dataset을 다운로드받아 전처리한 데이터셋들이 npy형식으로 저장되어있어 
별도로 데이터를 다운로드 받고 전처리할 필요가 없다. 
또한 학습이 완료된 모델은 다음에서 다운로드 받을 수 있다.
[Trained model and thresholds] (https://drive.google.com/drive/folders/1e-JLA7V73CQD5pjTFCSWnKCmB0gCpV1D)
다운로드 받은 후 model 폴더 안에 저장해 놓고 yolo_inference.py에서 model을 불러오는 부분에서 경로명을 확인해주면 별도로 학습시키지 않아도 저장된 모델을 불러와 바로 test할 수 있다. 
<br></br>
## To perform inference: 
비디오 파일을 inference하는 코드이다. 
```python
>  python yolo_inference.py --experiment_path proj/debug_exp --video_file C:\emotic-master\assets\video_file.mp4
```
experiment_path : experiment directory의 경로명으로 학습된 모델이 저장되어 있음
video_file: 입력 비디오 파일의 경로  

실행 결과 비디오는 
\model\results 에서 result_vid.mp4 형식으로 확인할 수 있다. 
<br></br>
이미지 파일을 inference하고 싶다면 다음의 코드를 실행시킨다. 
```python
>  python yolo_inference.py --experiment_path C:\emotic-master\model  --inference_file C:\emotic-master\assets\friends.jpg
```
experiment_path : experiment directory의 경로명으로 학습된 모델이 저장되어 있음
inference_file: 입력 이미지 파일의 경로 정보가 적혀있는 txt파일의 경로 
(assets/inference_file.txt 참고) 
<br></br>

## Results 

![Result GIF 1](https://github.com/Ponitor/Ponitor_DL/blob/main/EmotionRecognition/assets/test_result.gif "Result GIF 1")

<br></br>
## Acknowledgements

* [Places365-CNN](https://github.com/CSAILVision/places365) 
* [Pytorch-Yolo](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* 
<br></br>
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
<br></br>
## Reference
[Context Based Emotion Recognition using EMOTIC Dataset]([https://github.com/Tandon-A](https://paperswithcode.com/paper/context-based-emotion-recognition-using))


