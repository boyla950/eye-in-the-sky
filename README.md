# Eye In the Sky (Dissertation Project)
Eye in the Sky is the project I undertook in my final year studying BSc Computer Science at Durham University. The focus of the project was to investigate the performance of various object detection and tracking algorithms on the UAV (drone) imagery domain. This repository contains the configuration files and logs from the undertaken expirements, download links to the trained weights for each of the models and my final submitted [dissertation paper](https://github.com/boyla950/eye-in-the-sky/blob/main/project_paper.pdf).

## The Goal
The specific aims of the project could be broken down into 3 components:

 1. To investigate two end-to-end CNN object detection architechture, which difer in design, i.e one- stage vs two-stage from drone based imagery by balancing processing time and performance in real time detection.
 2. To examine the impact of two additional object detection algorithms chosen towards specific issues in drone based imagery such as high altitude and high-object density.
 3. To adopt an existing CNN object detection algorithm chosen, by associating every detection box with tracking for each and every frame under drone based imagery label.

## The Dataset
In this project, I used the [UAVDT dataset](s), which can be downloaded [here](https://gas.graviti.com/dataset/graviti/UAVDT). The open source dataset contains  80,000 fully annotated frames of traffic footage taken from 10 hours of UAV footage. The footage contains 3 class labels of 'car', 'bus' and 'truck' and has been captured at various altitudes, in various weather conditions and with varying levels of occlusion. As such, the dataset contains many common computer vision challenges but also some inherent to the UAV domain (i.e. low resolution objects and high object density)..

## The Algorithms

As part of the first component, the performance of [YOLOv3](https://arxiv.org/abs/1804.02767) (one-stage architecture) and [Faster R-CNN](https://arxiv.org/abs/1506.01497) (two-stage architecture) were investigated on the UAVDT dataset. 

For the second component, more complicated architectures were used which each addressed certain issues of the drone imagery domain. [FreeAnchor RetinaNet](https://arxiv.org/abs/1909.02466) was used as the paper which introduces the method showed improved performance on areas of high object density and on slender objects (potentially useful for detecting buses in UAVDT). The other algorithm used was a variation of FAster R-CNN which incorporates [Context-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188) (CARAFE) which   improves upon the feature upsampling methods used in the Feature Pyramid Network of Faster R-CNN, thus allowing it to better detect small objects (which are prevelant in UAV imagery due to the high altitude).

In final component, the detection algorithm from component 1 and 2 were used as input to two different tracking frameworks; [DeepSORT](https://arxiv.org/abs/1703.07402) and [BYTETrack](https://arxiv.org/abs/2110.06864). The performance of each detector-tracker pair was then investigated and compared.

## The Results

The results of the experiments are shown in detail within the [dissertation paper](https://github.com/boyla950/eye-in-the-sky/blob/main/project_paper.pdf), however in summary we saw the best detection accuracy from the CARAFE based Faster R-CNN model but that the algorithm performs arguably too slow to be practical in real-time detection applications. As such the most appropriate detection algorithm to used appears to be YOLOv3 due to its speed and accpetable levels of performance. As for the object tracking algorithms, the used of BYTETrack provides the best speed and tracking accuracy in most cases, making the YOLOv3-BYTETrack pairing the most appropriate for real-time tracking applications. 


## Future Work
Rather frustratingly, I no longer have access to all the files I used when working on the prject due to them being linked to my account on the university super computer (which I no longer have access to since graduating). As a result I am unable to showcase any of the outputs of the models, short of what is already on display in the project paper. If I ever get access to another high performance GPU I might run these models again and publish the results in this repository.


## Acknowledgements

The experiments I undertook as part of this project relied heavily on the previous work of [Open MM-Lab](https://github.com/open-mmlab) and their [MM-Detection](https://github.com/open-mmlab/mmdetection) and [MM-Tracking](https://github.com/open-mmlab/mmtracking) libraries which I used to train and test the various algorithms on the UAVDT dataset. I also relied on access to the [NCC GPU system](https://nccadmin.webspace.durham.ac.uk) supplied by Durham University which allowed me the resources to be able to run the experiments. Finally, I owe thanks to my supervisor [Dr Yona Falinie Binti-Abd-Gaus](https://github.com/yonafalinie) for the support and guidance she provided all year round.

## Feedback
The final mark received for the whole project was **77%**, of which I am very proud.

> By [boyla950](https://github.com/boyla950).
