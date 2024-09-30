# Regularizing Dynamic Radiance Fields <br> with Kinematic Fields

> This paper presents a novel approach for reconstructing dynamic radiance fields from monocular videos. We integrate kinematics with dynamic radiance fields, bridging the gap between the sparse nature of monocular videos and the real-world physics. Our method introduces the kinematic field, capturing motion through kinematic quantities: velocity, acceleration, and jerk. The kinematic field is jointly learned with the dynamic radiance field by minimizing the photometric loss without motion ground truth. We further augment our method with physics-driven regularizers grounded in kinematics. We propose physics-driven regularizers that ensure the physical validity of predicted kinematic quantities, including advective acceleration and jerk. Additionally, we control the motion trajectory based on rigidity equations formed with the predicted kinematic quantities. In experiments, our method outperforms the state-of-the-arts by capturing physical motion patterns within challenging real-world monocular videos.

<details>
<summary>Short Video</summary>

https://github.com/user-attachments/assets/82252634-17b1-486b-bdd1-49e524036a27
  
</details>

<img width="500" alt="kinematic" src="https://github.com/user-attachments/assets/9896ff36-dcc6-45e4-b9fa-8b0d5005d4ad">

> **Regularizing Dynamic Radiance Fields with Kinematic Fields**  
> [Woobin Im](https://iwbn.github.io), Geonho Cha, [Sebin Lee](https://sgvr.kaist.ac.kr/member/#post-2090), [Juhyeong Seon](https://sunjuhyeong.github.io/online-cv/), [Jumin Lee](https://zoomin-lee.github.io/), Dongyoon Wee, and [Sung-Eui Yoon](http://sgvr.kaist.ac.kr/~sungeui/)  
> in ECCV 2024

## Code Instructions
### Dataloaders
- We provide training configs of NDVS-24-frames in `config/ndvs_24f/*.yaml`
- For `ndvs_24f`, you can download the dataset from [google drive](https://drive.google.com/file/d/1cjBgF61D16IPeW1V_rL2JIRKJP7S3c6_/view?usp=share_link) (This link is provided by this GitHub repository: https://github.com/zhengqili/Neural-Scene-Flow-Fields)


### Training
- Kinematic loss functions are implemented in `model/HexPlaneSD_Flow.py`
  - About Naming
    - `HexPlane`: fields type
    - `SD`: static / dynamic separation
    - `Flow`: using kinematic fields
- ⚠️ Training loop (in `render/trainer.py`) is under company reviewing process (NAVER Cloud) before releasing it to public.
  - Even before the release, you can use the output of `HexPlanSD_Flow` module, and implement on your own. (see https://github.com/Caoang327/HexPlane/blob/main/hexplane/render/trainer.py)

## Inference
We provide pretrained weights of the provided `ndvs_24f` scenes ([ndvs_24f.tar.gz](https://sgvr.kaist.ac.kr/wp-content/uploads/2024/09/ndvs_24f.tar.gz), 2GB). After decompressing it, you can use the command below to get rendered images and flows at 69,999 step.


```bash
python main.py config=ndvs_24f/balloon1/cfg.yaml \
data.datadir=data/NSFF/dynamic_scene_data_full/nvidia_data_full/Balloon1-2/dense/ \
systems.ckpt=ndvs_24f/balloon1/balloon1_69999.th \
use_intermediate_ckpt=True \
render_only=True render_path=False render_test=True \
```


## Bibtex
If you find this code useful for your research, please consider citing our paper:

    @inproceedings{im2024kinematic,
        title={Regularizing Dynamic Radiance Fields with Kinematic Fields},
        author={Im, Woobin and Cha, Geonho and Lee, Sebin and Lee, Jumin and Seon, Juhyeong and Wee, Dongyoon and Yoon, Sung-Eui},
        booktitle={European Conference on Computer Vision},
        year={2024}
    }