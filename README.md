Official implementation of SIGIR'2021 paper "Circumstances enhanced Criminal Court View Generation".

## c3vg dataset
You can download this dataset from https://drive.google.com/file/d/1LzLpqe3YJwtQG8i3RBB49Gkx3Mpd_38m/view?usp=sharing

We have released the original c3vg code in ./original_c3vg.
Besides, considering the advantages of the current pre-trained models (e.g., [bart](https://arxiv.org/abs/1810.04805)), we implement c3vg with pretrained models (i.e., adopting bart to replace GRU in original c3vg), and corresponding codes are shown in ./bart_based_c3vg.

To train c3vg with bart:
```
python train_c3vg.py \
  --gpu_id=1 \
  --model_name=Gen \
  --lr=1e-4 \
  --max_len=512 \
  --types=train \
  --epochs=5 \
  --batch_size=8 \
  --class_num=62 \
  --save_path=./output/ \
```

## Citation
```
@inproceedings{yue2021circumstances,
  title={Circumstances enhanced criminal court view generation},
  author={Yue, Linan and Liu, Qi and Wu, Han and An, Yanqing and Wang, Li and Yuan, Senchao and Wu, Dayong},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1855--1859},
  year={2021}
}
```



