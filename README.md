# DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values (ICMLW-AIW 2025)

**DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values**<br>
Changhun Kim*, Yechan Mun*, Sangchul Hahn, Eunho Yang (*: equal contribution)<br>
ICML Workshop on Actionable Interpretability, 2025
![](https://github.com/drumpt/drumpt.github.io/blob/main/content/publications/deltashap/featured.png)



## Introduction
This repository contains the official PyTorch implementation of **DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values**. DeltaSHAP is a fast, faithful, and model-agnostic explanation method that attributes changes in real-time risk predictions to newly observed clinical features using temporal Shapley values. Built upon [WinIT](https://github.com/layer6ai-labs/WinIT)—with sincere thanks to the original authors.



## Environmental Setup
```bash
conda create -n deltashap python=3.10.9
conda activate deltashap
pip install -r requirements.txt
```



## Reproducing Experiments
```bash
bash scripts/run.sh
```



## Contact
If you have any questions or comments, feel free to contact us via changhun.a.kim@gmail.com.



## Citation
```
@inproceedings{kim2025deltashap,
  title={{DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values}},
  author={Kim, Changhun and Mun, Yechan and Hahn, Sangchul and Yang, Eunho},
  booktitle={ICML Workshop on Actionable Interpretability},
  year={2025}
}
```
