# ACKE: Adaptive Co-operative Knowledge Enhancement

[TOMM 2026] **Adaptive Co-operative Prompting and Uncertainty-Aware Implicit Knowledge
Enhancement for Cross-Modal Retrieval**
---

## ✨ Abstract

With the rapid growth of internet multimedia data, cross-modal retrieval techniques have garnered significant
attention. Given the inherent complexity and non-intuitive nature of cross-modal relationships, tunning pre-trained Large Multimodal Models (LMMs) with cross-modal data has become a mainstream approach. However, cross-modal data commonly exhibit inter-modal information asymmetry and intra-modal distribution diversity. Faced with these challenges, existing paradigms tend to learn ambiguous and asymmetric cross-modal associations, which introduces semantic noise. In addition, their limited adaptability to the high diversity of the real-world
content further hinders optimal retrieval performance. To address these challenges, this paper proposes Adaptive Co-operative
Knowledge Enhancement (ACKE) method, which comprises the Uncertainty-Aware Inspire Potential (UAIP) and Adaptive Cooperative Prompt (ACP) strategies. UAIP utilizes generative LLMs to generate multi-perspective textual descriptions that enrich semantic information, while employing Dempster-Shafer Theory (DST) to quantify their semantic uncertainty and adjust contribution weights, reducing inaccurate relational mappings
and balancing information asymmetry. ACP constructs a prompt pool where instance-specific visual prompts are dynamically
selected and projected into text prompts, which collaborate to guide modal encoders toward deep semantic consensus, thus mitigating alignment bias from intra-modal distribution diversity and improving accuracy. Extensive experiments are conducted on
two widely used datasets, Flickr30k and MSCOCO, demonstrating the effectiveness of our proposed method. 

ACKE achieves **state-of-the-art performance** on:
- MSCOCO
- Flickr30K


---

## 📦 Requirements

* Python 3.9 
* [PyTorch](http://pytorch.org/) (1.8.1)
* [NumPy](http://www.numpy.org/) (>=1.23.4)
* [transformers](https://huggingface.co/docs/transformers) (4.6.0)
* [timm](https://timm.fast.ai/) (0.4.12)
* [torchvision]()

## Pipline
The whole learning pipline of our model:
<img width="1122" height="828" alt="截屏2025-10-03 14 43 00" src="https://github.com/user-attachments/assets/357564ba-92af-4ab2-bb94-0c2c39a8571a" />


## ⌚️ Results
<img width="419" height="388" alt="截屏2025-10-03 14 44 23" src="https://github.com/user-attachments/assets/1cd25d1b-57a7-407b-9349-36b63c3e41c9" />



## Training
Run `run.py`:

For ACKE on Flickr30K:

```bash
python run.py with data_root=`$DATA_PATH`
```

For ACKE on MSCOCO:

```bash
python run.py with coco_config data_root=`$DATA_PATH`
```
Remember to change YOURDATAROOT into your own data root. 

## 😄 Contact
If there are any questions, please feel free to contact with the author: Shilong Wang (wangshilong@nynu.edn.cn).
