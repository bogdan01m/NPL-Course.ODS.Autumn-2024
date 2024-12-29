# NPL-Course.ODS.Autumn-2024
Project for NLP-course. Autumn 2024 in ODS

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IkgCnvOo3xt-TJUDOkKOPAl6XS3xKjtt#scrollTo=K4bDXI3jSCF2b)

Report is available as [pdf](NLP_Course_Template_Bogdan_Minko.pdf) and as [tex](NLP_Course_Template_Bogdan_Minko/main.tex)

## About work

In this work, the [**WILDGUARD** dataset](https://huggingface.co/datasets/allenai/wildguardmix?row=5) was collected and analyzed, including both valid and adversarial prompts. Two methods were considered for classification: **STS+LGBM** and **RAG**. These methods were applied to classify prompts based on three key categories: Prompt Harm, Response Harm, and Refusal Detection.

The evaluation showed that **Security-RAG** (RAG-based approach) outperformed the other models in Response Harm detection when considering the F1-weighted score, establishing a new state-of-the-art for this label, with an F1-weighted score of **89.9%**. For Prompt Harm detection, **Security-RAG** ranked third, after **GPT-4** and **WILDGUARD**, achieving **86.5%**. In Refusal Detection, **Security-RAG** took second place after **GPT-4**, with an F1 score of **92.0%**.

Additionally, the **STS+LGBM** model, while efficient, showed slightly lower performance, particularly in Response Harm detection, where it achieved **83.0%**. However, it still provided competitive results, demonstrating its potential as a lightweight alternative to more complex models.

The results for the **Adversarial** label classification, which were not addressed in the original article, are also provided in the notebook. These results contribute to a more comprehensive understanding of adversarial prompt detection and demonstrate the model's ability to handle such cases.

Overall, the study demonstrates that **Security-RAG** offers a robust solution for multi-task classification, especially in Response Harm and Refusal Detection, while also contributing valuable insights into adversarial prompt classification, marking significant progress in the field of harmful content detection.

