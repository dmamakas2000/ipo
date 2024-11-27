# Deep Learning Models for Corporate Event Prediction: Using Text and Financial Indicators :dollar: :bank: :chart_with_upwards_trend:

## About
This repository hosts the implementation of my M.Sc. thesis, which leverages advanced NLP techniques, including pre-trained Transformers, to predict IPO underpricing and overpricing. By analyzing extensive S-1 filings (up to 20,480 tokens) alongside financial indicators, this study outperforms traditional ML methods, addressing the challenges of long-document processing in financial text classification.

You can view the report of the thesis **[here](http://nlp.cs.aueb.gr/theses/d_mamakas_msc_thesis.pdf)**. Don't forget to like :+1: and share your thoughts :relaxed:.

## Abstract
During the past decades, Initial Public Offerings (IPOs) evolved into an irreplaceable tool for companies to raise capital. Generally, IPOs describe the procedure of offering private corporative shares to the primary market, thus attracting institutional and individual investors to purchase them. Afterward, the securities become available in the secondary market and are easily traded by individuals. Typically, when U.S. firms go public, they follow an explicit procedure. Specifically, the U.S. Securities and Exchange Commission (SEC) requires the submission of the S-1 filing document (also referred to as IPO prospectus) to the Electronic Data Gathering, Analysis, and Retrieval (EDGAR) system. This clause ensures investors have prior knowledge of the issuing company’s valuation, potential risks, or future business plans. Hence, IPO underpricing received considerable attention through the years by triggering economists and financial experts. Overall, underpricing denotes offering an IPO at a price lower than its entered value on the stock market after the first trading day. The opposite scenario indicates IPO overpricing. To investigate these phenomena, previous work applied conventional Machine Learning (ML) techniques that use features retrieved from the S-1 fillings or specific financial indications to classify IPOs. However, measuring the predictive power of the prospectuses becomes a complicated task because of the imposition of processing limitations due to their large document size, as they contain a considerably high number of words, making them hard to process and analyze. Therefore, in this study, we go beyond previous approaches and investigate the predictive power of IPOs by utilizing pre-trained Transformers. To detect underpricing, we use textual information retrieved from S-1 fillings along with specific economic knowledge coming from certain financial indicators. We introduce a collection of models that process texts of up to 20,480 tokens, thus making them a reliable option for facing the needs of this classification task. Finally, the findings indicate that our methods outperform previous ML approaches in most experiments and encourage further investigation in this field.

### Models
1. **BERT-based Approaches** <br>
    We explore the application of BERT (Bidirectional Encoder Representations from Transformers) in financial text analysis, leveraging its bidirectional context understanding and tokenization capabilities. The study utilizes the ``nlpaueb/sec-bert-base model`` ([Loukas et al., 2022](https://huggingface.co/nlpaueb/sec-bert-base)), pre-trained on 260,773 10-K filings and fine-tuned using unfrozen weights. Two architectures are proposed: One that processes only the tokenized textual data and another that combines text embeddings with financial indicators to enhance predictive accuracy. Both approaches allow flexibility in embedding selection, supporting CLS token embeddings or max-pooled contextual embeddings. The final model outputs, reduced through dense layers, are evaluated using Binary Cross Entropy with Logits Loss. The goal is to classify financial filings as underpriced or overpriced, with hyperparameters such as thresholds tuned to optimize performance. Readers should rely on the equivalent section of the thesis for further understanding.

1. **Hierarchical-BERT-based Approaches** <br>
    This family of models extends BERT's capabilities to handle the unique challenges of financial documents, which often exceed standard token limits. Specifically, we employ a modified Hierarchical-BERT architecture based on [Chalkidis et al., 2021](https://github.com/iliaschalkidis/lex-glue). This approach encodes fixed-length text segments (128 tokens each) using ``nlpaueb/sec-bert-base``, then stacks two Transformer encoder layers to integrate segment-level context. A max-pooling process generates a comprehensive embedding, which feeds into a classification head for binary decisions. The study supports two architectures: One using only textual data and another combining text embeddings with financial indicators to improve generalization. Experiments processed up to 8,192 or 20,480 tokens using 64 or 160 segments, significantly surpassing BERT's original 512-token limit. Binary Cross Entropy with logit loss is used for evaluation, and essential parameters, including output dimensions, are tuned to optimize performance.

1. **Longformer-based Approaches** <br>
    This section introduces Longformer-based variants, a family of models optimized for longer sequences. Developing using Longformer ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)), the ``ipo-longformer`` variants are warm-started from the financial-domain pretrained nlpaueb/sec-bert-base model. Positional embeddings from the base model are duplicated to support extended lengths of 8,192 and 20,480 tokens, corresponding to 64 and 160 text segments, respectively. An enhanced variant named ``ipo-longformer-extra-global`` introduces a global SEP token at the end of each segment for improved performance. Both variants allow for processing textual data alone or in combination with financial indicators to improve predictive accuracy. Architecturally, the models use fixed-length segments (128 tokens each), encoded via 12 stacked Transformer encoder layers, with final decisions evaluated using Binary Cross Entropy with Logits Loss. These enhancements aim to balance scalability with accurate financial text classification. Again, readers should rely on the equivalent section of the thesis for further understanding.

1. **Prompt-based Approaches** <br>
    Finally, we explore the innovative potential of GPT-3.5 Turbo, an advanced language model by OpenAI, to diversify approaches in financial text analysis. By leveraging prompts the research optimizes interactions with GPT-3.5 Turbo to tackle classification tasks. Given the model's 4,000-token context window, texts from the Management Discussion and Analysis section were truncated to 500 tokens, with two labeled examples appended to the prompt for context adaptation. The study highlights the importance of well-crafted prompts, addressing GPT-3.5 Turbo’s limited financial domain knowledge by providing essential financial context and specific instructions for categorizing firms as Overpriced or Underpriced. Inspired by Lopez-Lira and Tang (2023), the approach emphasizes prompt optimization to adapt the model to task-specific needs. Final evaluations were performed on a test set, demonstrating the potential of GPT-3.5 Turbo in financial classification tasks while identifying opportunities for future research.

## Dataset
To evaluate the performance of our implemented models, we experiment using a collection of large S-1 documents. For our analysis, we extract four major sections from each filing, named as follows: (1) Summary, (2) Risk Factors, (3) Use of Proceeds, and (4) Management Discussion and Analysis. Despite the information obtained from analyzing S-1 documents, we additionally focus on evaluating certain variables containing vital financial knowledge regarding the IPO firms’ valuation.

## Experiments
We evaluate the performance of our modes using key metrics such as Precision, Recall, F1-score per class, PR-AUC (Precision-Recall Area Under Curve), and macro-averaging. PR-AUC. Learning curves are also provided to validate findings and ensure no overfitting occurs. Each experiment runs with five random seeds, selecting the seed with the highest PR-AUC score achieved across the development set. The training set is balanced via undersampling to address the minority class (overpriced IPOs). While experiments were conducted with balanced and imbalanced training sets, only the best-performing models are presented in the results, with a note on which training approach was used.

### Experimental results
The following table compares the best experiments from each family of models with the optimal baseline that uses pure financial indicators to classify IPOs (and achieved the highest macro-averaged PR-AUC score across all the baseline sections). We show the results across the development and test sets (dev/test). Details regarding the experimental results of each model are analyzed in the equivalent section of the final report.

|                            |   **Baseline**  |     **BERT**    |  **Hier-BERT**  |    **Longformer**   |
|----------------------------|:---------------:|:---------------:|:---------------:|:-------------------:|
| **_Precision (Class 0)_**  | 70.48% / 77.54% | 77.43% / 75.29% | 82.07% / 77.70% | 78.60% / 79.29%     |
| **_Recall (Class 0)_**     | 79.21% / 74.21% | 91.57% / 51.37% | 91.29% / 45.10% | 94.94% / 43.53%     |
| **_Precision (Class 1)_**  | 76.27% / 44.79% | 89.69% / 37.69% | 90.19% / 37.78% | 93.62% / 38.20%     |
| **_Recall (Class 1)_**     | 66.82% / 49.32% | 73.31% / 63.56% | 80.06% / 72.03% | 74.16% / 75.42%     |
| **_PR-AUC (Class 0)_**     | 69.76% / 75.91% | 91.93% / 75.16% | 86.65% / 74.25% | 82.34% / 79.93%     |
| **_PR-AUC (Class 1)_**     | 79.84% / 43.98% | 88.80% / 44.40% | 88.73% / 39.86% | 90.45% / 44.54%     |
| **_F1 (Class 0)_**         | 74.59% / 75.84% | 83.91% / 61.07% | 86.44% / 57.07% | 86.01% / 56.20%     |
| **_F1 (Class 1)_**         | 71.23% / 46.95% | 80.68% / 47.32% | 84.82% / 49.56% | 82.76% / 50.71%     |
| **_Macro-avg. Precision_** | 73.37% / 61.17% | 83.56% / 56.49% | 86.13% / 57.74% | 86.11% / 58.74%     |
| **_Macro-avg. Recall_**    | 73.01% / 61.77% | 82.44% / 57.47% | 85.67% / 58.57% | 84.55% / 59.48%     |
| **_Macro-avg. PR-AUC_**    | 74.80% / 59.94% | 90.36% / 59.78% | 87.69% / 57.06% | **86.40% / 62.23%** |
| **_Macro-avg. F1_**        | 72.91% / 61.39% | 82.30% / 54.20% | 85.63% / 53.32% | 84.38% / 53.46%     |

Finally, we refer to the training requirements by displaying the training runtime, samples per second, and steps per second in the following table. However, we exclude two measurements for the baseline model, as they were not counted during the experimentation process.

|                            |              |          |                       |                |
|----------------------------|:------------:|:--------:|:---------------------:|:--------------:|
|                            | **Baseline** | **BERT** | **Hier-BERT** | **Longformer** |
| **_Train Runtime (sec.)_** |      26      |  165.58  |        1541.21        |     3104.44    |
| **_Train Samples/sec._**   |       -      |   43.00  |          4.62         |      2.29      |
| **_Train Steps/sec_**      |       -      |   1.75   |          1.16         |      0.77      |

## Frequently Asked Questions (FAQ)

### How to run experiments?
For any UNIX server, please follow these steps:

1. Clone the repository using `git clone https://github.com/dmamakas2000/ipo`.

1. Move into the desired directory representing the experiment you want to execute. For example, type `cd scripts/management/`

1. Export the Pythonpath using `export PYTHONPATH="${PYTHONPATH}:/your_directory_to_project/"`.

1. Open the corresponding file and edit the configurations based on your needs.

1. Make the bash script executable using `chmod`. For example, `chmod +x bert_management.sh`.

1. Execute the bash script. For example, `./bert_management.sh`.