"""
File: financial_features_trainer.py
Author: Dimitrios Mamakas (Athens University of Economics and Business)
Date: November 23, 2023
Description: Implementation of the trainer when additionally ustilizing financial indicators.


License:
This code is provided under the MIT License.
You are free to copy, modify, and distribute the code.
If you use this code in your research, please include a reference to the original study (please visit the home page).
"""

from torch import nn
from transformers import Trainer


class FinancialTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if 'labels' in inputs:
            labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Define Binary Cross Entropy Loss and Sigmoid
        loss_fct = nn.BCEWithLogitsLoss()

        # Compute the loss, and return
        loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss
