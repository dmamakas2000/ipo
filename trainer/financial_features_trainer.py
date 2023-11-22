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
