import time

import numpy as np
import torch
import torch.nn as nn

import data.data_utils as data_utils


class EarlyStopper:
    def __init__(self, patience_in_epochs):
        self.min_loss = 1000000
        self.epochs_since_minimum = 0
        self.patience_in_epochs = patience_in_epochs

    def should_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.epochs_since_minimum = 0
            return False

        self.epochs_since_minimum += 1
        if self.epochs_since_minimum > self.patience_in_epochs:
            return True

        return False


def fit_model(
    model,
    train_dataloader,
    validate_data: data_utils.MeDataset,
    learning_rate,
    max_epocs,
    weight_decay,
    num_q_annealing_epochs,
    q_initial_weight,
    lr_reducer_patience,
    lr_reducer_factor,
    patience_in_epochs,
    adam_beta_1,
    adam_beta_2,
    max_training_time_s,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta_1, adam_beta_2),
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_reducer_factor,
        patience=lr_reducer_patience,
    )

    early_stopper = EarlyStopper(patience_in_epochs)

    start_time = time.monotonic()
    was_run_interrupted_due_to_time = False

    train_loss_history = []
    validate_loss_history = []

    last_learning_rate = learning_rate

    for epoch in range(max_epocs):
        qw_term_weight = max(1, (1 - epoch / num_q_annealing_epochs) * q_initial_weight)

        loss_sum = 0
        for batch in train_dataloader:
            loss = model.loss(batch, qw_term_weight=qw_term_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        train_loss_history.append(loss_sum / len(train_dataloader))

        with torch.no_grad():

            validate_loss = model.loss(
                (
                    torch.tensor(validate_data.z),
                    torch.tensor(validate_data.t),
                    torch.tensor(validate_data.w),
                    torch.tensor(validate_data.y),
                    torch.tensor(validate_data.weights),
                ),
                qw_term_weight=qw_term_weight,
            )
            validate_loss_history.append(validate_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, train loss: {train_loss_history[-1]:>7f}, validation loss: {validate_loss_history[-1]:>7f}"
            )

        if epoch >= num_q_annealing_epochs:
            scheduler.step(validate_loss)
            new_learning_rate = scheduler.get_last_lr()[0]
            if new_learning_rate != last_learning_rate:
                print(f"Learning rate changed to {new_learning_rate}")
                last_learning_rate = new_learning_rate
            if early_stopper.should_stop(validate_loss):
                break

        if time.monotonic() - start_time > max_training_time_s:
            was_run_interrupted_due_to_time = True
            break

    print("Done!")
    print(f"Final train loss: {train_loss_history[-1]:>7f}")
    print(f"Final validation loss: {validate_loss_history[-1]:>7f}")

    return was_run_interrupted_due_to_time
