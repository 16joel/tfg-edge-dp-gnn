import torch
import copy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    train_logits = logits[data.train_mask]
    train_labels = data.y[data.train_mask]
    loss = torch.nn.functional.cross_entropy(train_logits, train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    masked_logits = logits[mask]
    masked_labels = data.y[mask]
    predictions = masked_logits.argmax(dim=1).cpu().numpy()
    labels = masked_labels.cpu().numpy()
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return accuracy, f1, predictions


def train_and_evaluate(model, data, num_epochs=500, epochs_log=50, patience=20):
    """
    Complete training pipeline with early stopping.

    Evaluates validation accuracy every epoch.
    Logs every epochs_log epochs.
    Stops when val accuracy has not improved for `patience` consecutive epochs.
    Restores the best model weights before final evaluation.

    Parameters
    ----------
    model : torch.nn.Module
    data : torch_geometric.data.Data
    num_epochs : int
        Maximum epochs (default 500)
    epochs_log : int
        Print progress every N epochs (default 50)
    patience : int
        Early stopping patience in epochs (default 20)

    Returns
    -------
    results : dict
        val_accuracy, val_f1, test_accuracy, test_f1
    predictions : dict
        val_pred, test_pred
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("  Training...")
    best_val_accuracy = -float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, data)

        # Evaluate every epoch (needed for correct early stopping)
        val_acc, val_f1, _ = test(model, data, data.val_mask)

        # Track best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log every epochs_log epochs
        if epoch % epochs_log == 0:
            print(f"    Epoch {epoch:3d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Stop if patience exceeded
        if epochs_without_improvement >= patience:
            print(f"    Early stopping at epoch {epoch} (best val acc: {best_val_accuracy:.4f})\n")
            break

    # Restore best model before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    test_acc, test_f1, test_pred = test(model, data, data.test_mask)
    val_acc,  val_f1,  val_pred  = test(model, data, data.val_mask)

    print(f"  Final Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    print(f"  Final Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}\n")

    results = {
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
    }
    predictions = {
        'val_pred': val_pred,
        'test_pred': test_pred,
    }

    return results, predictions


def train_and_evaluate_scenarios(model, data_perturbed, data_original,
                                 num_epochs=500, epochs_log=50, patience=20):
    """
    Training pipeline that evaluates the THREE deployment scenarios of the
    project with a single training loop.

    Scenario definitions
    --------------------
    All three scenarios TRAIN on the perturbed graph (data_perturbed). They
    differ in which graph is used for validation (model selection / early
    stopping) and for the final evaluation:

      S1: validation on perturbed, evaluation on perturbed.
          Fully private regime: the original graph is never accessed.
      S2: validation on perturbed, evaluation on original.
          Private development loop, but at inference time the real graph
          is available.
      S3: validation on original, evaluation on original.
          Model selection itself uses the real graph (non-private
          development loop); upper bound of what the trained weights can
          achieve on the real graph.

    Implementation
    --------------
    The training trajectory (weights at every epoch) is identical regardless
    of the validation graph, because validation never influences the gradient
    updates — it only decides WHICH checkpoint is kept. Therefore one single
    training loop suffices, with two independent early-stopping trackers:

      - tracker P: best val accuracy on the PERTURBED graph -> checkpoint
        shared by S1 and S2.
      - tracker O: best val accuracy on the ORIGINAL graph  -> checkpoint
        for S3.

    Each tracker is FROZEN once its own patience is exhausted (it stops
    updating its best checkpoint), which makes the selected checkpoints
    exactly identical to those of two separate runs with the same seeds.
    The loop ends when both trackers are frozen or num_epochs is reached.

    This function reuses train() and test() defined above.

    Parameters
    ----------
    model : torch.nn.Module
    data_perturbed : torch_geometric.data.Data
        DP-perturbed graph (training + S1/S2 validation + S1 evaluation).
    data_original : torch_geometric.data.Data
        Original graph (S3 validation + S2/S3 evaluation).
        Must share x, y and masks with data_perturbed; only edge_index
        may differ.
    num_epochs : int
        Maximum training epochs.
    epochs_log : int
        Print progress every N epochs.
    patience : int
        Early stopping patience, applied independently to each tracker.

    Returns
    -------
    results : dict
        {'s1': {...}, 's2': {...}, 's3': {...}}, each with keys
        val_accuracy, val_f1, test_accuracy, test_f1. The validation
        metrics of each scenario are reported on its own validation graph
        (perturbed for S1/S2, original for S3).
    """
    # Sanity: both graphs must describe the same nodes/labels/splits.
    assert torch.equal(data_perturbed.y, data_original.y), \
        "data_perturbed and data_original must share node labels"
    for mask_name in ('train_mask', 'val_mask', 'test_mask'):
        assert torch.equal(getattr(data_perturbed, mask_name),
                           getattr(data_original, mask_name)), \
            f"data_perturbed and data_original must share {mask_name}"

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("  Training (single loop, dual early-stopping trackers)...")

    # Tracker P: validation on the perturbed graph (S1, S2)
    best_val_p = -float('inf')
    best_state_p = None
    no_improve_p = 0
    frozen_p = False

    # Tracker O: validation on the original graph (S3)
    best_val_o = -float('inf')
    best_state_o = None
    no_improve_o = 0
    frozen_o = False

    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, data_perturbed)

        if not frozen_p:
            val_acc_p, val_f1_p, _ = test(model, data_perturbed,
                                          data_perturbed.val_mask)
            if val_acc_p > best_val_p:
                best_val_p = val_acc_p
                best_state_p = copy.deepcopy(model.state_dict())
                no_improve_p = 0
            else:
                no_improve_p += 1
            if no_improve_p >= patience:
                frozen_p = True
                print(f"    [S1/S2] tracker frozen at epoch {epoch} "
                      f"(best val acc on perturbed: {best_val_p:.4f})")

        if not frozen_o:
            val_acc_o, val_f1_o, _ = test(model, data_original,
                                          data_original.val_mask)
            if val_acc_o > best_val_o:
                best_val_o = val_acc_o
                best_state_o = copy.deepcopy(model.state_dict())
                no_improve_o = 0
            else:
                no_improve_o += 1
            if no_improve_o >= patience:
                frozen_o = True
                print(f"    [S3]    tracker frozen at epoch {epoch} "
                      f"(best val acc on original:  {best_val_o:.4f})")

        if epoch % epochs_log == 0:
            print(f"    Epoch {epoch:3d}, Loss: {loss:.4f}, "
                  f"Val(P): {best_val_p:.4f}{' [frozen]' if frozen_p else ''}, "
                  f"Val(O): {best_val_o:.4f}{' [frozen]' if frozen_o else ''}")

        if frozen_p and frozen_o:
            print(f"    Early stopping at epoch {epoch} (both trackers frozen)\n")
            break

    # ---- S1 and S2: checkpoint selected on perturbed validation ----
    if best_state_p is not None:
        model.load_state_dict(best_state_p)

    s1_val_acc, s1_val_f1, _ = test(model, data_perturbed, data_perturbed.val_mask)
    s1_test_acc, s1_test_f1, _ = test(model, data_perturbed, data_perturbed.test_mask)
    s2_test_acc, s2_test_f1, _ = test(model, data_original, data_original.test_mask)

    results_s1 = {
        'val_accuracy': s1_val_acc, 'val_f1': s1_val_f1,
        'test_accuracy': s1_test_acc, 'test_f1': s1_test_f1,
    }
    # S2 shares the model-selection step with S1, so its validation metrics
    # (on the perturbed graph) are by construction those of S1.
    results_s2 = {
        'val_accuracy': s1_val_acc, 'val_f1': s1_val_f1,
        'test_accuracy': s2_test_acc, 'test_f1': s2_test_f1,
    }

    # ---- S3: checkpoint selected on original validation ----
    if best_state_o is not None:
        model.load_state_dict(best_state_o)

    s3_val_acc, s3_val_f1, _ = test(model, data_original, data_original.val_mask)
    s3_test_acc, s3_test_f1, _ = test(model, data_original, data_original.test_mask)

    results_s3 = {
        'val_accuracy': s3_val_acc, 'val_f1': s3_val_f1,
        'test_accuracy': s3_test_acc, 'test_f1': s3_test_f1,
    }

    print(f"  [S1] perturbed/perturbed  -> Test Acc: {s1_test_acc:.4f}, F1: {s1_test_f1:.4f}")
    print(f"  [S2] perturbed/original   -> Test Acc: {s2_test_acc:.4f}, F1: {s2_test_f1:.4f}")
    print(f"  [S3] original/original    -> Test Acc: {s3_test_acc:.4f}, F1: {s3_test_f1:.4f}\n")

    return {'s1': results_s1, 's2': results_s2, 's3': results_s3}
