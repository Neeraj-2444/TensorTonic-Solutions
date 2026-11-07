import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    
    weights = [w.clone() for w in weights]
    biases = [b.clone() for b in biases]

    loss_history = []

    for _ in range(steps):

        activations = [inputs]

        for W, b in zip(weights, biases):
            h = torch.tanh(activations[-1] @ W.T + b)
            activations.append(h)

        predictions = activations[-1].squeeze(-1)

        loss = ((predictions - targets) ** 2).sum()

        loss_history.append(loss.clone())

       
        delta = 2 * (predictions - targets).unsqueeze(-1)

        grad_weights = [None] * len(weights)
        grad_biases = [None] * len(biases)

        for l in reversed(range(len(weights))):

            delta = delta * (1 - activations[l + 1] ** 2)

            grad_weights[l] = delta.T @ activations[l]
            grad_biases[l] = delta.sum(dim=0)

            if l > 0:
                delta = delta @ weights[l]

       
        for l in range(len(weights)):
            weights[l] -= learning_rate * grad_weights[l]
            biases[l] -= learning_rate * grad_biases[l]

    
    h = inputs

    for W, b in zip(weights, biases):
        h = torch.tanh(h @ W.T + b)

    predictions = h.squeeze(-1)

    final_loss = ((predictions - targets) ** 2).sum()

    return predictions, final_loss, weights, biases, loss_history