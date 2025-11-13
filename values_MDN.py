import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# MDN Model (Mixture Density Network)
class MDN(nn.Module):
    def __init__(self, input_dim=1, n_hidden=64, n_components=3):
        '''
        input_dim is number of input features
        n_hidden represents how many neurons in hidden layers
        N components represents how many Gaussians are used per distribution
        '''

        # initializing layers
        super().__init__()
        self.n_components = n_components
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.pi = nn.Linear(n_hidden, n_components)      # mixing coefficients
        self.sigma = nn.Linear(n_hidden, n_components)   # standard deviations
        self.mu = nn.Linear(n_hidden, n_components)      # means
        
    def forward(self, x):
        '''
        Computing the forward pass with the MDN
        first 2 layers use ReLU activation functions
        the output layers use softmax (pi), exponential (sigma) and linear (mu) activations
        '''
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        pi = torch.softmax(self.pi(h), dim=1)
        sigma = torch.exp(self.sigma(h)) + 1e-6  # positive
        mu = self.mu(h)
        return pi, sigma, mu


piece_labels = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen'}

print('Mixture Component Statistics per Piece (from saved model)\n')

for pt in range(1, 6):  # 1–5 for each piece type
    # initialize and load model
    model = MDN(input_dim=1, n_hidden=64, n_components=3)

    # Load the trained weights
    model.load_state_dict(torch.load('piece_weights_nn/'+piece_labels[pt]+'_weights.pth'))
    model.eval()
    x_in = torch.tensor([[pt / 5.0]], dtype=torch.float32)
    with torch.no_grad():
        pi, sigma, mu = model(x_in)

    # converting tensors to numpy arrays
    pi = pi.squeeze().cpu().numpy()
    sigma = sigma.squeeze().cpu().numpy()
    mu = mu.squeeze().cpu().numpy()

    # ouputting parameters of each normal component for each piece
    print(f'{piece_labels[pt]} (type {pt}):')
    for i, (w, m, s) in enumerate(zip(pi, mu, sigma)):
        print(f'  Component {i+1}: weight={w:.3f}, mean={m:.3f}, std={s:.3f}')
    print()

    piece_name = piece_labels[pt]
    # Plotting the mixture components as boxes
    fig, ax = plt.subplots(figsize=(7, 4))

    # Create colours for each Gaussian component
    cmap = plt.cm.viridis
    colours = [cmap(0.2), cmap(0.5), cmap(0.8)]

    for i in range(3):
        m = mu[i]
        s = sigma[i]
        w = pi[i]

        # Box coordinates
        x_center = i + 1
        width = max(0.1, float(w))     # π determines width
        height = 2 * s                 # total spread (±σ)

        # Draw a rounded rectangle for the box
        rect = plt.Rectangle(
            (x_center - width/2, m - s),
            width,
            height,
            facecolor=colours[i],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.4,
            zorder=2
        )
        ax.add_patch(rect)

        # Draw median line at μ
        ax.plot([x_center - width/2, x_center + width/2],
                [m, m],
                color='black',
                linewidth=2,
                zorder=3)

        # Add π label above box
        ax.text(x_center, m + s + 0.05,
                f'π = {w:.2f}',
                ha='center', va='bottom',
                fontsize=10)

    # Styling
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'Comp {i+1}' for i in range(3)], fontsize=11)
    ax.set_ylabel('Estimated Value', fontsize=12)
    ax.set_title(f'MDN Component Distributions for {piece_name}', fontsize=14)

    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.show()
