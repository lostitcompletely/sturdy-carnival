import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load data
with open('combined_piece_values.pkl', 'rb') as f:
    all_data = pickle.load(f)

piece_labels = {1:'Pawn', 2:'Knight', 3:'Bishop', 4:'Rook', 5:'Queen'}

# Remove None and NaN values
X, y = [], []
for pt, vals in all_data.items():
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        continue
    for v in vals:
        X.append([pt])
        y.append(v)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1,1)

# Normalize piece type for NN
X[:,0] = X[:,0] / 5.0  # scale to [0,1]

# Convert to torch tensors
X = torch.tensor(X)
y = torch.tensor(y)

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
        self.pi = nn.Linear(n_hidden, n_components)      # coefficients
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

# Loss function (negative log likelihood)
def mdn_loss(pi, sigma, mu, y):
    '''
    computes loss using log likelihood
    m is a log-normal distribution so we need to exponentiate it again
    weight the probabilities by pi and sum over components
    use negative log as we want to maximize likelihood while reducing underflow
    (maximising likelihood = minimizing negative log likelihood)
    '''
    m = torch.distributions.Normal(mu, sigma)

    # reshaping m to match y
    prob = m.log_prob(y.expand_as(mu)).exp()

    # weight probabilities
    weighted = prob * pi
    nll = -torch.log(weighted.sum(dim=1) + 1e-8)
    return nll.mean()


# Function to get PDF for a piece
def mdn_pdf(piece_type, y_vals):
    x_in = torch.tensor([[piece_type/5.0]], dtype=torch.float32)
    with torch.no_grad():
        pi, sigma, mu = model(x_in)
    pi = pi.numpy()[0]
    sigma = sigma.numpy()[0]
    mu = mu.numpy()[0]
    pdf = np.zeros_like(y_vals, dtype=float)

    # compute mixture PDF using outputs from model
    for i in range(len(pi)):
        pdf += pi[i] * (1/(np.sqrt(2*np.pi)*sigma[i])) * np.exp(-(y_vals - mu[i])**2 / (2*sigma[i]**2))
    return pdf

# Looping over each piece type to train and plot
for piece in range(1,6):
    # Train the MDN for this piece
    model = MDN(input_dim=1, n_hidden=64, n_components=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # train for n_epochs
    n_epochs = 500
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pi, sigma, mu = model(X)
        loss = mdn_loss(pi, sigma, mu, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1}, loss={loss.item():.4f}')

    # save the model weights for each piece
    torch.save(model.state_dict(), 'piece_weights_nn/' + piece_labels[piece]+'_weights.pth')
    y_vals = np.linspace(0, 20, 200)
    pdf_vals = mdn_pdf(piece, y_vals)

    # plotting
    plt.figure(figsize=(6,4))
    plt.plot(y_vals, pdf_vals, lw=2)
    plt.xlabel('Piece Value')
    plt.ylabel('Probability Density')
    plt.title(f'Predicted value distribution for {piece_labels[piece]}')
    plt.grid(alpha=0.3)
    plt.show()
