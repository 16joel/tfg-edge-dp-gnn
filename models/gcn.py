import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for node classification.

    Architecture:
    - Input: Node features (num_features) and edge_index
    - 2 GCN layers with ReLU activation
    - Output layer for node classification (num_classes)

    Notes
    -----
    El mètode get_embeddings s'ha afegit directament a la classe (en lloc
    d'injectar-lo dinàmicament via monkey-patch al mòdul d'atac) per evitar
    fragilitat: si la classe es reimportava o redefinida en qualsevol punt
    del programa, el mètode injectat desapareixia silenciosament.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialize GCN model for node classification.

        Parameters
        ----------
        in_channels : int
            Number of input node features
        hidden_channels : int
            Number of hidden dimensions
        out_channels : int
            Number of output classes
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass through GCN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, in_channels)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)

        Returns
        -------
        logits : torch.Tensor
            Node class logits (num_nodes, out_channels)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # logits for node classification

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Retorna els embeddings de la capa oculta (post-conv1, post-ReLU).

        Aquests embeddings representen la codificació del GCN per a cada node
        condicionada a l'estructura del graf d'entrada. A l'atac d'inferència
        (no supervisat) s'usen per calcular la similitud cosinus entre parelles
        de nodes com a puntuació de versemblança d'aresta; no s'entrena cap
        classificador.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, in_channels)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)

        Returns
        -------
        embeddings : torch.Tensor
            Hidden representations (num_nodes, hidden_channels)
        """
        self.eval()
        embeddings = self.conv1(x, edge_index)
        embeddings = F.relu(embeddings)
        return embeddings