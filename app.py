import streamlit as st
from PIL import Image
import base64
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool as gap
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import io

from deepchem_utils import (
    encode_with_one_hot, atom_type_one_hot, get_hybridization_one_hot,
    construct_hydrogen_bonding_info, get_atom_formal_charge,
    get_atom_is_in_aromatic_one_hot, get_atom_total_degree_one_hot,
    get_atom_total_num_Hs_one_hot, get_chirality_one_hot, smiles_to_edge_indices
)

# Set up the page configuration
st.set_page_config(
    page_title="graphB3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with information and logo
st.sidebar.title("About")
st.sidebar.info(
    """
    This application allows users to input SMILES strings of molecules and predicts if the molecules can cross the blood-brain barrier.
    """
)
st.sidebar.image("logo.png", use_column_width=True)  # Replace with your logo file

# Main page title and description
st.title("Blood-Brain Barrier Permeability Prediction for molecules")
st.markdown(
    """
    ### Enter your SMILES strings to get predictions.
    This tool uses a graph neural network to predict whether a molecule can cross the blood-brain barrier based on its SMILES representation.
    """
)


# Default SMILES strings
default_smiles = """NCc1c[n+]2ccccc2[nH]1
Nc1ccc(C(=O)O)cc1
O=C(O)c1ccccc1O
Nc1cc(N)c(N)cc1N
CSC(=N)N
NC1(C(=O)O)CCCC1
"""

# Text area for SMILES input
smiles_input = st.text_area("Enter SMILES strings (one per line)", value=default_smiles)


# Define the featurizer class
class MolGraphConvFeaturizerWoLabels:
    def __init__(self):
        pass

    def check_valid_smiles(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError("Invalid SMILES string provided")
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 1:
            raise ValueError("The provided SMILES represents a single atom, not a compound.")

    def featurize_one(self, smile):
        self.check_valid_smiles(smile)
        atom_type = atom_type_one_hot(smile)
        formal_charge = get_atom_formal_charge(smile)
        hybridization = get_hybridization_one_hot(smile)
        hydrogen_bonding_info = construct_hydrogen_bonding_info(smile)
        aromatic = get_atom_is_in_aromatic_one_hot(smile)
        degree = get_atom_total_degree_one_hot(smile)
        num_Hs = get_atom_total_num_Hs_one_hot(smile)
        chirality = get_chirality_one_hot(smile)
        node_features = torch.cat((atom_type, formal_charge, hybridization, hydrogen_bonding_info, 
                                   aromatic, degree, num_Hs, chirality), dim=1).float()  # Convert to float32
        return node_features

    def get_edge_index(self, smile):
        edge_index = smiles_to_edge_indices(smile)
        return edge_index

    def create_data_object(self, smile):
        node_features = self.featurize_one(smile)
        edge_index = self.get_edge_index(smile)
        data = Data(x=node_features, edge_index=edge_index)
        return data

    def get_graphs(self, smiles_list):
        graphs = [self.create_data_object(smile) for smile in smiles_list]
        graph_object = Batch.from_data_list(graphs)
        return graph_object

# Define the model class
class GraphConvModel(torch.nn.Module):
    def __init__(self, embedding_size, num_features):
        super(GraphConvModel, self).__init__()
        torch.manual_seed(4456)
        self.initial_conv = GraphConv(num_features, embedding_size)
        self.conv1 = GraphConv(embedding_size, embedding_size)
        self.conv2 = GraphConv(embedding_size, embedding_size)
        self.conv3 = GraphConv(embedding_size, embedding_size)
        self.out = torch.nn.Linear(embedding_size, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch_index):
        x = x.float()  # Ensure input is float32
        hidden = self.initial_conv(x, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = gap(hidden, batch_index)
        out = self.out(hidden)
        return out

# Load the state_dict
state_dict = torch.load('best_mcc_model_weights.pth')

# Create the model and load the state_dict
model = GraphConvModel(embedding_size=128, num_features=33)  # Example num_features
model.load_state_dict(state_dict)
model.eval()

# Process SMILES input and make predictions
if st.button("Predict"):
    if smiles_input:
        smiles_list = smiles_input.splitlines()
        
        st.write(f"**Received {len(smiles_list)} SMILES strings**")
        
        featurizer = MolGraphConvFeaturizerWoLabels()
        data_batch = featurizer.get_graphs(smiles_list)
        
        # Move data to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        data_batch = data_batch.to(device)
        
        # Predict BBB crossing
        with torch.no_grad():
            data_batch.x = data_batch.x.float()  # Ensure data_batch.x is float32
            logits = model(data_batch.x, data_batch.edge_index, data_batch.batch).cpu().numpy()
            probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        
        results = []
        images = []
        for smiles, probability in zip(smiles_list, probabilities):
            pred_label = "Positive" if probability >= 0.5 else "Negative"
            mol = Chem.MolFromSmiles(smiles)
            img = Draw.MolToImage(mol, size=(150, 150))  # Resize the image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            img_html = f'<img src="data:image/png;base64,{img_b64}" width="150">'
            results.append({"SMILES": smiles, "Molecule": img_html, "Probability": f"{probability[0]:.4f}", "Blood Brain Barrier Permeability": pred_label})
        
        # Display results as a table
        results_df = pd.DataFrame(results)
        st.markdown(results_df.to_html(escape=False), unsafe_allow_html=True)
        
        # Option to download results
        def get_table_download_link(df):
            csv = df.drop(columns=['Molecule']).to_csv(index=False)  # Remove molecule images from the download
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
            return href

        st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed by: Shashank Yadav")
st.markdown("[GitHub](https://github.com/xinformatics)")

# Style adjustments
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f0f5;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
