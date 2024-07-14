import streamlit as st
from PIL import Image
import base64
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import io
import pickle
import numpy as np
import random
import time
from torch_geometric.explain import Explainer, GNNExplainer, unfaithfulness
from torch_geometric.nn import global_add_pool as gap

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

# Define a function to clear the SMILES input
def clear_smiles_input():
    st.session_state['smiles_input'] = ""

# Default SMILES strings
default_smiles = """NCc1c[n+]2ccccc2[nH]1
Nc1ccc(C(=O)O)cc1
O=C(O)c1ccccc1O
Nc1cc(N)c(N)cc1N
CSC(=N)N
NC1(C(=O)O)CCCC1
"""

# Initialize the session state for SMILES input if not already set
if 'smiles_input' not in st.session_state:
    st.session_state['smiles_input'] = default_smiles

# Add a Clear button
if st.button("Clear"):
    clear_smiles_input()

# Text area for SMILES input with session state
smiles_input = st.text_area("Enter SMILES strings (one per line)", value=st.session_state['smiles_input'], key='smiles_input')

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
        graphs = []
        invalid_smiles = []
        for smile in smiles_list:
            try:
                graphs.append(self.create_data_object(smile))
            except ValueError as e:
                invalid_smiles.append(smile)
        return Batch.from_data_list(graphs) if graphs else None, invalid_smiles

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('best_mcc_model_weights.pth', map_location=device)
# state_dict = torch.load('best_mcc_model_weights.pth')

# Create the model and load the state_dict
model = GraphConvModel(embedding_size=128, num_features=33)  # Example num_features
model.load_state_dict(state_dict)
model.eval()

def generate_image(smiles, unique_number, label, image_size=(150, 150), highlight=False):
    m = Chem.MolFromSmiles(smiles)
    if not m:
        raise ValueError(f"Cannot create molecule from SMILES: {smiles}")

    if highlight:
        if label == 0:
            color = (0.68, 0.85, 0.90)  # Light blue for BBB-
            highlight_colors = {atom_idx: color for atom_idx in unique_number}
        else:
            highlight_colors = {}

        drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
        opts = drawer.drawOptions()
        drawer.DrawMolecule(m, highlightAtoms=unique_number, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
    else:
        img = Draw.MolToImage(m, size=image_size)

    buf = io.BytesIO()
    if highlight:
        buf.write(img)
        buf.seek(0)
        img_pil = Image.open(buf)
    else:
        img_pil = img

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return img_b64

# Function to set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Process SMILES input and make predictions
# Process SMILES input and make predictions
if st.button("Predict"):
    if smiles_input:
        smiles_list = smiles_input.splitlines()
        
        st.write(f"**Received {len(smiles_list)} SMILES strings**")
        
        # Show model loading
        status_text = st.text("Loading model...")
        time.sleep(1)  # Delay for 2 seconds

        featurizer = MolGraphConvFeaturizerWoLabels()
        data_batch, invalid_smiles = featurizer.get_graphs(smiles_list)
        
        results = []
        
        # Add invalid SMILES to results
        for smile in invalid_smiles:
            results.append({
                "SMILES": smile,
                "Original Molecule": "Invalid SMILES",
                "Probability": "N/A",
                "Blood Brain Barrier Permeability": "N/A",
                "Highlighted Molecule": "N/A"
            })
        
        if data_batch is not None:
            # Move data to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            data_batch = data_batch.to(device)
            
            # Show model prediction update
            status_text.text("Model predicting BBB permeability...")
            time.sleep(2)  # Delay for 2 seconds
            
            # Predict BBB crossing
            with torch.no_grad():
                data_batch.x = data_batch.x.float()  # Ensure data_batch.x is float32
                logits = model(data_batch.x, data_batch.edge_index, data_batch.batch).cpu().numpy()
                probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
            
            # Initialize the GNNExplainer
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=1000),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            # Set seed for reproducibility
            seed = 123

            all_explanations = []
            unfaith_metrics = []

            # Show GNN explainer start
            status_text.text("Starting GNN explainer...")
            time.sleep(1)  # Delay for 2 seconds

            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with torch.no_grad():
                with torch.set_grad_enabled(True):
                    for idx, data in enumerate(DataLoader(data_batch, batch_size=1, shuffle=False)):
                        set_seed(seed)
                        
                        X = data.x.float()
                        Edge_index = data.edge_index.long()
                        Batch_index = data.batch.long()
                    
                        explanation = explainer(x=X, edge_index=Edge_index, batch_index=Batch_index)
                        metric = unfaithfulness(explainer, explanation)
                    
                        all_explanations.append(explanation)
                        unfaith_metrics.append(metric)
                        
                        progress_percentage = (idx + 1) / (len(smiles_list) - len(invalid_smiles))
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Running GNN explainer: {int(progress_percentage * 100)}% complete")

            status_text.text("GNN explainer complete!")
            test_chem_all = Batch.from_data_list(all_explanations)

            chem_loader_exp = DataLoader(test_chem_all, batch_size=1, shuffle=False)

            edge_masks_model = []

            for i in chem_loader_exp:
                edge_score = i.edge_mask
                edge_score = edge_score.t()
                edge_masks_model.append(edge_score)

            edge_indices_model = []

            for i in chem_loader_exp:
                edge_idx = i.edge_index
                edge_idx = edge_idx.t()
                edge_indices_model.append(edge_idx)

            edge_indices_cpu_model = [tensor.to('cpu') for tensor in edge_indices_model]
            edge_masks_cpu_model = [tensor.to('cpu') for tensor in edge_masks_model]

            nchem_exp_dict = {
                'edge_indices': edge_indices_cpu_model,
                'edge_masks': edge_masks_cpu_model
            }

            def average_bidirectional_edges(graphs_dict):
                result = {'edge_indices': [], 'edge_masks': []}
                
                for edges, masks in zip(graphs_dict['edge_indices'], graphs_dict['edge_masks']):
                    averaged_edges = []
                    averaged_masks = []
                    
                    for i in range(0, edges.size(0), 2):
                        # Take the first edge of the bidirectional pair
                        edge = edges[i].tolist()
                        # Average the scores of the bidirectional pair
                        avg_masks = (masks[i].item() + masks[i+1].item()) / 2
                        
                        averaged_edges.append(edge)
                        averaged_masks.append(avg_masks)
                    
                    result['edge_indices'].append(torch.tensor(averaged_edges))
                    result['edge_masks'].append(torch.tensor(averaged_masks))
                
                return result

            nchem_exp_dict_avg = average_bidirectional_edges(nchem_exp_dict)

            def get_unique_list(index, dict):
                test_dict = {}
                for key, value in zip(dict['edge_indices'][index], dict['edge_masks'][index]):
                    test_dict[tuple(key.tolist())] = value.item()

                trimmed_dict = {key: value for key, value in test_dict.items() if value > 0.5}  # threshold for edge masks
                unique_numbers = sorted(set(sum(trimmed_dict.keys(), ())))

                return unique_numbers

            for i, (smiles, probability) in enumerate(zip(smiles_list, probabilities)):
                if smiles in invalid_smiles:
                    continue

                label = 1 if probability >= 0.5 else 0
                unique_number = get_unique_list(i, nchem_exp_dict_avg)
                try:
                    original_img_b64 = generate_image(smiles, unique_number, label, image_size=(150, 150), highlight=False)
                    highlight_img_b64 = generate_image(smiles, unique_number, label, image_size=(150, 150), highlight=True)
                    original_img_html = f'<img src="data:image/png;base64,{original_img_b64}" width="150">'
                    highlight_img_html = f'<img src="data:image/png;base64,{highlight_img_b64}" width="150">'
                except Exception as e:
                    original_img_html = f"Error generating image: {e}"
                    highlight_img_html = f"Error generating image: {e}"
                
                results.append({
                    "SMILES": smiles,
                    "Original Molecule": original_img_html,
                    "Probability": f"{probability[0]:.4f}",
                    "Blood Brain Barrier Permeability": "Positive" if label == 1 else "Negative",
                    "Highlighted Molecule": highlight_img_html
                })

        # Display results as a table
        results_df = pd.DataFrame(results)
        st.markdown(results_df.to_html(escape=False), unsafe_allow_html=True)

        # Option to download results
        def get_table_download_link(df):
            csv = df.drop(columns=['Original Molecule', 'Highlighted Molecule']).to_csv(index=False)  # Remove molecule images from the download
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
            return href

        st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed by: Shashank Yadav, Biomedical Engineering, University of Arizona")
st.markdown("Conceptualized by: Translational Biology Laboratory - IIIT Delhi")
# st.markdown("[GitHub](https://github.com/xinformatics)")

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
