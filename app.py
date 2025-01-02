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
from torch_geometric.explain import Explainer, GNNExplainer, unfaithfulness

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
CC(=O)OCC[N+](C)(C)C
CC(C[N+](C)(C)C)OC(=O)C
O=C1CCCN1CC#CC[N+](C)(C)C
NC(=O)OCC[N+](C)(C)C
CC(C[N+](C)(C)C)OC(=O)N
Cc1ccc(o1)C[N+](C)(C)C
COC(=O)C1=CCCN(C1)C
O=C1CCCN1CC#CCN1CCCC1
CON=CC1=CCCN(C1)C
CN1CC(=CCC1)C(=O)OCC#C
CCC1C(=O)OCC1Cc1cncn1C
CC(=O)OC1C[NH+]2CCC1CC2
"""

# Initialize the session state for SMILES input if not already set
if 'smiles_input' not in st.session_state:
    st.session_state['smiles_input'] = default_smiles

# Add a Clear button
if st.button("Clear"):
    clear_smiles_input()

# Text area for SMILES input with session state
smiles_input = st.text_area("Enter SMILES strings (one per line)", value=st.session_state['smiles_input'], key='smiles_input')

smiles_input = smiles_input.encode('utf-8').decode('utf-8')

# Convert the SMILES input into a DataFrame internally
if smiles_input:
    # Split the input into lines and remove empty lines
    smiles_list = [line.strip() for line in smiles_input.splitlines() if line.strip()]
    # Create a DataFrame for internal use
    smiles_df = pd.DataFrame(smiles_list, columns=["SMILES"])


import base64
from html import escape

def is_valid_base64(b64_string):
    try:
        base64.b64decode(b64_string, validate=True)
        return True
    except Exception:
        return False


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
        torch.manual_seed(565)
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('best_mcc_model_weights.pth', map_location="cpu")


# Create the model and load the state_dict
model = GraphConvModel(embedding_size=128, num_features=33)  # Example num_features
model.load_state_dict(state_dict)
model.eval()


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
        
        probabilities = []  # To store the probabilities
        predictions = []  # To store binary predictions (0/1)
        
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
                probabilities = torch.sigmoid(torch.tensor(logits)).numpy().flatten().tolist()  # Flatten for appending
                predictions = [1 if prob > 0.5 else 0 for prob in probabilities]  # Threshold at 0.5
        
        # Add invalid SMILES with NaN probabilities and predictions
        for smile in invalid_smiles:
            smiles_df = smiles_df.append({
                "SMILES": smile, 
                "Probability": "N/A", 
                "Prediction": "N/A"
            }, ignore_index=True)
        
        # Append probabilities and predictions to the DataFrame
        if probabilities:
            smiles_df["Probability"] = probabilities[:len(smiles_df)]
            smiles_df["Prediction"] = predictions[:len(smiles_df)]

            """ Explainer"""

            # Function to set the seed
            def set_seed(seed):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # for multi-GPU.
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            # Initialize the GNNExplainer
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(lr=0.0005, epochs=4000),
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
            seed = 454

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

            # print('completed till here!')

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

            def get_atom_indices(index, dict):
                # print(f"Index: {index}, Length of edge_indices: {len(dict['edge_indices'])}, Length of edge_masks: {len(dict['edge_masks'])}")
                
                test_dict = {}
                for key, value in zip(dict['edge_indices'][index], dict['edge_masks'][index]):
                    test_dict[tuple(key.tolist())] = value.item()
                
                window_dict = {f'{i/10}-{(i+1)/10}': [] for i in range(10)}
                
                for key, value in test_dict.items():
                    window_key = f'{int(value*10)/10}-{int(value*10 + 1)/10}'
                    unique_numbers = sorted(set(sum([key], ())))
                    window_dict[window_key].extend(unique_numbers)
                    
                for k in window_dict:
                    window_dict[k] = sorted(set(window_dict[k]))
                    
                return window_dict
            
            index_list = smiles_df.index.tolist()

            atom_indices = {index: get_atom_indices(index, nchem_exp_dict_avg) for index in index_list}
################################################################################################################################################################
            def generate_gradient_colors(start_color, end_color, n_colors):
                """Generate gradient colors between two RGB colors."""
                start_color = np.array(start_color)
                end_color = np.array(end_color)
                colors = [tuple((start_color + (end_color - start_color) * (i / (n_colors - 1))).tolist()) for i in range(n_colors)]
                return colors

##################################################################################################################################################################
            def highlight_molecule(smiles, importance_dict, label, image_size=(300, 300), highlight=False):
                """Generate a molecule image with optional highlights."""
                # Create molecule from SMILES
                m = Chem.MolFromSmiles(smiles)
                if not m:
                    raise ValueError(f"Cannot create molecule from SMILES: {smiles}")

                # # Define gradients for positive and negative labels
                # if label == 1:  # Positive
                #     light_color = [0.8, 1.0, 0.8]  # Light green
                #     dark_color = [0.0, 0.6, 0.0]   # Dark green
                # else:  # Negative
                #     light_color = [1.0, 0.8, 0.8]  # Light red
                #     dark_color = [0.6, 0.0, 0.0]   # Dark red

                # Define gradients for positive and negative labels
                if label == 1:  # Positive
                    # low_importance = [1.0, 1.0, 0.5]  # Yellow for low importance
                    # medium_importance = [0.5, 1.0, 0.5]  # Green for medium importance
                    # high_importance = [0.0, 0.4, 0.0]  # Dark green for high importance
                    low_importance = [0.9, 1.0, 0.6]  # Light yellow-green
                    medium_importance = [0.2, 0.9, 0.3]  # Vivid green
                    high_importance = [0.0, 0.3, 0.0]  # Dark green
                else:  # Negative
                    # low_importance = [1.0, 1.0, 0.5]  # Yellow for low importance
                    # medium_importance = [1.0, 0.5, 0.5]  # Light red for medium importance
                    # high_importance = [0.6, 0.0, 0.0]  # Dark red for high importance
                    low_importance = [1.0, 0.9, 0.7]  # Light orange
                    medium_importance = [1.0, 0.4, 0.3]  # Vivid red-orange
                    high_importance = [0.5, 0.0, 0.0]  # Crimson


                # Generate gradient: Combine low-medium and medium-high ranges
                gradient_colors = (
                    generate_gradient_colors(low_importance, medium_importance, 5) +
                    generate_gradient_colors(medium_importance, high_importance, 5)
                )

                # Prepare highlight colors
                highlight_colors = {}
                if highlight and importance_dict:
                    for i, (score_range, atom_indices) in enumerate(importance_dict.items()):
                        if atom_indices:
                            color = gradient_colors[i]
                            # print(color)
                            for atom_index in atom_indices:
                                highlight_colors[atom_index] = color

                # Configure drawing options
                drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
                opts = drawer.drawOptions()

                # Ensure highlighted atoms override defaults
                highlight_atoms = list(highlight_colors.keys())  # Atoms to be highlighted
                highlight_atom_colors = highlight_colors  # Custom highlight colors

                # print()

                # Draw the molecule

                if highlight:
                    drawer.drawOptions().useBWAtomPalette()
                    drawer.DrawMolecule(
                        m,
                        highlightAtoms=highlight_atoms,
                        highlightAtomColors=highlight_atom_colors
                    )
                else:
                    # drawer.drawOptions().useBWAtomPalette()
                    drawer.DrawMolecule(
                        m,
                        highlightAtoms=None,
                        highlightAtomColors=None
                    )
                drawer.FinishDrawing()
                img_binary = drawer.GetDrawingText()


                # Convert image to Base64
                buf = io.BytesIO()
                buf.write(img_binary)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()
                return img_b64

            # def highlight_molecule(smiles, importance_dict, label, image_size=(600, 600), highlight=False):
            #     """Generate a molecule image with optional highlights."""
            #     # Create molecule from SMILES
            #     m = Chem.MolFromSmiles(smiles)
            #     if not m:
            #         raise ValueError(f"Cannot create molecule from SMILES: {smiles}")

            #     # If highlight is False, return a plain molecule image
            #     if not highlight:
            #         drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
            #         drawer.DrawMolecule(m)
            #         drawer.FinishDrawing()
            #         img_binary = drawer.GetDrawingText()

            #         # # Save binary data for inspection
            #         # with open("test_image_raw.png", "wb") as f:
            #         #     f.write(img_binary)

            #         # Convert to Base64
            #         buf = io.BytesIO()
            #         buf.write(img_binary)
            #         buf.seek(0)
            #         img_b64 = base64.b64encode(buf.read()).decode()
            #         return img_b64

            #     # Highlight logic
            #     # light_red = [1.0, 1.0, 1.0]
            #     # dark_red = [1.0, 0.1, 0.1]

            #     light_color = [0.9, 0.9, 1.0]
            #     dark_color  = [0.4, 0.0, 0.8]  



            #     # gradient_colors = generate_gradient_colors(light_red, dark_red, 10)
            #     gradient_colors = generate_gradient_colors(light_color, dark_color, 10)

            #     # Prepare highlight colors
            #     highlight_colors = {}
            #     if importance_dict:
            #         for i, (score_range, atom_indices) in enumerate(importance_dict.items()):
            #             if atom_indices:
            #                 color = gradient_colors[i]
            #                 for atom_index in atom_indices:
            #                     highlight_colors[atom_index] = color

            #     # Draw molecule with highlights
            #     drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
            #     drawer.DrawMolecule(m, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors)
            #     drawer.FinishDrawing()
            #     img_binary = drawer.GetDrawingText()

            #     # Save highlighted image for inspection
            #     # with open("test_image_highlighted.png", "wb") as f:
            #     #     f.write(img_binary)

            #     # Convert to Base64
            #     buf = io.BytesIO()
            #     buf.write(img_binary)
            #     buf.seek(0)
            #     img_b64 = base64.b64encode(buf.read()).decode()
            #     return img_b64
##########################################################################################################################################################

            
            # def highlight_molecule(smiles, importance_dict, label, image_size=(300, 300), highlight=False):
                
            #     # Create molecule from SMILES
            #     m = Chem.MolFromSmiles(smiles)
            #     if not m:
            #         raise ValueError(f"Cannot create molecule from SMILES: {smiles}")
                
            #     if not highlight:
            #         drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
            #         drawer.DrawMolecule(m)
            #         drawer.FinishDrawing()
            #         return drawer.GetDrawingText()
                
            #     # Define colors
            #     light_red = [1.0, 1.0, 1.0]  # Light red
            #     dark_red = [1.0, 0.1, 0.1]   # Dark red
                
            #     # Generate gradient colors
            #     label_1_colors = generate_gradient_colors(light_red, dark_red, 10)
                
            #     # Define colors
            #     light_blue = [1.0, 1.0, 1.0]  # Light blue
            #     dark_blue = [0.2, 0.4, 0.9]  # Dark blue
                
            #     # Generate gradient colors
            #     label_0_colors = generate_gradient_colors(light_blue, dark_blue, 10)
                
            #     # Define color schemes
            #     if label == 1:
            #         gradient_colors = label_1_colors
            #     else:
            #         gradient_colors = label_0_colors
                    
            #     # Set sssAtoms property to None initially
            #     m.__sssAtoms = None
                
            #     # Create a drawer object
            #     drawer = Draw.MolDraw2DCairo(image_size[0], image_size[1])
                
            #     # Prepare highlight colors
            #     highlight_colors = {}
            #     if importance_dict:
            #         for i, (score_range, atom_indices) in enumerate(importance_dict.items()):
            #             if atom_indices:  # only assign color if there are atoms to highlight 
            #                 color = gradient_colors[i]  # Get the color for the current score range
            #                 for atom_index in atom_indices:
            #                     highlight_colors[atom_index] = color

            #     # Set the sssAtoms property with the atom indices to be highlighted
            #     if highlight_colors:
            #         m.__sssAtoms = list(highlight_colors.keys())
    
            #         # Draw molecule with highlight
            #         opts = drawer.drawOptions()
            #         drawer.DrawMolecule(m, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors)
            #         drawer.FinishDrawing()
            #         img = drawer.GetDrawingText()
                
            #     else:
            #         img = Draw.MolToImage(m, size=image_size)

            #     buf = io.BytesIO()
            #     if highlight:
            #         buf.write(img)
            #         buf.seek(0)
            #         img_pil = Image.open(buf)
            #     else:
            #         img_pil = img

            #     buf = io.BytesIO()
            #     img_pil.save(buf, format="PNG")
            #     buf.seek(0)
            #     img_b64 = base64.b64encode(buf.read()).decode()

            #     return img_b64

#####################################################################################################################################################
            smiles_images = []  # Collect rows for the final DataFrame

            for idx in smiles_df.index:
                smiles = smiles_df.loc[idx, 'SMILES']
                label = smiles_df.loc[idx, 'Prediction']  # Ensure this contains 0/1 predictions
                probability = smiles_df.loc[idx, 'Probability']  # Ensure this contains probabilities
                importance_dict = get_atom_indices(idx, nchem_exp_dict_avg)  # Ensure this returns a valid dictionary

                try:
                    original_img_b64 = highlight_molecule(smiles, importance_dict, label, image_size=(400, 400), highlight=False)
                    highlight_img_b64 = highlight_molecule(smiles, importance_dict, label, image_size=(400, 400), highlight=True)

                    if not (is_valid_base64(original_img_b64) and is_valid_base64(highlight_img_b64)):
                        raise ValueError("Invalid Base64 encoding")

                    smiles_images.append({
                        "SMILES": smiles,
                        "Original Molecule": f'<img src="data:image/png;base64,{escape(original_img_b64)}" width="400">',
                        "Probability": f"{probability:.4f}",
                        "Status": "Positive" if label == 1 else "Negative",
                        "Highlighted Molecule by Explainer": f'<img src="data:image/png;base64,{escape(highlight_img_b64)}" width="400">'
                    })

                except Exception as e:
                    smiles_images.append({
                        "SMILES": smiles,
                        "Original Molecule": f"Error: {e}",
                        "Probability": f"{probability:.4f}" if probability else "N/A",
                        "Status": "Error",
                        "Highlighted Molecule by Explainer": f"Error: {e}"
                    })

            # Convert list of dictionaries to DataFrame
            results_df = pd.DataFrame(smiles_images)

            # Function to generate a download link for the DataFrame (without images)
            def get_table_download_link(df):
            # Remove molecule images from the download
                csv = df.drop(columns=['Original Molecule', 'Highlighted Molecule by Explainer']).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
                return href

            # Display the DataFrame with images in the app
            st.markdown(results_df.to_html(escape=False), unsafe_allow_html=True)

            # Provide option to download the DataFrame
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
