import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import SAGEConv
from dgl.dataloading import NeighborSampler, EdgeDataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import math
import requests
import zipfile
import io
import os

# --- Configuration ---
DATASET_NAME = 'ml-100k' # Example dataset: MovieLens 100k
DATA_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATA_DIR = './ml-100k'
EMBEDDING_DIM = 64
NUM_LAYERS = 2
FANOUTS = [10, 10] # Number of neighbors to sample per layer
BATCH_SIZE = 512
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
MARGIN = 0.1 # Margin for max-margin loss
NUM_NEGATIVE_SAMPLES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading and Preprocessing ---

def download_and_extract_data(url, dir_name):
    if not os.path.exists(dir_name):
        print(f"Downloading and extracting {DATASET_NAME}...")
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("Download complete.")
    else:
        print(f"Dataset found at {dir_name}.")

def load_movielens_data(dir_name):
    ratings_file = os.path.join(dir_name, 'u.data')
    ratings_df = pd.read_csv(
        ratings_file,
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    # Treat ratings >= 4 as positive interactions
    ratings_df = ratings_df[ratings_df['rating'] >= 4]

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['user_id'])
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['item_id'])

    num_users = ratings_df['user_idx'].nunique()
    num_items = ratings_df['item_idx'].nunique()

    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of positive interactions: {len(ratings_df)}")

    # Split data (using edge IDs for simplicity)
    eids = np.arange(len(ratings_df))
    train_eids, test_eids = train_test_split(eids, test_size=0.1, random_state=42)

    return ratings_df, num_users, num_items, user_encoder, item_encoder, train_eids, test_eids

def build_graph(ratings_df, num_users, num_items):
    user_indices = torch.tensor(ratings_df['user_idx'].values, dtype=torch.int64)
    item_indices = torch.tensor(ratings_df['item_idx'].values, dtype=torch.int64)

    graph_data = {
        ('user', 'rates', 'item'): (user_indices, item_indices),
        ('item', 'rated_by', 'user'): (item_indices, user_indices)
    }
    num_nodes_dict = {'user': num_users, 'item': num_items}

    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    return g

# --- Model Definition ---

class PinSAGEModel(nn.Module):
    def __init__(self, graph, node_features, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.graph = graph
        self.node_features = node_features
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))

        # Output layer
        self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type='mean'))

        self.dropout = nn.Dropout(0.5)
        self.hidden_dim = hidden_dim

    def get_repr(self, blocks, input_nodes):
        h = self.node_features[input_nodes].to(DEVICE)
        # Apply initial linear transformation if needed (or directly use embeddings)
        # h = self.input_proj(h) # Optional input projection

        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            # Ensure block is on the correct device
            block = block.to(DEVICE)
            
            # Handle node features - SAGEConv expects single tensor input
            # We extract relevant node features based on the block's structure
            current_layer_input_nodes = block.srcdata[dgl.NID]
            h_src = h[input_nodes.index_select(0, current_layer_input_nodes)] # Select correct features
            
            h_dst = h[input_nodes.index_select(0, block.dstdata[dgl.NID])] # Select correct dest features
            
            # Apply SAGEConv
            h = layer(block, (h_src, h_dst))

            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def forward(self, pos_graph, neg_graph, blocks, input_node_ids):
        # Get embeddings for all nodes involved in the blocks
        node_embeddings = self.get_repr(blocks, input_node_ids)

        # Extract embeddings for positive and negative graph edges
        pos_src_nids = pos_graph.srcdata[dgl.NID]
        pos_dst_nids = pos_graph.dstdata[dgl.NID]
        neg_src_nids = neg_graph.srcdata[dgl.NID]
        neg_dst_nids = neg_graph.dstdata[dgl.NID]
        
        # Need mapping from original graph NID to the current subgraph's internal index
        input_node_map = {nid.item(): i for i, nid in enumerate(input_node_ids)}

        pos_src_indices = torch.tensor([input_node_map[nid.item()] for nid in pos_src_nids], device=DEVICE)
        pos_dst_indices = torch.tensor([input_node_map[nid.item()] for nid in pos_dst_nids], device=DEVICE)
        neg_src_indices = torch.tensor([input_node_map[nid.item()] for nid in neg_src_nids], device=DEVICE)
        neg_dst_indices = torch.tensor([input_node_map[nid.item()] for nid in neg_dst_nids], device=DEVICE)

        pos_src_emb = node_embeddings[pos_src_indices]
        pos_dst_emb = node_embeddings[pos_dst_indices]
        neg_src_emb = node_embeddings[neg_src_indices]
        neg_dst_emb = node_embeddings[neg_dst_indices]

        return pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb

# --- Loss Function ---
def compute_loss(pos_src, pos_dst, neg_src, neg_dst, margin):
    # Using dot product similarity
    pos_score = torch.sum(pos_src * pos_dst, dim=1)
    neg_score = torch.sum(neg_src * neg_dst, dim=1)

    # Max-margin loss
    loss = F.relu(margin - pos_score + neg_score).mean()
    return loss

# --- Training Loop ---

def train(model, graph, train_eids_dict, optimizer, num_epochs, batch_size, fanouts, num_negative_samples, margin, device):
    sampler = NeighborSampler(fanouts)
    # Negative sampler for link prediction
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_samples)

    # Create DataLoader for edges
    dataloader = EdgeDataLoader(
        graph, train_eids_dict, sampler,
        negative_sampler=neg_sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True, # Usually good practice with GPU
        num_workers=0 # Set > 0 for parallel data loading if needed
    )

    # Type specific initial features (learnable embeddings)
    node_features = {}
    for ntype in graph.ntypes:
        emb = nn.Embedding(graph.num_nodes(ntype), EMBEDDING_DIM, sparse=False) # Using dense embeddings
        nn.init.xavier_uniform_(emb.weight)
        node_features[ntype] = emb.weight.to(device) # Move embeddings to device

    # Make features trainable parameters
    model.node_features = nn.ParameterDict({ntype: nn.Parameter(feat) for ntype, feat in node_features.items()})
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=LEARNING_RATE)


    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            # Ensure graphs and input_nodes are on the correct device
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            # input_nodes are dictionary {ntype: tensor}
            # Extract combined tensor of unique input node IDs across all types
            all_input_nodes_list = []
            node_map_for_feature_lookup = {} # Maps original NID to index in combined tensor
            current_idx = 0
            for ntype in input_nodes:
                nids = input_nodes[ntype].to(device)
                all_input_nodes_list.append(nids)
                for nid in nids:
                    node_map_for_feature_lookup[(ntype, nid.item())] = current_idx
                    current_idx += 1
            
            # Concatenate features (careful with order)
            combined_features = torch.cat([model.node_features[ntype][input_nodes[ntype].to(device)] for ntype in input_nodes], dim=0)

            # --- Re-defining the Model with HeteroGraphConv ---
            class HeteroPinSAGEModel(nn.Module):
                def __init__(self, graph, feat_dict, hidden_dim, out_dim, n_layers, etypes):
                    super().__init__()
                    self.feat_dict = feat_dict # Should be nn.ParameterDict
                    self.layers = nn.ModuleList()
                    
                    # Input layer proj (optional, if input dims differ)
                    # self.input_proj = nn.ModuleDict({
                    #     ntype: nn.Linear(feat.shape[1], hidden_dim) for ntype, feat in feat_dict.items()
                    # })

                    self.layers.append(dgl.nn.HeteroGraphConv({
                        etype: SAGEConv(hidden_dim, hidden_dim, 'mean')
                        for etype in etypes}, aggregate='sum'))

                    for _ in range(n_layers - 2):
                         self.layers.append(dgl.nn.HeteroGraphConv({
                            etype: SAGEConv(hidden_dim, hidden_dim, 'mean')
                            for etype in etypes}, aggregate='sum'))

                    self.layers.append(dgl.nn.HeteroGraphConv({
                        etype: SAGEConv(hidden_dim, out_dim, 'mean')
                        for etype in etypes}, aggregate='sum'))
                    
                    self.dropout = nn.Dropout(0.5)
                
                def forward(self, blocks, input_features_dict):
                    h_dict = input_features_dict
                    # h_dict = {ntype: self.input_proj[ntype](feat) for ntype, feat in h_dict.items()}
                    # h_dict = {ntype: F.relu(feat) for ntype, feat in h_dict.items()}

                    for i, layer in enumerate(self.layers):
                        if i > 0:
                            h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
                        
                        # Ensure blocks are on the correct device before passing to layer
                        blocks_on_device = [b.to(DEVICE) for b in blocks]
                        h_dict = layer(blocks_on_device[i], h_dict)
                        
                        if i < len(self.layers) - 1:
                            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
                    return h_dict # Returns dict of output features {ntype: tensor}


            # --- Restarting training loop logic with HeteroPinSAGEModel ---

            input_features = {ntype: model.feat_dict[ntype][input_nodes[ntype].to(DEVICE)] for ntype in input_nodes}

            output_embs_dict = model(blocks, input_features)

            # Extract embeddings for positive and negative graph edges based on output nodes
            pos_src_emb = output_embs_dict['user'][pos_graph.nodes['user'].data[dgl.NID].long()] # Assuming 'rates' edge type
            pos_dst_emb = output_embs_dict['item'][pos_graph.nodes['item'].data[dgl.NID].long()]
            neg_src_emb = output_embs_dict['user'][neg_graph.nodes['user'].data[dgl.NID].long()] # Assuming 'rates' edge type
            neg_dst_emb = output_embs_dict['item'][neg_graph.nodes['item'].data[dgl.NID].long()]

            loss = compute_loss(pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb, margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    print("Training finished.")
    return model # Return the trained model

# --- Main Execution ---
if __name__ == "__main__":
    download_and_extract_data(DATA_URL, DATASET_NAME)
    ratings_df, num_users, num_items, user_encoder, item_encoder, train_eids, test_eids = load_movielens_data(DATASET_NAME)
    
    graph = build_graph(ratings_df, num_users, num_items)
    graph = graph.to(DEVICE)

    # Map edge IDs to the specific edge type they belong to
    train_eids_dict = {('user', 'rates', 'item'): torch.tensor(train_eids, dtype=torch.int64)}
    # test_eids_dict = {('user', 'rates', 'item'): torch.tensor(test_eids)} # For potential evaluation

    # Prepare initial features (embeddings)
    node_features = {}
    for ntype in graph.ntypes:
        emb = nn.Embedding(graph.num_nodes(ntype), EMBEDDING_DIM, sparse=False)
        nn.init.xavier_uniform_(emb.weight)
        # Don't move to device yet, handled inside training loop via ParameterDict
        node_features[ntype] = nn.Parameter(emb.weight) 
        
    node_feature_params = nn.ParameterDict(node_features)

    # Instantiate the HeteroPinSAGEModel
    model = HeteroPinSAGEModel(
        graph=graph,
        feat_dict=node_feature_params,
        hidden_dim=EMBEDDING_DIM, # Using same dim for hidden layers
        out_dim=EMBEDDING_DIM,
        n_layers=NUM_LAYERS,
        etypes=graph.canonical_etypes
    ).to(DEVICE)

    # Define optimizer (pass model.parameters())
    optimizer = torch.optim.Adam
    
    # Train the model
    trained_model = train(
        model=model,
        graph=graph,
        train_eids_dict=train_eids_dict,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        fanouts=FANOUTS,
        num_negative_samples=NUM_NEGATIVE_SAMPLES,
        margin=MARGIN,
        device=DEVICE
    )

    # --- Inference/Recommendation Example ---
    print("\n--- Generating Recommendations (Example) ---")
    
    trained_model.eval()
    with torch.no_grad():
        
        item_embeddings = {}
        if isinstance(trained_model, HeteroPinSAGEModel):
             # Perform full-graph inference (simpler for small graphs)
            full_graph_features = {ntype: trained_model.feat_dict[ntype].to(DEVICE) for ntype in graph.ntypes}
            all_node_embeddings = trained_model.module.forward(graph, full_graph_features) if isinstance(trained_model, nn.DataParallel) else trained_model.forward(graph, full_graph_features) # Adjust based on DP
            item_embeddings = all_node_embeddings['item'].cpu().numpy()
        else:
             # Fallback or error if model structure is unexpected
             print("Warning: Inference assumes HeteroPinSAGEModel structure.")
             # A full implementation would require a separate inference function using sampling


        if len(item_embeddings) > 0:
            # Example: Find top 5 similar items for item_id 0 (encoded)
            query_item_idx = 0
            k = 5

            query_embedding = item_embeddings[query_item_idx]
            
            # Calculate cosine similarity (or use dot product if normalized)
            similarities = np.dot(item_embeddings, query_embedding) / (np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(query_embedding))
            
            # Get top k indices (excluding the item itself)
            sorted_indices = np.argsort(similarities)[::-1]
            top_k_indices = [idx for idx in sorted_indices if idx != query_item_idx][:k]

            # Convert indices back to original item IDs
            original_query_id = item_encoder.inverse_transform([query_item_idx])[0]
            recommended_ids = item_encoder.inverse_transform(top_k_indices)

            print(f"Recommendations for item ID {original_query_id} (internal index {query_item_idx}):")
            for i, rec_id in enumerate(recommended_ids):
                print(f"  {i+1}. Item ID: {rec_id} (Similarity: {similarities[top_k_indices[i]]:.4f})")

        else:
            print("Could not generate item embeddings for recommendations.")