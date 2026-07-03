import os
import glob
import pandas as pd
import networkx as nx

class MetadataGraphBuilder:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Graph Builder with a pandas DataFrame containing the dataset metadata.
        """
        self.df = df
        
    def build_hetero_graph(self) -> nx.Graph:
        """
        Builds a scalable bipartite/heterogeneous graph.
        - Nodes: Articles (news_id), Sources (source_type), Categories (category).
        - Edges: (Article -> Source), (Article -> Category).
        This topology is highly optimized for GraphSAGE to learn source reliability and category bias.
        """
        G = nx.Graph()
        
        print(f"Constructing Metadata Graph for {len(self.df)} articles...")
        
        for _, row in self.df.iterrows():
            # Create unique identifiers for each node type to avoid collisions
            article_id = f"ARTICLE_{row['news_id']}"
            source_id = f"SOURCE_{row.get('source_type', 'unknown')}"
            category_id = f"CAT_{row.get('category', 'unknown')}"
            
            # 1. Add Article Node (with its ground truth label for training)
            label = int(row.get('label', 0))
            G.add_node(article_id, type='article', label=label)
            
            # 2. Add Source and Category Nodes
            G.add_node(source_id, type='source')
            G.add_node(category_id, type='category')
            
            # 3. Add Relational Edges
            G.add_edge(article_id, source_id, relation='published_by')
            G.add_edge(article_id, category_id, relation='belongs_to')
            
        return G

if __name__ == "__main__":
    # Quick sanity test
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = os.path.join(project_root, 'data set')
    
    if os.path.exists(test_dir):
        print(f"Loading data from {test_dir} to test graph building...")
        all_files = glob.glob(os.path.join(test_dir, "*.csv"))
        df_list = [pd.read_csv(f, on_bad_lines='skip') for f in all_files]
        df = pd.concat(df_list, ignore_index=True).dropna(subset=['news_context', 'label'])
        
        # Build the graph
        builder = MetadataGraphBuilder(df)
        metadata_graph = builder.build_hetero_graph()
        
        # Output graph statistics
        print("\nGraph Construction Complete!")
        print(f"Total Nodes: {metadata_graph.number_of_nodes()}")
        print(f"Total Edges: {metadata_graph.number_of_edges()}")
        
        # Count node types
        article_nodes = [n for n, attr in metadata_graph.nodes(data=True) if attr.get('type') == 'article']
        source_nodes = [n for n, attr in metadata_graph.nodes(data=True) if attr.get('type') == 'source']
        cat_nodes = [n for n, attr in metadata_graph.nodes(data=True) if attr.get('type') == 'category']
        
        print(f" - Article Nodes: {len(article_nodes)}")
        print(f" - Source Nodes: {len(source_nodes)}")
        print(f" - Category Nodes: {len(cat_nodes)}")
    else:
        print(f"Data directory not found at {test_dir}")
