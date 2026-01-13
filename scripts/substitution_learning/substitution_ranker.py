"""
Substitution Ranker
Combines embedding similarity and PMI co-occurrence using Reciprocal Rank Fusion.
Provides category-aware filtering and confidence scoring.
"""

import yaml
import json
import torch
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
import torch.nn.functional as F

# Grain substitution constraints
GRAIN_GROUPS = {
    'asian_grains': ['Rice -Chamal-', 'Beaten Rice -Chiura-'],
    'western_grains': ['Wheat', 'Corn', 'Bread'],
    'flexible': ['Corn'],  # Can substitute across groups
}

def get_grain_group(ingredient):
    """Get grain group for ingredient"""
    for group, members in GRAIN_GROUPS.items():
        if ingredient in members:
            return group
    return None

def is_valid_grain_substitution(query_ingredient, candidate):
    """Check if grain substitution is valid"""
    query_group = get_grain_group(query_ingredient)
    candidate_group = get_grain_group(candidate)
    
    # If either not in grain groups, allow
    if not query_group or not candidate_group:
        return True
    
    # If either is flexible, allow
    if query_group == 'flexible' or candidate_group == 'flexible':
        return True
    
    # Otherwise must be same group
    return query_group == candidate_group


class SubstitutionRanker:
    """
    Ranks ingredient substitution candidates using multiple signals:
    - Semantic embedding similarity
    - PMI co-occurrence patterns
    - Category compatibility
    - Grain-specific rules
    """
    
    def __init__(self, data_yaml_path, embeddings_path, pmi_matrix_path, 
                 categories_path=None, ingredient_names_path=None):
        """Initialize ranker with required data"""
        
        # Load detection classes
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.detection_classes = data['names']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.detection_classes)}
        
        # Load embeddings
        self.embeddings = torch.load(embeddings_path)
        print(f"Loaded embeddings: {self.embeddings.shape}")
        
        # Load PMI matrix
        self.pmi_matrix = load_npz(pmi_matrix_path).toarray()
        print(f"Loaded PMI matrix: {self.pmi_matrix.shape}")
        
        # Load categories (optional)
        self.categories = {}
        if categories_path and Path(categories_path).exists():
            with open(categories_path, 'r', encoding='utf-8') as f:
                self.categories = json.load(f)
            print(f"Loaded categories for {len(self.categories)} ingredients")
        
        print(f"Ranker initialized for {len(self.detection_classes)} ingredients")
    
    def get_embedding_neighbors(self, ingredient, top_k=20):
        """Get top K neighbors by embedding similarity"""
        if ingredient not in self.class_to_idx:
            return []
        
        idx = self.class_to_idx[ingredient]
        query_embedding = self.embeddings[idx].unsqueeze(0)
        
        # Calculate cosine similarities
        similarities = F.cosine_similarity(query_embedding, self.embeddings, dim=1)
        
        # Exclude self
        similarities[idx] = -1
        
        # Get top K
        top_indices = torch.topk(similarities, min(top_k, len(self.detection_classes)-1)).indices
        
        results = []
        for i in top_indices:
            results.append((
                self.detection_classes[i.item()],
                similarities[i.item()].item()
            ))
        
        return results
    
    def get_pmi_neighbors(self, ingredient, top_k=20):
        """Get top K neighbors by PMI score"""
        if ingredient not in self.class_to_idx:
            return []
        
        idx = self.class_to_idx[ingredient]
        pmi_scores = self.pmi_matrix[idx]
        
        # Get top K (excluding self)
        pmi_scores[idx] = -np.inf
        top_indices = np.argsort(pmi_scores)[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            if pmi_scores[i] > -np.inf:
                results.append((
                    self.detection_classes[i],
                    float(pmi_scores[i])
                ))
        
        return results
    
    def reciprocal_rank_fusion(self, embedding_neighbors, pmi_neighbors, k=60):
        """
        Combine rankings using Reciprocal Rank Fusion
        RRF score = sum(1 / (k + rank_i)) across all rankings
        """
        scores = {}
        
        # Add embedding similarity scores
        for rank, (ingredient, _) in enumerate(embedding_neighbors):
            scores[ingredient] = scores.get(ingredient, 0) + 1 / (k + rank + 1)
        
        # Add PMI scores
        for rank, (ingredient, _) in enumerate(pmi_neighbors):
            scores[ingredient] = scores.get(ingredient, 0) + 1 / (k + rank + 1)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def filter_by_pmi(self, candidates, query_ingredient, pmi_low=-0.5, pmi_high=0.65):
        """
        Filter candidates by PMI thresholds
        - PMI too high (>0.65): likely used together, not substitutes
        - PMI too low (<-0.5): incompatible cuisines
        """
        if query_ingredient not in self.class_to_idx:
            return candidates
        
        query_idx = self.class_to_idx[query_ingredient]
        filtered = []
        
        for candidate, score in candidates:
            if candidate not in self.class_to_idx:
                continue
            
            candidate_idx = self.class_to_idx[candidate]
            pmi = self.pmi_matrix[query_idx, candidate_idx]
            
            # Apply filters
            if pmi < pmi_low:  # Too unrelated
                continue
            if pmi > pmi_high:  # Used together, not substitutes
                continue
            
            filtered.append((candidate, score, float(pmi)))
        
        return filtered
    
    def filter_by_category(self, candidates, query_ingredient):
        """Filter candidates to same category"""
        if not self.categories or query_ingredient not in self.categories:
            # No category info, return all
            return candidates
        
        query_category = self.categories[query_ingredient]
        filtered = []
        
        for item in candidates:
            candidate = item[0]
            candidate_category = self.categories.get(candidate)
            
            if candidate_category == query_category:
                filtered.append(item)
        
        return filtered
    
    def filter_by_grain_rules(self, candidates, query_ingredient):
        """Filter candidates by grain substitution rules"""
        # Only apply to grains
        query_category = self.categories.get(query_ingredient)
        if query_category != 'grain':
            return candidates
        
        filtered = []
        for item in candidates:
            candidate = item[0]
            candidate_category = self.categories.get(candidate)
            
            # Only filter grain-to-grain substitutions
            if candidate_category != 'grain':
                filtered.append(item)
                continue
            
            # Apply grain rules
            if is_valid_grain_substitution(query_ingredient, candidate):
                filtered.append(item)
        
        return filtered
    
    def get_substitutions(self, ingredient, top_k=10, use_category_filter=True):
        """
        Get substitution candidates for an ingredient
        Returns list of (candidate, score, pmi, confidence_level) tuples
        """
        if ingredient not in self.class_to_idx:
            return []
        
        # Get neighbors from both signals
        embedding_neighbors = self.get_embedding_neighbors(ingredient, top_k=20)
        pmi_neighbors = self.get_pmi_neighbors(ingredient, top_k=20)
        
        # Combine with RRF
        ranked = self.reciprocal_rank_fusion(embedding_neighbors, pmi_neighbors)
        
        # Filter by PMI thresholds
        filtered = self.filter_by_pmi(ranked, ingredient)
        
        # Filter by category if enabled
        if use_category_filter:
            filtered = self.filter_by_category(filtered, ingredient)
        
        # Apply grain-specific rules
        filtered = self.filter_by_grain_rules(filtered, ingredient)
        
        # Assign confidence levels
        results = []
        for candidate, score, pmi in filtered[:top_k]:
            # Confidence based on score thresholds
            if score > 0.015:
                confidence = 'high'
            elif score > 0.010:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            results.append((candidate, score, pmi, confidence))
        
        return results

def main():
    """Test the substitution ranker"""
    # Paths
    data_yaml = Path("data/merged/data.yaml")
    embeddings_path = Path("data/embeddings/semantic_embeddings.pt")
    pmi_matrix_path = Path("data/substitution/pmi_matrix.npz")
    categories_path = Path("data/substitution/ingredient_categories.json")
    
    # Check files
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return
    if not embeddings_path.exists():
        print(f"Error: {embeddings_path} not found. Run generate_embeddings.py first")
        return
    if not pmi_matrix_path.exists():
        print(f"Error: {pmi_matrix_path} not found. Run build_cooccurrence_graph.py first")
        return
    
    # Initialize ranker
    print("Initializing substitution ranker...")
    ranker = SubstitutionRanker(
        data_yaml,
        embeddings_path,
        pmi_matrix_path,
        categories_path
    )
    
    # Test on sample ingredients
    test_ingredients = [
        'Paneer',
        'Tomato',
        'Chili Pepper -Khursani-',
        'Rice -Chamal-',
        'Egg'
    ]
    
    print("\nTesting substitution suggestions:\n")
    
    for ingredient in test_ingredients:
        print(f"{ingredient}:")
        substitutions = ranker.get_substitutions(ingredient, top_k=5)
        
        if not substitutions:
            print("  No substitutions found")
        else:
            for i, (candidate, score, pmi, confidence) in enumerate(substitutions, 1):
                print(f"  {i}. {candidate}")
                print(f"     Score: {score:.4f} | PMI: {pmi:.3f} | Confidence: {confidence}")
        
        print()
    
    print("Ranker test complete")
    print("\nUsage in other scripts:")
    print("  from substitution_ranker import SubstitutionRanker")
    print("  ranker = SubstitutionRanker(...)")
    print("  substitutions = ranker.get_substitutions('Paneer', top_k=10)")

if __name__ == "__main__":
    main()