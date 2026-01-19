"""
Smart Substitution Engine v2 for Bawarchi
Now with:
- Dish pattern matching (cuisine → dish templates)
- Role-based substitution (roti fulfills "flatbread" role for tacos AND Indian bread)
- Cuisine-aware recipe suggestions
- Clickable fusion suggestions

Key insight: Same ingredients can make different dishes based on cuisine context
- [chicken, tomato, onion, roti] + Mexican → Chicken Tacos (roti = tortilla)
- [chicken, tomato, onion, roti] + Indian → Chicken Curry with Roti
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from scipy import sparse


@dataclass
class SubstitutionTip:
    original: str
    substitute: str
    category: str
    score: float
    reason: str
    tip: str


@dataclass
class DishSuggestion:
    """A suggested dish based on available ingredients"""
    dish_name: str
    cuisine: str
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    substitutions_used: Dict[str, str]  # role → ingredient used
    match_score: float
    description: str
    
    def to_prompt_context(self) -> str:
        """Convert to context for recipe generation"""
        subs_text = ""
        if self.substitutions_used:
            subs_text = " Using: " + ", ".join([f"{v} as {k}" for k, v in self.substitutions_used.items()])
        return f"{self.dish_name} ({self.cuisine}){subs_text}"


# ============================================================================
# DISH PATTERN DATABASE
# ============================================================================

DISH_PATTERNS = {
    "mexican": {
        "tacos": {
            "required_roles": ["protein", "flatbread", "vegetable"],
            "optional_roles": ["cheese", "sauce", "herb"],
            "role_mappings": {
                "protein": ["chicken", "beef", "pork", "fish", "paneer", "tofu", "egg"],
                "flatbread": ["tortilla", "roti", "chapati", "naan", "paratha", "wrap"],
                "vegetable": ["tomato", "onion", "capsicum", "lettuce", "corn", "beans"],
                "cheese": ["cheese", "paneer", "mozzarella", "cheddar"],
                "sauce": ["salsa", "sour cream", "guacamole", "yogurt", "raita"],
                "herb": ["cilantro", "coriander", "mint", "lime"],
            },
            "description": "Seasoned filling in flatbread wrap"
        },
        "burrito_bowl": {
            "required_roles": ["protein", "grain", "vegetable"],
            "optional_roles": ["beans", "cheese", "sauce"],
            "role_mappings": {
                "protein": ["chicken", "beef", "pork", "tofu", "paneer"],
                "grain": ["rice", "quinoa"],
                "vegetable": ["tomato", "onion", "capsicum", "corn", "lettuce"],
                "beans": ["beans", "black beans", "kidney beans", "chickpeas", "rajma"],
                "cheese": ["cheese", "paneer"],
                "sauce": ["salsa", "guacamole", "sour cream", "yogurt"],
            },
            "description": "Deconstructed burrito served over rice"
        },
        "quesadilla": {
            "required_roles": ["flatbread", "cheese"],
            "optional_roles": ["protein", "vegetable"],
            "role_mappings": {
                "flatbread": ["tortilla", "roti", "chapati", "paratha"],
                "cheese": ["cheese", "paneer", "mozzarella"],
                "protein": ["chicken", "beef", "paneer"],
                "vegetable": ["onion", "capsicum", "tomato"],
            },
            "description": "Grilled flatbread with melted cheese filling"
        },
    },
    "indian": {
        "curry": {
            "required_roles": ["protein", "vegetable", "spice"],
            "optional_roles": ["dairy", "herb"],
            "role_mappings": {
                "protein": ["chicken", "paneer", "fish", "egg", "mutton", "tofu", "dal", "chickpeas"],
                "vegetable": ["tomato", "onion", "capsicum", "potato", "cauliflower", "spinach"],
                "spice": ["cumin", "turmeric", "garam masala", "coriander", "chili", "ginger", "garlic"],
                "dairy": ["yogurt", "cream", "milk", "ghee", "butter"],
                "herb": ["cilantro", "coriander", "mint", "curry leaves"],
            },
            "description": "Spiced gravy dish"
        },
        "biryani": {
            "required_roles": ["protein", "grain", "spice"],
            "optional_roles": ["vegetable", "herb", "dairy"],
            "role_mappings": {
                "protein": ["chicken", "mutton", "paneer", "egg", "fish"],
                "grain": ["rice", "basmati"],
                "spice": ["garam masala", "saffron", "cardamom", "cinnamon", "bay leaf"],
                "vegetable": ["onion", "tomato", "potato"],
                "herb": ["mint", "cilantro", "coriander"],
                "dairy": ["yogurt", "ghee"],
            },
            "description": "Layered spiced rice with protein"
        },
        "tikka": {
            "required_roles": ["protein", "spice", "dairy"],
            "optional_roles": ["vegetable"],
            "role_mappings": {
                "protein": ["chicken", "paneer", "fish", "mutton"],
                "spice": ["garam masala", "turmeric", "chili", "cumin", "paprika"],
                "dairy": ["yogurt", "cream"],
                "vegetable": ["onion", "capsicum", "tomato"],
            },
            "description": "Marinated and grilled protein pieces"
        },
        "sabzi": {
            "required_roles": ["vegetable", "spice"],
            "optional_roles": ["protein", "herb"],
            "role_mappings": {
                "vegetable": ["potato", "cauliflower", "spinach", "capsicum", "beans", "okra", "cabbage", "carrot"],
                "spice": ["cumin", "turmeric", "coriander", "chili", "garam masala"],
                "protein": ["paneer", "tofu", "chickpeas"],
                "herb": ["cilantro", "coriander"],
            },
            "description": "Dry or semi-dry vegetable preparation"
        },
        "dal": {
            "required_roles": ["legume", "spice"],
            "optional_roles": ["vegetable", "herb", "dairy"],
            "role_mappings": {
                "legume": ["dal", "lentils", "chickpeas", "beans", "moong", "toor", "masoor"],
                "spice": ["cumin", "turmeric", "ginger", "garlic", "chili"],
                "vegetable": ["tomato", "onion", "spinach"],
                "herb": ["cilantro", "coriander"],
                "dairy": ["ghee", "butter"],
            },
            "description": "Spiced lentil preparation"
        },
    },
    "italian": {
        "pasta": {
            "required_roles": ["pasta", "sauce_base"],
            "optional_roles": ["protein", "vegetable", "cheese", "herb"],
            "role_mappings": {
                "pasta": ["pasta", "spaghetti", "noodles", "penne", "macaroni"],
                "sauce_base": ["tomato", "cream", "olive oil", "butter"],
                "protein": ["chicken", "bacon", "sausage", "paneer", "tofu"],
                "vegetable": ["onion", "garlic", "capsicum", "mushroom", "spinach"],
                "cheese": ["parmesan", "mozzarella", "cheese", "paneer"],
                "herb": ["basil", "oregano", "parsley"],
            },
            "description": "Pasta with sauce and toppings"
        },
        "risotto": {
            "required_roles": ["grain", "dairy", "liquid"],
            "optional_roles": ["protein", "vegetable", "cheese"],
            "role_mappings": {
                "grain": ["rice", "arborio"],
                "dairy": ["butter", "cream", "ghee"],
                "liquid": ["stock", "broth", "wine", "water"],
                "protein": ["chicken", "shrimp", "mushroom"],
                "vegetable": ["onion", "mushroom", "spinach", "peas"],
                "cheese": ["parmesan", "cheese"],
            },
            "description": "Creamy Italian rice dish"
        },
    },
    "asian": {
        "stir_fry": {
            "required_roles": ["protein", "vegetable", "sauce"],
            "optional_roles": ["grain", "spice"],
            "role_mappings": {
                "protein": ["chicken", "tofu", "paneer", "egg", "shrimp", "beef", "pork"],
                "vegetable": ["onion", "capsicum", "carrot", "cabbage", "broccoli", "beans"],
                "sauce": ["soy sauce", "oyster sauce", "teriyaki", "chili sauce"],
                "grain": ["rice", "noodles"],
                "spice": ["ginger", "garlic", "chili"],
            },
            "description": "Quick-cooked ingredients in wok"
        },
        "fried_rice": {
            "required_roles": ["grain", "vegetable", "sauce"],
            "optional_roles": ["protein", "egg"],
            "role_mappings": {
                "grain": ["rice"],
                "vegetable": ["onion", "carrot", "peas", "capsicum", "cabbage", "corn"],
                "sauce": ["soy sauce", "oyster sauce"],
                "protein": ["chicken", "shrimp", "tofu", "paneer"],
                "egg": ["egg"],
            },
            "description": "Wok-fried rice with vegetables"
        },
    },
    "fusion": {
        "indo_mexican": {
            "required_roles": ["protein", "flatbread", "spice"],
            "optional_roles": ["vegetable", "dairy"],
            "role_mappings": {
                "protein": ["chicken", "paneer", "tofu", "chickpeas"],
                "flatbread": ["tortilla", "roti", "naan", "paratha"],
                "spice": ["cumin", "garam masala", "chili", "coriander"],
                "vegetable": ["tomato", "onion", "capsicum"],
                "dairy": ["yogurt", "sour cream", "cheese", "paneer"],
            },
            "description": "Indian spices meet Mexican format"
        },
        "indo_italian": {
            "required_roles": ["pasta", "spice", "sauce_base"],
            "optional_roles": ["protein", "vegetable", "cheese"],
            "role_mappings": {
                "pasta": ["pasta", "noodles", "spaghetti"],
                "spice": ["garam masala", "cumin", "turmeric", "chili"],
                "sauce_base": ["tomato", "cream"],
                "protein": ["chicken", "paneer", "tofu"],
                "vegetable": ["onion", "capsicum", "spinach"],
                "cheese": ["cheese", "paneer", "parmesan"],
            },
            "description": "Italian format with Indian flavors"
        },
    }
}

# Ingredient role classification (what role can each ingredient play)
INGREDIENT_ROLES = {
    # Proteins
    "chicken": ["protein"],
    "paneer": ["protein", "cheese"],
    "tofu": ["protein"],
    "egg": ["protein", "egg"],
    "fish": ["protein"],
    "mutton": ["protein"],
    "beef": ["protein"],
    "pork": ["protein"],
    "bacon": ["protein"],
    "sausage": ["protein"],
    "chickpeas": ["protein", "legume"],
    "dal": ["protein", "legume"],
    "lentils": ["protein", "legume"],
    
    # Flatbreads (CRITICAL for cross-cuisine substitution)
    "roti": ["flatbread", "bread"],
    "chapati": ["flatbread", "bread"],
    "naan": ["flatbread", "bread"],
    "paratha": ["flatbread", "bread"],
    "tortilla": ["flatbread", "bread"],
    "bread": ["bread"],
    
    # Grains
    "rice": ["grain"],
    "pasta": ["pasta"],
    "noodles": ["pasta"],
    "quinoa": ["grain"],
    
    # Vegetables
    "tomato": ["vegetable", "sauce_base"],
    "onion": ["vegetable"],
    "garlic": ["vegetable", "spice"],
    "ginger": ["vegetable", "spice"],
    "capsicum": ["vegetable"],
    "potato": ["vegetable"],
    "carrot": ["vegetable"],
    "spinach": ["vegetable"],
    "cauliflower": ["vegetable"],
    "cabbage": ["vegetable"],
    "corn": ["vegetable"],
    "peas": ["vegetable"],
    "beans": ["vegetable", "legume"],
    "mushroom": ["vegetable"],
    "broccoli": ["vegetable"],
    "lettuce": ["vegetable"],
    
    # Dairy
    "milk": ["dairy", "liquid"],
    "cream": ["dairy", "sauce_base"],
    "yogurt": ["dairy", "sauce"],
    "butter": ["dairy"],
    "ghee": ["dairy"],
    "cheese": ["cheese", "dairy"],
    
    # Spices
    "cumin": ["spice"],
    "turmeric": ["spice"],
    "garam masala": ["spice"],
    "coriander": ["spice", "herb"],
    "cilantro": ["herb"],
    "chili": ["spice"],
    "paprika": ["spice"],
    "oregano": ["herb"],
    "basil": ["herb"],
    "mint": ["herb"],
    
    # Sauces
    "soy sauce": ["sauce"],
    "salsa": ["sauce"],
    "ketchup": ["sauce"],
}


class SmartSubstitutionEngine:
    """
    Autonomous substitution + dish pattern matching system.
    """
    
    def __init__(
        self,
        categories_path: str = "data/substitution/ingredient_categories.json",
        embeddings_path: str = "data/embeddings/semantic_embeddings.pt",
        pmi_path: str = "data/substitution/pmi_matrix.npz",
        class_names_path: str = "data/merged/data.yaml"
    ):
        self.categories_path = Path(categories_path)
        self.embeddings_path = Path(embeddings_path)
        self.pmi_path = Path(pmi_path)
        self.class_names_path = Path(class_names_path)
        
        # Load existing data
        self.categories = self._load_categories()
        self.embeddings = self._load_embeddings()
        self.pmi_matrix = self._load_pmi()
        self.class_names = self._load_class_names()
        
        # Build lookups
        self.ingredient_to_category = {}
        self.category_to_ingredients = {}
        self._build_category_index()
        
        # Precompute similarity
        self.similarity_matrix = self._compute_similarity_matrix()
        
        # Dish patterns
        self.dish_patterns = DISH_PATTERNS
        self.ingredient_roles = INGREDIENT_ROLES
    
    def _load_categories(self) -> Dict:
        if not self.categories_path.exists():
            return {}
        with open(self.categories_path, 'r') as f:
            return json.load(f)
    
    def _load_embeddings(self) -> Optional[torch.Tensor]:
        if not self.embeddings_path.exists():
            return None
        return torch.load(self.embeddings_path, map_location='cpu')
    
    def _load_pmi(self) -> Optional[np.ndarray]:
        if not self.pmi_path.exists():
            return None
        try:
            loaded = np.load(self.pmi_path, allow_pickle=True)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if 'data' in loaded.files:
                    return sparse.csr_matrix(
                        (loaded['data'], loaded['indices'], loaded['indptr']),
                        shape=loaded['shape']
                    ).toarray()
                elif 'matrix' in loaded.files:
                    return loaded['matrix']
            return loaded
        except:
            return None
    
    def _load_class_names(self) -> List[str]:
        if not self.class_names_path.exists():
            if self.categories:
                all_names = []
                for cat, items in self.categories.items():
                    if isinstance(items, list):
                        all_names.extend(items)
                return all_names
            return []
        try:
            import yaml
            with open(self.class_names_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        except:
            return []
    
    def _build_category_index(self):
        for category, ingredients in self.categories.items():
            if not isinstance(ingredients, list):
                continue
            self.category_to_ingredients[category] = []
            for ing in ingredients:
                ing_lower = ing.lower().strip()
                ing_normalized = ing_lower.replace(' ', '_')
                self.ingredient_to_category[ing_lower] = category
                self.ingredient_to_category[ing_normalized] = category
                self.ingredient_to_category[ing] = category
                self.category_to_ingredients[category].append(ing)
    
    def _compute_similarity_matrix(self) -> Optional[np.ndarray]:
        if self.embeddings is None:
            return None
        emb = self.embeddings.numpy() if isinstance(self.embeddings, torch.Tensor) else self.embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_normalized = emb / norms
        return np.dot(emb_normalized, emb_normalized.T)
    
    def _get_ingredient_index(self, ingredient: str) -> Optional[int]:
        ing_lower = ingredient.lower().strip()
        ing_normalized = ing_lower.replace(' ', '_')
        for i, name in enumerate(self.class_names):
            name_lower = name.lower()
            name_normalized = name_lower.replace(' ', '_')
            if name_lower == ing_lower or name_normalized == ing_normalized:
                return i
            if ing_lower in name_lower or name_lower in ing_lower:
                return i
        return None
    
    def get_category(self, ingredient: str) -> Optional[str]:
        ing_lower = ingredient.lower().strip()
        ing_normalized = ing_lower.replace(' ', '_')
        if ing_lower in self.ingredient_to_category:
            return self.ingredient_to_category[ing_lower]
        if ing_normalized in self.ingredient_to_category:
            return self.ingredient_to_category[ing_normalized]
        for stored_ing, category in self.ingredient_to_category.items():
            if ing_lower in stored_ing.lower() or stored_ing.lower() in ing_lower:
                return category
        return None
    
    def get_ingredient_roles(self, ingredient: str) -> List[str]:
        """Get all roles an ingredient can fulfill"""
        ing_lower = ingredient.lower().strip().replace('_', ' ')
        
        # Direct lookup
        if ing_lower in self.ingredient_roles:
            return self.ingredient_roles[ing_lower]
        
        # Partial match
        for key, roles in self.ingredient_roles.items():
            if key in ing_lower or ing_lower in key:
                return roles
        
        # Fallback: use category as role
        cat = self.get_category(ingredient)
        if cat:
            return [cat.lower()]
        
        return ["ingredient"]
    
    def get_same_category_substitutes(self, ingredient: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Get substitution candidates from same category"""
        category = self.get_category(ingredient)
        if not category:
            return []
        
        candidates = self.category_to_ingredients.get(category, [])
        if not candidates:
            return []
        
        query_idx = self._get_ingredient_index(ingredient)
        
        results = []
        for candidate in candidates:
            if candidate.lower() == ingredient.lower():
                continue
            
            cand_idx = self._get_ingredient_index(candidate)
            score = 0.5
            
            if self.similarity_matrix is not None and query_idx is not None and cand_idx is not None:
                try:
                    semantic_sim = self.similarity_matrix[query_idx, cand_idx]
                    score = 0.3 + (0.7 * semantic_sim)
                except:
                    pass
            
            if self.pmi_matrix is not None and query_idx is not None and cand_idx is not None:
                try:
                    pmi_score = self.pmi_matrix[query_idx, cand_idx]
                    pmi_boost = max(0, min(0.2, pmi_score * 0.1))
                    score += pmi_boost
                except:
                    pass
            
            results.append((candidate, min(score, 1.0), f"Same category ({category})"))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    # =========================================================================
    # DISH PATTERN MATCHING
    # =========================================================================
    
    def match_dish_pattern(
        self, 
        ingredients: List[str], 
        dish_name: str, 
        pattern: Dict,
        cuisine: str
    ) -> Optional[DishSuggestion]:
        """
        Check if ingredients can make a specific dish.
        Returns DishSuggestion if possible, None otherwise.
        """
        # Build role → available ingredients mapping
        available_by_role = {}
        for ing in ingredients:
            roles = self.get_ingredient_roles(ing)
            for role in roles:
                if role not in available_by_role:
                    available_by_role[role] = []
                available_by_role[role].append(ing)
        
        # Check required roles
        required = pattern.get("required_roles", [])
        role_mappings = pattern.get("role_mappings", {})
        
        matched = []
        missing = []
        substitutions = {}
        
        for role in required:
            # Check if we have an ingredient that can fulfill this role
            found = False
            
            # Direct role match
            if role in available_by_role:
                matched.append(available_by_role[role][0])
                substitutions[role] = available_by_role[role][0]
                found = True
            else:
                # Check if any available ingredient is in role_mappings for this role
                valid_for_role = role_mappings.get(role, [])
                for ing in ingredients:
                    ing_lower = ing.lower().replace('_', ' ')
                    for valid in valid_for_role:
                        if valid in ing_lower or ing_lower in valid:
                            matched.append(ing)
                            substitutions[role] = ing
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                missing.append(role)
        
        # If missing required ingredients, dish can't be made
        if missing:
            return None
        
        # Calculate match score
        optional = pattern.get("optional_roles", [])
        optional_matched = 0
        for role in optional:
            if role in available_by_role:
                optional_matched += 1
            else:
                valid_for_role = role_mappings.get(role, [])
                for ing in ingredients:
                    ing_lower = ing.lower().replace('_', ' ')
                    for valid in valid_for_role:
                        if valid in ing_lower or ing_lower in valid:
                            optional_matched += 1
                            break
                    break
        
        # Score: 100% for required, bonus for optional
        base_score = 1.0
        if optional:
            optional_bonus = (optional_matched / len(optional)) * 0.2
            base_score = 0.8 + optional_bonus
        
        return DishSuggestion(
            dish_name=dish_name.replace('_', ' ').title(),
            cuisine=cuisine.title(),
            matched_ingredients=matched,
            missing_ingredients=missing,
            substitutions_used=substitutions,
            match_score=base_score,
            description=pattern.get("description", "")
        )
    
    def get_dish_suggestions(
        self, 
        ingredients: List[str], 
        cuisine: Optional[str] = None,
        top_k: int = 5
    ) -> List[DishSuggestion]:
        """
        Get all dishes that can be made with given ingredients.
        Optionally filter by cuisine.
        """
        suggestions = []
        
        cuisines_to_check = [cuisine.lower()] if cuisine else list(self.dish_patterns.keys())
        
        for cuisine_name in cuisines_to_check:
            if cuisine_name not in self.dish_patterns:
                continue
            
            dishes = self.dish_patterns[cuisine_name]
            for dish_name, pattern in dishes.items():
                suggestion = self.match_dish_pattern(ingredients, dish_name, pattern, cuisine_name)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Sort by score
        suggestions.sort(key=lambda x: x.match_score, reverse=True)
        return suggestions[:top_k]
    
    def get_cuisine_specific_suggestion(
        self, 
        ingredients: List[str], 
        cuisine: str
    ) -> Optional[DishSuggestion]:
        """
        Get the BEST dish for a specific cuisine.
        This is what gets triggered when user selects a cuisine.
        """
        suggestions = self.get_dish_suggestions(ingredients, cuisine, top_k=1)
        return suggestions[0] if suggestions else None
    
    def get_cross_cuisine_suggestions(
        self, 
        ingredients: List[str]
    ) -> List[Dict]:
        """
        Show what dishes can be made across different cuisines.
        E.g., [chicken, tomato, onion, roti]:
        - Mexican: Tacos
        - Indian: Curry
        - Fusion: Indo-Mexican Tacos
        """
        results = []
        
        for cuisine in self.dish_patterns.keys():
            best = self.get_cuisine_specific_suggestion(ingredients, cuisine)
            if best:
                results.append({
                    "cuisine": cuisine.title(),
                    "dish": best.dish_name,
                    "score": best.match_score,
                    "substitutions": best.substitutions_used,
                    "description": best.description,
                    "suggestion_object": best
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    # =========================================================================
    # SMART TIPS (Enhanced)
    # =========================================================================
    
    def get_smart_tips(self, ingredients: List[str]) -> List[SubstitutionTip]:
        """Generate smart substitution tips"""
        tips = []
        
        for ing in ingredients:
            cat = self.get_category(ing)
            if not cat:
                continue
            
            # Only suggest for key categories
            key_categories = ['protein', 'meat', 'dairy', 'vegetable', 'legume']
            if not any(k in cat.lower() for k in key_categories):
                continue
            
            subs = self.get_same_category_substitutes(ing, top_k=2)
            for sub_name, score, reason in subs:
                ing_clean = ing.replace('_', ' ').title()
                sub_clean = sub_name.replace('_', ' ').title()
                
                tips.append(SubstitutionTip(
                    original=ing_clean,
                    substitute=sub_clean,
                    category=cat,
                    score=score,
                    reason=reason,
                    tip=f"💡 {sub_clean} can replace {ing_clean} ({cat})"
                ))
        
        tips.sort(key=lambda x: x.score, reverse=True)
        
        # Deduplicate
        seen = set()
        unique = []
        for tip in tips:
            key = (tip.original.lower(), tip.substitute.lower())
            if key not in seen:
                seen.add(key)
                unique.append(tip)
        
        return unique[:5]
    
    def detect_cuisine_context(self, ingredients: List[str]) -> Dict:
        """Detect cuisine from ingredients"""
        cuisine_markers = {
            'indian': ['paneer', 'garam masala', 'turmeric', 'cumin', 'coriander', 
                      'dal', 'ghee', 'curry', 'roti', 'naan', 'chapati'],
            'italian': ['pasta', 'parmesan', 'basil', 'oregano', 'mozzarella',
                       'olive oil'],
            'mexican': ['tortilla', 'jalapeno', 'cilantro', 'lime', 'avocado',
                       'beans', 'salsa'],
            'asian': ['soy sauce', 'ginger', 'sesame', 'tofu', 'noodles'],
        }
        
        detected = {}
        for cuisine, markers in cuisine_markers.items():
            matches = []
            for ing in ingredients:
                ing_lower = ing.lower().replace('_', ' ')
                for marker in markers:
                    if marker in ing_lower or ing_lower in marker:
                        matches.append(ing)
                        break
            if matches:
                detected[cuisine] = matches
        
        return {
            'detected_cuisines': detected,
            'fusion_potential': len(detected) > 1,
            'primary_cuisine': max(detected.keys(), key=lambda k: len(detected[k])) if detected else None
        }
    
    def get_fusion_suggestions(self, ingredients: List[str]) -> List[str]:
        """Generate fusion suggestions"""
        context = self.detect_cuisine_context(ingredients)
        
        if not context['fusion_potential']:
            return []
        
        suggestions = []
        cuisines = list(context['detected_cuisines'].keys())
        
        if 'indian' in cuisines and 'mexican' in cuisines:
            suggestions.append("🌮 Indo-Mexican: Paneer Tacos, Tikka Quesadillas, Masala Nachos")
        if 'indian' in cuisines and 'italian' in cuisines:
            suggestions.append("🍝 Indo-Italian: Masala Pasta, Curry Risotto, Tikka Pizza")
        if 'asian' in cuisines and 'indian' in cuisines:
            suggestions.append("🍜 Indo-Asian: Hakka Noodles, Chili Paneer, Manchurian")
        
        return suggestions
    
    def get_clickable_suggestions(self, ingredients: List[str]) -> List[Dict]:
        """
        Get suggestions that can be clicked to generate specific recipes.
        Returns list of dicts with all info needed for one-click generation.
        """
        suggestions = []
        
        # Get cross-cuisine suggestions
        cross = self.get_cross_cuisine_suggestions(ingredients)
        
        for item in cross:
            subs_text = ""
            if item["substitutions"]:
                subs_parts = [f"{v} as {k}" for k, v in item["substitutions"].items()]
                subs_text = f" (using {', '.join(subs_parts)})"
            
            suggestions.append({
                "display": f"🍽️ {item['cuisine']}: {item['dish']}{subs_text}",
                "cuisine": item["cuisine"].lower(),
                "dish": item["dish"],
                "score": item["score"],
                "prompt_hint": f"Make {item['dish']}, a {item['cuisine']} dish. {item['description']}",
                "substitutions": item["substitutions"]
            })
        
        return suggestions


# Factory function
def create_smart_engine():
    return SmartSubstitutionEngine(
        categories_path="data/substitution/ingredient_categories.json",
        embeddings_path="data/embeddings/semantic_embeddings.pt",
        pmi_path="data/substitution/pmi_matrix.npz",
        class_names_path="data/merged/data.yaml"
    )


# Test
if __name__ == "__main__":
    print("Testing Smart Substitution Engine v2...")
    
    engine = create_smart_engine()
    
    # Test: Same ingredients, different cuisines
    test_ingredients = ['chicken', 'tomato', 'onion', 'roti']
    
    print(f"\nIngredients: {test_ingredients}")
    print("\n" + "="*50)
    print("CROSS-CUISINE SUGGESTIONS:")
    print("="*50)
    
    suggestions = engine.get_cross_cuisine_suggestions(test_ingredients)
    for s in suggestions:
        print(f"\n{s['cuisine']}: {s['dish']}")
        print(f"  Score: {s['score']:.2f}")
        print(f"  Substitutions: {s['substitutions']}")
        print(f"  Description: {s['description']}")
    
    print("\n" + "="*50)
    print("CLICKABLE SUGGESTIONS:")
    print("="*50)
    
    clickable = engine.get_clickable_suggestions(test_ingredients)
    for c in clickable:
        print(f"\n{c['display']}")
        print(f"  → Click to generate: {c['prompt_hint']}")
    
    # Test fusion
    print("\n" + "="*50)
    print("FUSION TEST: [paneer, tomato, tortilla, cumin]")
    print("="*50)
    
    fusion_ingredients = ['paneer', 'tomato', 'tortilla', 'cumin']
    fusion_suggestions = engine.get_fusion_suggestions(fusion_ingredients)
    for f in fusion_suggestions:
        print(f"  {f}")