"""
Role-Based Cuisine-Aware Training Data Generator
=================================================

This generates the MISSING training data that teaches the model:
1. Same ingredients → Different dishes based on cuisine context
2. Role-based substitution (roti=flatbread=can be tortilla)
3. Cuisine-specific dish patterns

Example output:
- Input: "I have chicken, tomato, onion, roti. Make Mexican."
- Output: "Chicken Tacos (using roti as tortilla)..."

- Input: "I have chicken, tomato, onion, roti. Make Indian."
- Output: "Chicken Curry with Roti..."

Generate 5-10K examples, fine-tune for 1-2 hours.
"""

import json
import random
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple
from tqdm import tqdm

# ============================================================================
# INGREDIENT ROLE DATABASE
# ============================================================================

INGREDIENT_ROLES = {
    # Proteins
    "chicken": ["protein"],
    "paneer": ["protein", "dairy"],
    "tofu": ["protein"],
    "egg": ["protein"],
    "fish": ["protein"],
    "mutton": ["protein"],
    "beef": ["protein"],
    "pork": ["protein"],
    "chickpeas": ["protein", "legume"],
    "dal": ["protein", "legume"],
    "lentils": ["legume"],
    "beans": ["legume"],
    
    # Flatbreads (CRITICAL - these are interchangeable across cuisines)
    "roti": ["flatbread", "bread", "wrap"],
    "chapati": ["flatbread", "bread", "wrap"],
    "naan": ["flatbread", "bread"],
    "paratha": ["flatbread", "bread", "wrap"],
    "tortilla": ["flatbread", "wrap"],
    "pita": ["flatbread", "bread"],
    "bread": ["bread"],
    
    # Grains
    "rice": ["grain", "base"],
    "pasta": ["grain", "base"],
    "noodles": ["grain", "base"],
    "quinoa": ["grain", "base"],
    
    # Vegetables
    "tomato": ["vegetable", "sauce_base"],
    "onion": ["vegetable", "aromatic"],
    "garlic": ["aromatic", "spice"],
    "ginger": ["aromatic", "spice"],
    "capsicum": ["vegetable"],
    "potato": ["vegetable", "starch"],
    "carrot": ["vegetable"],
    "spinach": ["vegetable", "leafy"],
    "cabbage": ["vegetable"],
    "cauliflower": ["vegetable"],
    "corn": ["vegetable", "grain"],
    "peas": ["vegetable"],
    "mushroom": ["vegetable"],
    "lettuce": ["vegetable", "leafy"],
    
    # Dairy
    "milk": ["dairy", "liquid"],
    "cream": ["dairy", "sauce_base"],
    "yogurt": ["dairy", "sauce"],
    "butter": ["dairy", "fat"],
    "ghee": ["dairy", "fat"],
    "cheese": ["dairy", "topping"],
    
    # Spices (cuisine indicators)
    "cumin": ["spice", "indian"],
    "turmeric": ["spice", "indian"],
    "garam masala": ["spice", "indian"],
    "coriander": ["spice", "herb", "indian"],
    "cilantro": ["herb", "mexican", "indian"],
    "chili": ["spice"],
    "paprika": ["spice", "mexican"],
    "oregano": ["herb", "italian", "mexican"],
    "basil": ["herb", "italian"],
    "soy sauce": ["sauce", "asian"],
    "lime": ["citrus", "mexican", "asian"],
}

# ============================================================================
# DISH PATTERNS BY CUISINE
# ============================================================================

DISH_PATTERNS = {
    "mexican": {
        "tacos": {
            "required": {"protein", "flatbread", "vegetable"},
            "description": "Seasoned {protein} in {flatbread} with fresh {vegetable}",
            "template": """# {Protein} Tacos

**Using {flatbread} as the wrap** (works great as a tortilla substitute!)

## Ingredients
- 300g {protein}, seasoned with cumin and paprika
- 4 {flatbread}s, warmed
- {vegetables}, diced
- Fresh cilantro and lime

## Instructions
1. Season {protein} with cumin, paprika, salt, and chili
2. Cook {protein} until done (adjust time based on protein type)
3. Warm {flatbread}s on a dry pan
4. Assemble: {flatbread} + {protein} + {vegetables} + cilantro
5. Squeeze lime on top and serve

## Substitution Note
{flatbread} works perfectly here as a tortilla alternative - both are flatbreads!
""",
        },
        "burrito_bowl": {
            "required": {"protein", "grain", "vegetable"},
            "description": "Deconstructed burrito over {grain}",
            "template": """# {Protein} Burrito Bowl

## Ingredients
- 300g {protein}, seasoned
- 1 cup {grain}, cooked
- {vegetables}, sautéed
- Yogurt or sour cream (optional)

## Instructions
1. Cook {grain} according to package
2. Season and cook {protein}
3. Sauté {vegetables} with cumin
4. Layer: {grain} → {protein} → {vegetables}
5. Top with yogurt and cilantro

## Tip
This is a great way to use leftover {grain}!
""",
        },
        "quesadilla": {
            "required": {"flatbread", "dairy"},
            "optional": {"protein", "vegetable"},
            "description": "Grilled {flatbread} with melted cheese",
            "template": """# Cheese Quesadilla{with_protein}

**Using {flatbread}** - any flatbread works great for this!

## Ingredients
- 2 {flatbread}s
- 100g {cheese}, grated
{protein_line}
{vegetable_line}

## Instructions
1. Heat a pan over medium heat
2. Place one {flatbread}, add {cheese}{protein_add}{vegetable_add}
3. Top with second {flatbread}
4. Cook until golden (2-3 min per side)
5. Cut into triangles and serve

## Why This Works
{flatbread} crisps up beautifully just like a tortilla would!
""",
        },
    },
    
    "indian": {
        "curry": {
            "required": {"protein", "vegetable", "aromatic"},
            "description": "Spiced {protein} curry",
            "template": """# {Protein} Curry

A classic Indian curry with rich, aromatic spices.

## Ingredients
- 400g {protein}
- {vegetables}
- {aromatics} (minced)
- 1 tsp each: cumin, turmeric, garam masala
- 1 cup tomato puree or fresh tomatoes
- Fresh coriander

## Instructions
1. Sauté {aromatics} until fragrant
2. Add spices, cook 30 seconds
3. Add {protein}, brown lightly
4. Add tomatoes, simmer until {protein} is cooked
5. Garnish with coriander

## Serving Suggestion
Best served with rice or {flatbread}!
""",
        },
        "sabzi": {
            "required": {"vegetable", "aromatic"},
            "description": "Dry vegetable preparation",
            "template": """# {Vegetable} Sabzi

Simple, flavorful dry vegetable dish.

## Ingredients
- 300g {vegetables}, chopped
- {aromatics}
- 1 tsp cumin seeds
- 1/2 tsp turmeric
- Salt and chili to taste

## Instructions
1. Heat oil, add cumin seeds
2. Add {aromatics}, sauté
3. Add {vegetables} and spices
4. Cover and cook until tender
5. Serve with roti or rice
""",
        },
        "biryani": {
            "required": {"protein", "grain"},
            "description": "Layered spiced rice with {protein}",
            "template": """# {Protein} Biryani

Aromatic layered rice dish.

## Ingredients
- 300g {protein}
- 2 cups basmati {grain}
- Whole spices (bay leaf, cardamom, cinnamon)
- Yogurt for marination
- Fried onions, mint, saffron

## Instructions
1. Marinate {protein} in yogurt and spices (1 hour)
2. Parboil {grain} with whole spices
3. Layer: {protein} → {grain} → fried onions → mint
4. Seal and cook on low heat (dum) for 25 min
5. Gently mix and serve

## Note
{grain} should be 70% cooked before layering.
""",
        },
    },
    
    "italian": {
        "pasta": {
            "required": {"grain", "sauce_base"},
            "optional": {"protein", "vegetable"},
            "description": "{grain} with sauce",
            "template": """# {Protein} Pasta{style}

## Ingredients
- 300g {grain}
- {sauce_base} for sauce
{protein_line}
- {vegetables}
- Garlic, olive oil
- Parmesan (optional)

## Instructions
1. Cook {grain} al dente
2. Make sauce with {sauce_base} and garlic
3. {protein_step}
4. Toss {grain} with sauce
5. Top with parmesan

## Fusion Note
Try adding Indian spices like garam masala for an Indo-Italian twist!
""",
        },
        "risotto": {
            "required": {"grain", "dairy"},
            "description": "Creamy Italian {grain}",
            "template": """# Creamy Risotto

## Ingredients
- 1.5 cups arborio rice (or regular {grain})
- 4 cups stock, warm
- {dairy} (butter/cream)
- Parmesan
- {vegetables}

## Instructions
1. Toast {grain} in butter
2. Add stock gradually, stirring
3. Cook until creamy (18-20 min)
4. Finish with {dairy} and parmesan
5. Add sautéed {vegetables}
""",
        },
    },
    
    "asian": {
        "stir_fry": {
            "required": {"protein", "vegetable"},
            "description": "Quick wok-fried {protein}",
            "template": """# {Protein} Stir Fry

Fast, healthy, delicious.

## Ingredients
- 300g {protein}, sliced thin
- {vegetables}, julienned
- Soy sauce, ginger, garlic
- Sesame oil

## Instructions
1. Heat wok until smoking
2. Flash-fry {protein} (2 min), remove
3. Stir-fry {vegetables} (2 min)
4. Return {protein}, add sauce
5. Serve over rice or noodles
""",
        },
        "fried_rice": {
            "required": {"grain", "vegetable"},
            "optional": {"protein", "egg"},
            "description": "Wok-fried {grain}",
            "template": """# {Protein} Fried Rice

Best with day-old rice!

## Ingredients
- 3 cups cooked {grain} (cold)
- {vegetables}, diced small
{protein_line}
- Soy sauce
- Eggs (optional)

## Instructions
1. Heat wok, scramble eggs, set aside
2. Stir-fry {vegetables}
3. Add cold {grain}, break up clumps
4. Add soy sauce, toss well
5. Return eggs, mix through
""",
        },
    },
    
    "fusion": {
        "indo_mexican": {
            "required": {"protein", "flatbread"},
            "description": "Indian-Mexican fusion",
            "template": """# Tikka {Protein} Tacos

The best of both worlds - Indian spices meet Mexican format!

## Ingredients
- 300g {protein}, cubed
- Tikka marinade (yogurt, garam masala, chili, ginger-garlic)
- 4 {flatbread}s
- {vegetables}
- Mint chutney and lime

## Instructions
1. Marinate {protein} in tikka spices (30 min)
2. Grill or pan-fry until charred
3. Warm {flatbread}s
4. Assemble: {flatbread} + tikka {protein} + vegetables
5. Drizzle mint chutney, squeeze lime

## Why This Works
{flatbread} is perfect here - it's a flatbread just like tortilla!
The Indian spices create an amazing fusion flavor.
""",
        },
        "indo_italian": {
            "required": {"grain", "aromatic"},
            "description": "Indian-Italian fusion",
            "template": """# Masala Pasta

Italian pasta meets Indian spices!

## Ingredients
- 300g {grain}
- {aromatics}
- Indian spices (cumin, garam masala, turmeric)
- Tomatoes and cream
- {vegetables}

## Instructions
1. Cook {grain} al dente
2. Sauté {aromatics} with Indian spices
3. Add tomatoes, simmer to sauce
4. Add cream for richness
5. Toss with {grain}

## The Magic
The familiar pasta format with bold Indian flavors!
""",
        },
    },
}


# ============================================================================
# TRAINING EXAMPLE GENERATOR
# ============================================================================

class RoleBasedTrainingGenerator:
    def __init__(self):
        self.ingredient_roles = INGREDIENT_ROLES
        self.dish_patterns = DISH_PATTERNS
        self.output_file = Path("data/training/role_based_cuisine_examples.jsonl")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_roles(self, ingredient: str) -> List[str]:
        """Get all roles an ingredient can play"""
        ing_lower = ingredient.lower().replace('_', ' ')
        if ing_lower in self.ingredient_roles:
            return self.ingredient_roles[ing_lower]
        # Partial match
        for key, roles in self.ingredient_roles.items():
            if key in ing_lower or ing_lower in key:
                return roles
        return ["ingredient"]
    
    def ingredients_satisfy_dish(self, ingredients: List[str], dish_pattern: Dict) -> Tuple[bool, Dict]:
        """Check if ingredients can make this dish, return role mappings"""
        required_roles = dish_pattern.get("required", set())
        
        # Build available roles from ingredients
        role_to_ingredients = {}
        for ing in ingredients:
            for role in self.get_roles(ing):
                if role not in role_to_ingredients:
                    role_to_ingredients[role] = []
                role_to_ingredients[role].append(ing)
        
        # Check all required roles are satisfied
        mappings = {}
        for role in required_roles:
            if role in role_to_ingredients:
                mappings[role] = role_to_ingredients[role][0]
            else:
                return False, {}
        
        return True, mappings
    
    def generate_recipe_from_template(
        self, 
        template: str, 
        role_mappings: Dict,
        ingredients: List[str]
    ) -> str:
        """Fill in recipe template with actual ingredients"""
        recipe = template
        
        # Map roles to ingredients
        protein = role_mappings.get("protein", "protein")
        flatbread = role_mappings.get("flatbread", "flatbread")
        grain = role_mappings.get("grain", "rice")
        
        # Get vegetables
        vegetables = [i for i in ingredients 
                     if "vegetable" in self.get_roles(i) and i not in role_mappings.values()]
        veg_str = ", ".join(vegetables[:3]) if vegetables else "onion, tomato"
        
        # Get aromatics
        aromatics = [i for i in ingredients if "aromatic" in self.get_roles(i)]
        arom_str = ", ".join(aromatics) if aromatics else "garlic, ginger"
        
        # Get dairy
        dairy = [i for i in ingredients if "dairy" in self.get_roles(i)]
        dairy_str = dairy[0] if dairy else "butter"
        
        # Replace placeholders
        recipe = recipe.replace("{protein}", protein.title())
        recipe = recipe.replace("{Protein}", protein.title())
        recipe = recipe.replace("{flatbread}", flatbread)
        recipe = recipe.replace("{grain}", grain)
        recipe = recipe.replace("{vegetables}", veg_str)
        recipe = recipe.replace("{vegetable}", veg_str.split(",")[0])
        recipe = recipe.replace("{Vegetable}", veg_str.split(",")[0].title())
        recipe = recipe.replace("{aromatics}", arom_str)
        recipe = recipe.replace("{dairy}", dairy_str)
        recipe = recipe.replace("{cheese}", "cheese")
        recipe = recipe.replace("{sauce_base}", "tomato")
        
        # Handle optional sections
        recipe = recipe.replace("{with_protein}", f" with {protein.title()}" if "protein" in role_mappings else "")
        recipe = recipe.replace("{protein_line}", f"- 200g {protein}" if "protein" in role_mappings else "")
        recipe = recipe.replace("{vegetable_line}", f"- {veg_str}" if vegetables else "")
        recipe = recipe.replace("{protein_add}", f", {protein}" if "protein" in role_mappings else "")
        recipe = recipe.replace("{vegetable_add}", f", {veg_str}" if vegetables else "")
        recipe = recipe.replace("{protein_step}", f"Cook {protein} separately, add to sauce." if "protein" in role_mappings else "")
        recipe = recipe.replace("{style}", "")
        
        return recipe
    
    def generate_example(
        self, 
        ingredients: List[str], 
        cuisine: str, 
        dish_name: str,
        dish_pattern: Dict,
        role_mappings: Dict
    ) -> Dict:
        """Generate a single training example"""
        
        # Create instruction (input)
        ing_str = ", ".join(ingredients)
        
        # Varied instruction templates
        instruction_templates = [
            f"I have these ingredients: {ing_str}. Make a {cuisine} dish.",
            f"Create a {cuisine} recipe using: {ing_str}",
            f"What {cuisine} dish can I make with {ing_str}?",
            f"Make me something {cuisine} with: {ing_str}",
            f"Ingredients available: {ing_str}. Cuisine preference: {cuisine}. What can I cook?",
            f"Using {ing_str}, prepare a {cuisine} meal.",
            f"I want to cook {cuisine} food. I have: {ing_str}",
        ]
        
        instruction = random.choice(instruction_templates)
        
        # Generate recipe output
        template = dish_pattern.get("template", "")
        if template:
            recipe = self.generate_recipe_from_template(template, role_mappings, ingredients)
        else:
            # Fallback
            recipe = f"# {dish_name.title()}\n\nUsing: {ing_str}\n\n(Recipe details...)"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": recipe,
            "task_type": "cuisine_context_generation",
            "cuisine": cuisine,
            "dish": dish_name,
            "ingredients": ingredients,
            "role_mappings": role_mappings,
        }
    
    def generate_cross_cuisine_pair(self, base_ingredients: List[str]) -> List[Dict]:
        """
        Generate examples showing SAME ingredients → DIFFERENT dishes based on cuisine.
        This is the KEY training signal we need!
        """
        examples = []
        
        for cuisine, dishes in self.dish_patterns.items():
            for dish_name, pattern in dishes.items():
                can_make, mappings = self.ingredients_satisfy_dish(base_ingredients, pattern)
                if can_make:
                    example = self.generate_example(
                        base_ingredients, cuisine, dish_name, pattern, mappings
                    )
                    examples.append(example)
        
        return examples
    
    def generate_substitution_awareness_example(self) -> Dict:
        """
        Generate example explicitly teaching role-based substitution.
        E.g., "Roti can be used as tortilla because both are flatbreads"
        """
        # Define substitution pairs with explanations
        substitution_pairs = [
            ("roti", "tortilla", "flatbread", "Both are flatbreads - roti works perfectly as a wrap in Mexican dishes!"),
            ("chapati", "tortilla", "flatbread", "Chapati is essentially the same as a flour tortilla - use interchangeably."),
            ("paneer", "tofu", "protein", "Both are protein-rich, mild-flavored blocks that absorb spices well."),
            ("paneer", "chicken", "protein", "Both serve as the main protein - adjust cooking time for chicken."),
            ("yogurt", "sour cream", "dairy/sauce", "Both add creaminess and tang - yogurt is a great healthier substitute."),
            ("ghee", "butter", "fat", "Ghee has a nuttier flavor but works the same way as butter."),
            ("rice", "quinoa", "grain", "Both are grains that serve as a base - quinoa adds more protein."),
            ("cilantro", "coriander leaves", "herb", "Same plant! Cilantro is the American name for coriander leaves."),
        ]
        
        original, substitute, role, explanation = random.choice(substitution_pairs)
        
        instruction_templates = [
            f"Can I use {original} instead of {substitute}?",
            f"Is {original} a good substitute for {substitute}?",
            f"I don't have {substitute}, can I use {original}?",
            f"Will {original} work in place of {substitute}?",
        ]
        
        output = f"""Yes! {original.title()} works great as a substitute for {substitute}.

**Why it works:** {explanation}

**Role:** Both serve as {role} in recipes.

**Tips:**
- Use 1:1 ratio
- Adjust cooking time if needed
- The flavor will be slightly different but the dish will work well

**Best for:** Any recipe where {substitute} is used as a {role}."""

        return {
            "instruction": random.choice(instruction_templates),
            "input": "",
            "output": output,
            "task_type": "substitution_role_explanation",
            "original": original,
            "substitute": substitute,
            "role": role,
        }
    
    def generate_all_examples(self, num_examples: int = 10000) -> List[Dict]:
        """Generate complete training dataset"""
        examples = []
        
        # Common ingredient combinations
        ingredient_sets = [
            ["chicken", "tomato", "onion", "roti"],
            ["chicken", "tomato", "onion", "rice"],
            ["paneer", "tomato", "onion", "capsicum", "roti"],
            ["paneer", "spinach", "cream", "garlic"],
            ["egg", "onion", "tomato", "bread"],
            ["tofu", "cabbage", "carrot", "soy sauce"],
            ["fish", "tomato", "garlic", "rice"],
            ["chickpeas", "tomato", "onion", "cumin"],
            ["potato", "cauliflower", "cumin", "turmeric"],
            ["chicken", "cream", "tomato", "garam masala", "rice"],
            ["mutton", "rice", "yogurt", "onion"],
            ["pasta", "tomato", "garlic", "basil"],
            ["noodles", "cabbage", "carrot", "soy sauce"],
        ]
        
        print("Generating cross-cuisine examples...")
        for _ in tqdm(range(num_examples // 2)):
            # Either use predefined set or create random combination
            if random.random() < 0.5:
                ingredients = random.choice(ingredient_sets)
            else:
                # Random combination
                proteins = ["chicken", "paneer", "tofu", "egg", "fish", "chickpeas"]
                vegetables = ["tomato", "onion", "capsicum", "potato", "spinach", "cabbage", "carrot"]
                bases = ["rice", "roti", "pasta", "noodles", "bread"]
                
                ingredients = [
                    random.choice(proteins),
                    random.choice(vegetables),
                    random.choice(vegetables),
                    random.choice(bases),
                ]
            
            # Add some aromatics randomly
            if random.random() < 0.5:
                ingredients.append(random.choice(["garlic", "ginger", "onion"]))
            
            # Generate cross-cuisine examples
            cross_examples = self.generate_cross_cuisine_pair(ingredients)
            examples.extend(cross_examples)
        
        print("Generating substitution awareness examples...")
        for _ in tqdm(range(num_examples // 4)):
            example = self.generate_substitution_awareness_example()
            examples.append(example)
        
        # Shuffle
        random.shuffle(examples)
        
        return examples[:num_examples]
    
    def save_examples(self, examples: List[Dict]):
        """Save to JSONL"""
        print(f"Saving {len(examples)} examples to {self.output_file}")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print("Done!")
        
        # Stats
        task_counts = {}
        cuisine_counts = {}
        for ex in examples:
            task = ex.get('task_type', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
            
            cuisine = ex.get('cuisine', 'unknown')
            cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
        
        print("\nTask distribution:")
        for task, count in task_counts.items():
            print(f"  {task}: {count}")
        
        print("\nCuisine distribution:")
        for cuisine, count in sorted(cuisine_counts.items()):
            print(f"  {cuisine}: {count}")
    
    def run(self, num_examples: int = 10000):
        """Main entry point"""
        print("="*70)
        print("GENERATING ROLE-BASED CUISINE-AWARE TRAINING DATA")
        print("="*70)
        print(f"\nTarget: {num_examples} examples")
        print("This teaches the model: Same ingredients + different cuisine = different dish\n")
        
        examples = self.generate_all_examples(num_examples)
        self.save_examples(examples)
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"\nTo fine-tune, add this file to your training data and run:")
        print("  python scripts/recipe_generation/train_llama.py --additional_data data/training/role_based_cuisine_examples.jsonl")


def main():
    generator = RoleBasedTrainingGenerator()
    generator.run(num_examples=10000)


if __name__ == "__main__":
    main()