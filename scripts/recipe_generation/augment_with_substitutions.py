"""
Augment with Substitutions
Generates synthetic training data teaching the model context-aware substitutions.
This is critical for learning WHY and HOW to substitute, not just WHAT to substitute.
"""

import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add scripts directory to path to enable importing substitution_learning
script_dir = Path(__file__).parent.parent  # Go up from recipe_generation to scripts
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from substitution_learning.substitution_ranker import SubstitutionRanker

class SubstitutionAugmentor:
    def __init__(self):
        self.input_dir = Path("data/training")
        self.output_file = self.input_dir / "augmented_substitutions.jsonl"
        
        # Initialize substitution ranker
        print("Loading substitution ranker...")
        self.ranker = SubstitutionRanker(
            data_yaml_path=Path("data/merged/data.yaml"),
            embeddings_path=Path("data/embeddings/semantic_embeddings.pt"),
            pmi_matrix_path=Path("data/substitution/pmi_matrix.npz"),
            categories_path=Path("data/substitution/ingredient_categories.json")
        )
        
        # Ingredient properties for context generation
        self.ingredient_properties = self.build_ingredient_properties()
        
        # Targets
        self.num_substitution_examples = 50000
        self.num_fusion_examples = 20000
        self.num_explanation_examples = 10000
    
    def build_ingredient_properties(self):
        """Build comprehensive knowledge base of ingredient properties"""
        return {
            # Proteins - detailed properties
            'Chicken': {
                'texture': 'tender', 'cook_time': 'medium', 'flavor': 'mild',
                'cook_temp': 165, 'methods': ['grill', 'bake', 'fry', 'boil'],
                'prep': 'trim fat, cut evenly'
            },
            'Beef': {
                'texture': 'firm', 'cook_time': 'long', 'flavor': 'strong',
                'cook_temp': 145, 'methods': ['grill', 'roast', 'braise', 'stir-fry'],
                'prep': 'marinate for tenderness'
            },
            'Tofu': {
                'texture': 'soft', 'cook_time': 'short', 'flavor': 'neutral',
                'cook_temp': 0, 'methods': ['pan-fry', 'bake', 'stir-fry', 'steam'],
                'prep': 'press to remove water, cut into cubes'
            },
            'Fish': {
                'texture': 'delicate', 'cook_time': 'short', 'flavor': 'light',
                'cook_temp': 145, 'methods': ['bake', 'grill', 'steam', 'poach'],
                'prep': 'pat dry, season gently'
            },
            'Pork': {
                'texture': 'tender', 'cook_time': 'medium', 'flavor': 'rich',
                'cook_temp': 145, 'methods': ['roast', 'grill', 'braise', 'fry'],
                'prep': 'score fat, season well'
            },
            'Egg': {
                'texture': 'soft', 'cook_time': 'short', 'flavor': 'mild',
                'cook_temp': 0, 'methods': ['fry', 'boil', 'scramble', 'bake'],
                'prep': 'room temperature for baking'
            },
            
            # Vegetables - context-aware
            'Tomato': {
                'texture': 'juicy', 'cook_time': 'short', 'role': 'sauce_base',
                'methods': ['roast', 'sauté', 'raw'], 'water_content': 'high'
            },
            'Potato': {
                'texture': 'starchy', 'cook_time': 'long', 'role': 'carb_base',
                'methods': ['boil', 'roast', 'fry', 'bake'], 'water_content': 'medium'
            },
            'Onion': {
                'texture': 'crisp', 'cook_time': 'medium', 'role': 'aromatic',
                'methods': ['sauté', 'caramelize', 'raw', 'grill'], 'water_content': 'high'
            },
            'Carrot': {
                'texture': 'crunchy', 'cook_time': 'medium', 'role': 'vegetable',
                'methods': ['roast', 'steam', 'boil', 'raw'], 'water_content': 'medium'
            },
            'Broccoli': {
                'texture': 'crisp', 'cook_time': 'short', 'role': 'vegetable',
                'methods': ['steam', 'roast', 'stir-fry', 'blanch'], 'water_content': 'medium'
            },
            
            # Dairy - state and function
            'Milk': {
                'state': 'liquid', 'fat_content': 'medium', 'role': 'liquid_base',
                'substitution_ratio': 1.0, 'usage': 'sauces, baking, drinks'
            },
            'Butter': {
                'state': 'solid', 'fat_content': 'high', 'role': 'fat_source',
                'substitution_ratio': 0.75, 'usage': 'sautéing, baking, flavoring'
            },
            'Cheese': {
                'state': 'solid', 'fat_content': 'high', 'role': 'flavor_topping',
                'substitution_ratio': 1.0, 'usage': 'topping, melting, flavoring'
            },
            
            # Grains - cultural and physical
            'Rice -Chamal-': {
                'texture': 'fluffy', 'culture': 'asian', 'form': 'grain',
                'cook_time': 'medium', 'water_ratio': 2.0
            },
            'Wheat': {
                'texture': 'chewy', 'culture': 'western', 'form': 'grain',
                'cook_time': 'long', 'water_ratio': 2.5
            },
            'Corn': {
                'texture': 'crunchy', 'culture': 'americas', 'form': 'grain',
                'cook_time': 'short', 'water_ratio': 1.5
            },
        }
    
    def load_base_recipes(self):
        """Load base recipes for augmentation"""
        print("Loading base recipes...")
        recipes = []
        
        with open(self.input_dir / "train_instructions.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                recipe = json.loads(line)
                # Only use recipes with mapped ingredients
                if 'ingredients' in recipe:
                    recipes.append(recipe)
        
        print(f"Loaded {len(recipes)} base recipes")
        return recipes
    
    def get_substitution_context(self, original, substitute):
        """Generate detailed context explanation for substitution"""
        orig_props = self.ingredient_properties.get(original, {})
        sub_props = self.ingredient_properties.get(substitute, {})
        
        adjustments = []
        
        # Cooking time adjustments
        orig_time = orig_props.get('cook_time', 'medium')
        sub_time = sub_props.get('cook_time', 'medium')
        if orig_time != sub_time:
            if orig_time == 'long' and sub_time == 'short':
                adjustments.append("⏱️ Reduce cooking time by 30-40% (substitute cooks faster)")
            elif orig_time == 'short' and sub_time == 'long':
                adjustments.append("⏱️ Increase cooking time by 40-50% (substitute needs more time)")
            elif orig_time == 'medium' and sub_time == 'short':
                adjustments.append("⏱️ Reduce cooking time by 20-25%")
            elif orig_time == 'medium' and sub_time == 'long':
                adjustments.append("⏱️ Increase cooking time by 25-30%")
        
        # Temperature adjustments
        orig_temp = orig_props.get('cook_temp', 0)
        sub_temp = sub_props.get('cook_temp', 0)
        if orig_temp > 0 and sub_temp > 0 and orig_temp != sub_temp:
            adjustments.append(f"🌡️ Adjust internal temperature target: {orig_temp}°F → {sub_temp}°F")
        
        # Texture considerations
        orig_texture = orig_props.get('texture', 'standard')
        sub_texture = sub_props.get('texture', 'standard')
        if orig_texture != sub_texture:
            if sub_texture == 'delicate':
                adjustments.append("🥢 Handle gently to prevent breaking (delicate texture)")
            elif sub_texture == 'soft':
                adjustments.append("🥢 May need coating/breading for better texture")
            elif sub_texture == 'firm':
                adjustments.append("🥢 Can withstand higher heat and longer cooking")
        
        # Preparation requirements
        sub_prep = sub_props.get('prep', '')
        if sub_prep:
            adjustments.append(f"🔪 Prep: {sub_prep}")
        
        # Flavor profile
        orig_flavor = orig_props.get('flavor', 'neutral')
        sub_flavor = sub_props.get('flavor', 'neutral')
        if orig_flavor != sub_flavor:
            if sub_flavor == 'neutral':
                adjustments.append("🌶️ Add extra seasoning (substitute has milder flavor)")
            elif sub_flavor == 'strong':
                adjustments.append("🌶️ Reduce strong seasonings (substitute has bold flavor)")
        
        # State change (dairy)
        orig_state = orig_props.get('state', '')
        sub_state = sub_props.get('state', '')
        if orig_state and sub_state and orig_state != sub_state:
            if orig_state == 'solid' and sub_state == 'liquid':
                adjustments.append("💧 Liquid substitute: reduce other liquids in recipe by 2-3 tbsp per cup")
            elif orig_state == 'liquid' and sub_state == 'solid':
                adjustments.append("💧 Solid substitute: add liquid (milk/water) to compensate")
            
            # Substitution ratio
            sub_ratio = sub_props.get('substitution_ratio', 1.0)
            if sub_ratio != 1.0:
                adjustments.append(f"📏 Use {sub_ratio}x amount (e.g., 1 cup → {sub_ratio} cup)")
        
        # Method compatibility
        orig_methods = set(orig_props.get('methods', []))
        sub_methods = set(sub_props.get('methods', []))
        if orig_methods and sub_methods:
            incompatible = orig_methods - sub_methods
            if incompatible:
                compatible = list(orig_methods & sub_methods)
                if compatible:
                    adjustments.append(f"🍳 Best cooking methods: {', '.join(compatible[:3])}")
        
        # Water content (for vegetables)
        orig_water = orig_props.get('water_content', '')
        sub_water = sub_props.get('water_content', '')
        if orig_water and sub_water and orig_water != sub_water:
            if orig_water == 'low' and sub_water == 'high':
                adjustments.append("💦 High water content: drain/press before cooking to avoid sogginess")
            elif orig_water == 'high' and sub_water == 'low':
                adjustments.append("💦 Lower water content: may need additional liquid or shorter cooking")
        
        # Default message if no specific adjustments
        if not adjustments:
            adjustments.append("✓ Direct 1:1 substitution works well with minimal adjustments")
        
        return "\n".join(adjustments)
    
    def generate_substitution_example(self, recipe):
        """Generate substitution adaptation example with quality validation"""
        # Find ingredients we can substitute
        available_ingredients = []
        for ing in recipe['ingredients']:
            # Check if ingredient exists in our detection classes
            matches = [cls for cls in self.ranker.detection_classes if cls.lower() in ing.lower()]
            if matches:
                available_ingredients.append((ing, matches[0]))
        
        if not available_ingredients:
            return None
        
        # Pick random ingredient to substitute
        original_ing_text, original_class = random.choice(available_ingredients)
        
        # Get substitution
        substitutions = self.ranker.get_substitutions(original_class, top_k=5)
        if not substitutions:
            return None
        
        # Pick substitute (prefer top 3 for quality)
        substitute_class = substitutions[random.randint(0, min(2, len(substitutions)-1))][0]
        
        # QUALITY CHECK: Skip if substitute is too similar (likely duplicate)
        if original_class.lower() == substitute_class.lower():
            return None
        
        # QUALITY CHECK: Skip if substitute is in original recipe (circular)
        if any(substitute_class.lower() in ing.lower() for ing in recipe['ingredients']):
            return None
        
        # Get context
        context = self.get_substitution_context(original_class, substitute_class)
        
        # QUALITY CHECK: Ensure context is meaningful (not just default message)
        if "Direct 1:1 substitution" in context and len(context.split('\n')) < 2:
            # If only default message, skip low-quality example
            # Only keep if we have at least 2 adjustment points
            return None
        
        # Create instruction with more variance
        instruction_templates = [
            f"I want to make this recipe but need to substitute {original_class} with {substitute_class}. How should I adjust it?\n\nOriginal recipe:\n{recipe['output'][:500]}...",
            
            f"Adapt this recipe by replacing {original_class} with {substitute_class}. What changes are needed?\n\nRecipe:\n{recipe['output'][:500]}...",
            
            f"I'm allergic to {original_class}. Can I use {substitute_class} instead in this recipe? What adjustments?\n\n{recipe['output'][:500]}...",
            
            f"Convert this recipe to use {substitute_class} instead of {original_class}. Explain the modifications:\n\n{recipe['output'][:500]}..."
        ]
        
        instruction = random.choice(instruction_templates)
        
        # Create adapted output with detailed explanation
        output = f"""**Substitution: {original_class} → {substitute_class}**

**Analysis & Adjustments:**
{context}

**Modified Recipe Approach:**
Since we're replacing {original_class} with {substitute_class}, here's how the recipe changes:

1. **Ingredient Swap**: Replace {original_class} with {substitute_class} (adjust quantity/prep as noted above)

2. **Cooking Method**: Review the adjustments above - you may need to modify temperature, time, or technique

3. **Seasoning**: Taste and adjust seasonings to complement the {substitute_class}'s flavor profile

4. **Final Notes**: The texture and flavor will differ slightly from the original, but this substitution maintains the dish's essence while accommodating your needs.

**Expected Outcome:**
The recipe will work successfully with {substitute_class}, producing a similar dish with its own character.
"""
        
        return {
            'instruction': instruction,
            'input': '',
            'output': output,
            'task_type': 'substitution_adaptation',
            'original_ingredient': original_class,
            'substitute_ingredient': substitute_class,
            'quality_score': len(context.split('\n'))  # More adjustments = higher quality
        }
    
    def generate_fusion_example(self, recipe):
        """Generate fusion recipe example"""
        # Fusion cuisine pairs
        fusion_pairs = [
            ('indian', 'mexican', 'Indo-Mexican'),
            ('italian', 'american', 'Italian-American'),
            ('chinese', 'mexican', 'Asian-Latin'),
            ('thai', 'italian', 'Thai-Italian'),
            ('indian', 'italian', 'Indo-Italian'),
            ('mexican', 'mediterranean', 'Mexiterranean'),
        ]
        
        # Select fusion target
        base_cuisine = recipe.get('cuisine', 'general')
        possible_fusions = [name for c1, c2, name in fusion_pairs if c1 == base_cuisine or c2 == base_cuisine]
        
        if not possible_fusions:
            # Create generic fusion
            fusion_cuisines = ['mexican', 'indian', 'italian', 'thai', 'chinese']
            target = random.choice([c for c in fusion_cuisines if c != base_cuisine])
            fusion_name = f"{base_cuisine.title()}-{target.title()} Fusion"
        else:
            fusion_name = random.choice(possible_fusions)
        
        # Create instruction
        ingredients_list = ', '.join(recipe['ingredients'][:8])
        instruction = f"""Create a {fusion_name} fusion recipe using: {ingredients_list}. Blend cooking techniques and flavors from both cuisines."""
        
        # Create output
        output = f"""**{fusion_name} Fusion Recipe**

**Fusion Concept:**
This recipe combines traditional {base_cuisine} ingredients with {fusion_name.split('-')[1]} cooking techniques and flavor profiles, creating a unique culinary experience.

**Cultural Elements:**
- Base cuisine: {base_cuisine} (ingredient selection, core flavors)
- Fusion elements: [Target cuisine techniques and seasonings]

**Ingredients:**
[Recipe ingredients with fusion additions]

**Instructions:**
[Cooking method blending both traditions]

**Fusion Notes:**
- Traditional meets modern: honors both culinary traditions
- Flavor balance: complementary spice profiles
- Technique fusion: combines cooking methods from both cultures
"""
        
        return {
            'instruction': instruction,
            'input': '',
            'output': output,
            'task_type': 'fusion_recipe',
            'base_cuisine': base_cuisine,
            'fusion_type': fusion_name
        }
    
    def generate_explanation_example(self):
        """Generate substitution explanation example"""
        # Pick random ingredient pair from ranker
        ingredient = random.choice(self.ranker.detection_classes)
        substitutions = self.ranker.get_substitutions(ingredient, top_k=3)
        
        if not substitutions:
            return None
        
        substitute = substitutions[0][0]
        
        # Create instruction
        instruction = f"Explain how to substitute {ingredient} with {substitute} in cooking. What adjustments are needed and why?"
        
        # Get context
        context = self.get_substitution_context(ingredient, substitute)
        
        # Create detailed explanation
        output = f"""**Substituting {ingredient} with {substitute}**

**Why This Substitution Works:**
Both ingredients share similar functional properties in recipes, though they differ in specific characteristics that require adjustment.

**Key Differences:**
{context}

**Adjustment Guidelines:**

1. **Ratio Adjustment:**
   - Generally use 1:1 substitution
   - May need slight adjustment based on moisture content

2. **Cooking Method:**
   - [Specific adjustments based on ingredient properties]

3. **Flavor Compensation:**
   - [Seasoning adjustments needed]

4. **Texture Considerations:**
   - [How final dish texture will change]

**Best Used In:**
- [Types of dishes where this substitution works well]

**Avoid In:**
- [Dishes where this substitution may not work]

**Pro Tips:**
- [Additional chef's insights for best results]
"""
        
        return {
            'instruction': instruction,
            'input': '',
            'output': output,
            'task_type': 'substitution_explanation',
            'original_ingredient': ingredient,
            'substitute_ingredient': substitute
        }
    
    def augment(self):
        """Main augmentation pipeline"""
        print("="*70)
        print("AUGMENTING WITH SUBSTITUTION DATA")
        print("="*70)
        
        # Load base recipes
        recipes = self.load_base_recipes()
        
        augmented_data = []
        failed_count = {'substitution': 0, 'fusion': 0, 'explanation': 0}
        
        # Generate substitution adaptations
        print(f"\nGenerating {self.num_substitution_examples} substitution examples...")
        for _ in tqdm(range(self.num_substitution_examples)):
            try:
                recipe = random.choice(recipes)
                example = self.generate_substitution_example(recipe)
                if example:
                    augmented_data.append(example)
                else:
                    failed_count['substitution'] += 1
            except Exception as e:
                failed_count['substitution'] += 1
                if failed_count['substitution'] < 10:  # Only print first 10 errors
                    print(f"\nWarning: Substitution generation error: {e}")
        
        # Generate fusion recipes
        print(f"\nGenerating {self.num_fusion_examples} fusion examples...")
        for _ in tqdm(range(self.num_fusion_examples)):
            try:
                recipe = random.choice(recipes)
                example = self.generate_fusion_example(recipe)
                if example:
                    augmented_data.append(example)
                else:
                    failed_count['fusion'] += 1
            except Exception as e:
                failed_count['fusion'] += 1
                if failed_count['fusion'] < 10:
                    print(f"\nWarning: Fusion generation error: {e}")
        
        # Generate explanations
        print(f"\nGenerating {self.num_explanation_examples} explanation examples...")
        for _ in tqdm(range(self.num_explanation_examples)):
            try:
                example = self.generate_explanation_example()
                if example:
                    augmented_data.append(example)
                else:
                    failed_count['explanation'] += 1
            except Exception as e:
                failed_count['explanation'] += 1
                if failed_count['explanation'] < 10:
                    print(f"\nWarning: Explanation generation error: {e}")
        
        # Report failures
        print(f"\nGeneration summary:")
        print(f"  Substitutions: {self.num_substitution_examples - failed_count['substitution']}/{self.num_substitution_examples} succeeded ({failed_count['substitution']} failed)")
        print(f"  Fusions: {self.num_fusion_examples - failed_count['fusion']}/{self.num_fusion_examples} succeeded ({failed_count['fusion']} failed)")
        print(f"  Explanations: {self.num_explanation_examples - failed_count['explanation']}/{self.num_explanation_examples} succeeded ({failed_count['explanation']} failed)")
        
        # Shuffle
        random.shuffle(augmented_data)
        
        # Save
        print(f"\nSaving {len(augmented_data)} augmented examples...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved to {self.output_file}")
        
        # Stats
        task_counts = {}
        for item in augmented_data:
            task = item.get('task_type', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print("\nAugmentation complete:")
        print(f"  Total examples: {len(augmented_data)}")
        for task, count in task_counts.items():
            print(f"  {task}: {count}")
        
        print("\n" + "="*70)
        print("AUGMENTATION COMPLETE")
        print("="*70)

def main():
    augmentor = SubstitutionAugmentor()
    augmentor.augment()

if __name__ == "__main__":
    main()