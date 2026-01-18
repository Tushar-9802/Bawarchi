"""
Prepare Training Data
Converts Food.com recipes to instruction-following format for Llama training.
Implements quality filtering and curriculum-aware dataset creation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from collections import Counter

class DataPreparator:
    def __init__(self):
        self.recipes_file = Path("data/recipes/food_com/recipes.parquet")
        self.output_dir = Path("data/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.min_ingredients = 3
        self.max_ingredients = 20
        self.min_instructions_length = 100
        self.max_instructions_length = 2000
        self.min_cook_time = 5  # minutes
        self.max_cook_time = 480  # 8 hours
        
        # Curriculum bins
        self.curriculum_bins = {
            'simple': {'min_ing': 3, 'max_ing': 6, 'difficulty': 'easy'},
            'medium': {'min_ing': 5, 'max_ing': 10, 'difficulty': 'medium'},
            'complex': {'min_ing': 8, 'max_ing': 15, 'difficulty': 'hard'}
        }
    
    def load_recipes(self):
        """Load and validate recipes"""
        print("Loading recipes from Food.com...")
        df = pd.read_parquet(self.recipes_file)
        print(f"Loaded {len(df)} recipes")
        
        # Debug: Show available columns
        print(f"\nAvailable columns: {list(df.columns)}")
        print(f"\nFirst recipe sample:")
        if len(df) > 0:
            first = df.iloc[0]
            for col in df.columns:
                val = first[col]
                if isinstance(val, str):
                    print(f"  {col}: {val[:100]}..." if len(str(val)) > 100 else f"  {col}: {val}")
                else:
                    print(f"  {col}: {type(val).__name__}")
        print()
        
        return df
    
    def clean_instructions(self, instructions):
        """Clean and format instruction text with quality checks"""
        # Early exit for None/empty
        if instructions is None:
            return None
        
        # Handle ndarray (Food.com format)
        if hasattr(instructions, '__len__') and not isinstance(instructions, str):
            # It's an array, convert to string
            try:
                if len(instructions) == 0:
                    return None
                # Join array elements or take first element
                if len(instructions) == 1:
                    instructions = str(instructions[0])
                else:
                    # Multiple instruction steps in array
                    instructions = ' '.join([str(s) for s in instructions if s])
            except (TypeError, ValueError):
                return None
        
        # Handle empty strings, whitespace
        if isinstance(instructions, str):
            instructions = instructions.strip()
            if not instructions:
                return None
        
        # Convert to string if needed (handles other types)
        if not isinstance(instructions, str):
            try:
                instructions = str(instructions).strip()
                if not instructions or instructions == 'nan':
                    return None
            except:
                return None
        
        # Check minimum length first
        if len(instructions) < self.min_instructions_length:
            return None
        
        instructions = re.sub(r'\s+', ' ', instructions)  # Normalize whitespace
        instructions = re.sub(r'\.{2,}', '.', instructions)  # Fix multiple periods
        
        # Quality checks - RELAXED for Food.com format
        # 1. Must have cooking-related words (very broad list)
        cooking_words = [
            'cook', 'heat', 'add', 'mix', 'stir', 'bake', 'fry', 'boil', 'cut', 'chop', 
            'blend', 'pour', 'season', 'place', 'put', 'combine', 'serve', 'prepare',
            'remove', 'slice', 'dice', 'mince', 'whisk', 'beat', 'fold', 'spread',
            'grill', 'roast', 'simmer', 'sauté', 'steam', 'microwave', 'chill',
            'cover', 'uncover', 'drain', 'rinse', 'preheat', 'garnish', 'sprinkle',
            'top', 'layer', 'arrange', 'transfer', 'set', 'let', 'allow', 'bring',
            'reduce', 'increase', 'adjust', 'taste', 'check', 'until', 'minute',
            'oven', 'pan', 'bowl', 'pot', 'dish', 'baking', 'cooking'
        ]
        instructions_lower = instructions.lower()
        if not any(word in instructions_lower for word in cooking_words):
            return None
        
        # 2. REMOVED: Multi-sentence check (too strict for Food.com)
        # Food.com uses various formats including step lists
        
        # 3. RELAXED: Filter only extremely garbled text
        # Simple check: ratio of alphabetic characters
        alpha_ratio = sum(c.isalpha() or c.isspace() or c in '.,;:-' for c in instructions) / max(len(instructions), 1)
        if alpha_ratio < 0.4:  # Relaxed from 0.6 to 0.4 (allow numbers, punctuation)
            return None
        
        # Check maximum length
        if len(instructions) > self.max_instructions_length:
            # Truncate at sentence boundary
            instructions = instructions[:self.max_instructions_length]
            last_period = instructions.rfind('.')
            if last_period > self.min_instructions_length:
                instructions = instructions[:last_period + 1]
        
        return instructions
    
    def parse_ingredients(self, ingredients):
        """Parse and clean ingredient list"""
        # Check for None or empty first
        if ingredients is None:
            return None
        
        # Handle numpy arrays and lists
        if hasattr(ingredients, '__len__') and not isinstance(ingredients, str):
            # It's already a list/array
            if len(ingredients) == 0:
                return None
            ingredients = list(ingredients)
        # Handle strings
        elif isinstance(ingredients, str):
            # Try to parse as list-like string
            ingredients = re.split(r'[,;]|\n', ingredients)
        # Single value or other type
        else:
            # Try to check if it's NaN (scalar)
            try:
                if pd.isna(ingredients):
                    return None
            except (ValueError, TypeError):
                # Not a scalar NaN, treat as single item
                ingredients = [str(ingredients)]
        
        if not isinstance(ingredients, list):
            return None
        
        # Clean each ingredient
        cleaned = []
        for ing in ingredients:
            # Check each individual element for NaN
            try:
                if pd.isna(ing):
                    continue
            except (ValueError, TypeError):
                # Not a scalar, continue with cleaning
                pass
            
            if not ing:
                continue
                
            ing = str(ing).strip()
            if len(ing) > 3:  # Skip very short entries
                cleaned.append(ing)
        
        # Check count
        if len(cleaned) < self.min_ingredients or len(cleaned) > self.max_ingredients:
            return None
        
        return cleaned
    
    def extract_cuisine(self, tags, keywords):
        """Extract cuisine from tags/keywords"""
        cuisine_keywords = {
            'italian': ['italian', 'italy', 'pasta', 'pizza'],
            'mexican': ['mexican', 'mexico', 'taco', 'burrito'],
            'indian': ['indian', 'india', 'curry', 'masala'],
            'chinese': ['chinese', 'china', 'stir-fry', 'wok'],
            'thai': ['thai', 'thailand'],
            'french': ['french', 'france'],
            'american': ['american', 'usa', 'bbq', 'burger'],
            'mediterranean': ['mediterranean', 'greek'],
            'asian': ['asian', 'oriental']
        }
        
        # Combine tags and keywords
        all_tags = []
        
        # Handle tags (could be string, list, or array)
        if tags is not None:
            try:
                if isinstance(tags, str):
                    all_tags.extend(tags.lower().split())
                elif isinstance(tags, (list, tuple)) or hasattr(tags, '__iter__'):
                    all_tags.extend([str(t).lower() for t in tags if t])
            except (TypeError, ValueError):
                pass
        
        # Handle keywords (could be string, list, or array)
        if keywords is not None:
            try:
                if isinstance(keywords, str):
                    all_tags.extend(keywords.lower().split())
                elif isinstance(keywords, (list, tuple)) or hasattr(keywords, '__iter__'):
                    all_tags.extend([str(k).lower() for k in keywords if k])
            except (TypeError, ValueError):
                pass
        
        # Match cuisine
        for cuisine, keywords in cuisine_keywords.items():
            for keyword in keywords:
                if any(keyword in tag for tag in all_tags):
                    return cuisine
        
        return 'general'
    
    def estimate_difficulty(self, num_ingredients, cook_time, instructions_length):
        """Estimate recipe difficulty"""
        score = 0
        
        # Ingredient complexity
        if num_ingredients <= 5:
            score += 1
        elif num_ingredients <= 10:
            score += 2
        else:
            score += 3
        
        # Time complexity
        if cook_time <= 30:
            score += 1
        elif cook_time <= 60:
            score += 2
        else:
            score += 3
        
        # Instruction complexity
        if instructions_length <= 300:
            score += 1
        elif instructions_length <= 800:
            score += 2
        else:
            score += 3
        
        # Map to difficulty
        if score <= 4:
            return 'easy'
        elif score <= 7:
            return 'medium'
        else:
            return 'hard'
    
    def format_recipe_output(self, row, ingredients):
        """Format recipe as structured output"""
        # Title - handle safely
        try:
            title = row.get('Name', 'Delicious Recipe')
            if title is None or (isinstance(title, float) and (title != title)):  # Check for NaN
                title = "Delicious Recipe"
            elif not title:
                title = "Delicious Recipe"
            else:
                title = str(title).strip().title()
        except (ValueError, TypeError):
            title = "Delicious Recipe"
        
        # Format ingredients with quantities
        ingredients_text = "\n".join([f"- {ing}" for ing in ingredients])
        
        # Instructions - use cleaned version
        instructions = row.get('CleanedInstructions', '')
        if not isinstance(instructions, str):
            instructions = str(instructions)
        
        # Format instructions as numbered steps
        steps = re.split(r'\.\s+', instructions.strip())
        steps = [s.strip() for s in steps if s.strip()]
        instructions_text = "\n".join([f"{i+1}. {step}." if not step.endswith('.') else f"{i+1}. {step}" 
                                       for i, step in enumerate(steps)])
        
        # Metadata - handle safely
        try:
            cook_time_raw = row.get('CookTime', 'N/A')
            if cook_time_raw and isinstance(cook_time_raw, str) and cook_time_raw.startswith('PT'):
                # Parse ISO 8601 duration
                import re
                hours = 0
                minutes = 0
                hour_match = re.search(r'(\d+)H', cook_time_raw)
                if hour_match:
                    hours = int(hour_match.group(1))
                minute_match = re.search(r'(\d+)M', cook_time_raw)
                if minute_match:
                    minutes = int(minute_match.group(1))
                cook_time = hours * 60 + minutes
            elif cook_time_raw:
                try:
                    cook_time = float(cook_time_raw)
                except:
                    cook_time = 'N/A'
            else:
                cook_time = 'N/A'
        except (ValueError, TypeError, AttributeError):
            cook_time = 'N/A'
        
        try:
            servings = row.get('RecipeServings', 'N/A')
            if servings is None or (isinstance(servings, float) and (servings != servings)):  # Check for NaN
                servings = 'N/A'
            elif servings:
                try:
                    servings = int(float(servings))
                except:
                    servings = 'N/A'
            else:
                servings = 'N/A'
        except (ValueError, TypeError):
            servings = 'N/A'
        
        # Assemble output
        output = f"""**{title}**

Ingredients:
{ingredients_text}

Instructions:
{instructions_text}

Servings: {servings} | Cook time: {cook_time} minutes"""
        
        return output
    
    def create_instruction(self, row, ingredients, cuisine, difficulty):
        """Create instruction prompt with varied styles"""
        # More natural, varied prompt styles
        templates = [
            f"Create a {difficulty}-level {cuisine} recipe using: {', '.join(ingredients[:10])}{'...' if len(ingredients) > 10 else ''}",
            f"I need a {cuisine} dish that's {difficulty} to make. I have: {', '.join(ingredients[:10])}{'...' if len(ingredients) > 10 else ''}",
            f"Generate a {cuisine} recipe (difficulty: {difficulty}). Available ingredients: {', '.join(ingredients[:10])}{'...' if len(ingredients) > 10 else ''}",
            f"Make a {difficulty} {cuisine} meal with {', '.join(ingredients[:8])}{'...' if len(ingredients) > 8 else ''}",
            f"Recipe needed: {cuisine} cuisine, {difficulty} difficulty level, using {', '.join(ingredients[:10])}{'...' if len(ingredients) > 10 else ''}",
            f"Please create a {difficulty} {cuisine} recipe. Ingredients: {', '.join(ingredients[:10])}{'...' if len(ingredients) > 10 else ''}",
        ]
        
        # Select template based on hash (deterministic but more varied)
        # Use Name as hash key - works for both Series and dict
        name = row.get('Name', 'recipe') if isinstance(row, dict) else row.get('Name', 'recipe')
        idx = hash(str(name)) % len(templates)
        return templates[idx]
    
    def process_recipes(self, df):
        """Process all recipes and create instruction pairs"""
        print("Processing recipes...")
        
        instructions_data = []
        stats = {
            'total': len(df),
            'processed': 0,
            'filtered_ingredients': 0,
            'filtered_instructions': 0,
            'filtered_time': 0,
            'by_difficulty': Counter(),
            'by_cuisine': Counter()
        }
        
        # Debug: Check first few recipes
        debug_count = 0
        max_debug = 5
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Debug first few recipes
            if debug_count < max_debug:
                if idx < 10:
                    debug_count += 1
                    print(f"\n[DEBUG] Recipe {idx}:")
                    print(f"  Name: {row.get('Name', 'N/A')[:50]}")
                    print(f"  Has RecipeInstructions: {row.get('RecipeInstructions') is not None}")
                    if row.get('RecipeInstructions') is not None:
                        inst = row.get('RecipeInstructions')
                        # Handle ndarray
                        if hasattr(inst, '__len__') and not isinstance(inst, str):
                            if len(inst) > 0:
                                inst = str(inst[0])[:100] if inst[0] else "Empty"
                        else:
                            inst = str(inst)[:100]
                        print(f"  Instructions preview: {inst}...")
            
            # Parse ingredients
            ingredients = self.parse_ingredients(row.get('RecipeIngredientParts'))
            if ingredients is None:
                stats['filtered_ingredients'] += 1
                continue
            
            # Clean instructions - USE CORRECT COLUMN NAME
            instructions = self.clean_instructions(row.get('RecipeInstructions'))
            if instructions is None:
                stats['filtered_instructions'] += 1
                if debug_count < max_debug and idx < 10:
                    print(f"  [FILTERED] Instructions did not pass quality checks")
                continue
            
            # Validate cook time - handle ISO 8601 duration format (PT24H, PT45M, etc.)
            cook_time_raw = row.get('CookTime', 0)
            cook_time = 0
            
            try:
                if cook_time_raw and isinstance(cook_time_raw, str):
                    # Parse ISO 8601 duration: PT24H, PT45M, PT1H30M, etc.
                    import re
                    duration_str = str(cook_time_raw)
                    
                    # Extract hours and minutes
                    hours = 0
                    minutes = 0
                    
                    hour_match = re.search(r'(\d+)H', duration_str)
                    if hour_match:
                        hours = int(hour_match.group(1))
                    
                    minute_match = re.search(r'(\d+)M', duration_str)
                    if minute_match:
                        minutes = int(minute_match.group(1))
                    
                    cook_time = hours * 60 + minutes
                elif cook_time_raw:
                    # Try direct conversion
                    cook_time = float(cook_time_raw)
            except (ValueError, TypeError, AttributeError):
                cook_time = 0
            
            if cook_time < self.min_cook_time or cook_time > self.max_cook_time:
                stats['filtered_time'] += 1
                continue
            
            # Extract metadata
            cuisine = self.extract_cuisine(row.get('Keywords'), row.get('RecipeCategory'))
            difficulty = self.estimate_difficulty(len(ingredients), cook_time, len(instructions))
            
            # Store cleaned instructions for format_recipe_output
            # Create a copy of row as dict to avoid SettingWithCopyWarning
            row_dict = row.to_dict()
            row_dict['CleanedInstructions'] = instructions
            
            # Create instruction pair
            instruction_text = self.create_instruction(row_dict, ingredients, cuisine, difficulty)
            output_text = self.format_recipe_output(row_dict, ingredients)
            
            instructions_data.append({
                'instruction': instruction_text,
                'input': '',
                'output': output_text,
                'ingredients': ingredients,
                'num_ingredients': len(ingredients),
                'cuisine': cuisine,
                'difficulty': difficulty,
                'cook_time': cook_time,
                'recipe_id': idx
            })
            
            stats['processed'] += 1
            stats['by_difficulty'][difficulty] += 1
            stats['by_cuisine'][cuisine] += 1
        
        print(f"\nProcessing complete:")
        print(f"  Total recipes: {stats['total']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Filtered (ingredients): {stats['filtered_ingredients']}")
        print(f"  Filtered (instructions): {stats['filtered_instructions']}")
        print(f"  Filtered (time): {stats['filtered_time']}")
        print(f"\nDifficulty distribution:")
        for diff, count in stats['by_difficulty'].most_common():
            print(f"  {diff}: {count} ({count/stats['processed']*100:.1f}%)")
        print(f"\nCuisine distribution (top 10):")
        for cuisine, count in stats['by_cuisine'].most_common(10):
            print(f"  {cuisine}: {count} ({count/stats['processed']*100:.1f}%)")
        
        return instructions_data, stats
    
    def split_data(self, data, train_ratio=0.77, val_ratio=0.12):
        """Split data into train/val/test sets with deduplication and stratification"""
        print("\nDeduplicating data...")
        
        # Check if we have data
        if not data:
            print("ERROR: No data to split! All recipes were filtered out.")
            print("\nPossible causes:")
            print("  - Quality filters too strict")
            print("  - Data format mismatch")
            print("  - All recipes missing required fields")
            return [], [], []
        
        # Simple deduplication based on normalized recipe names
        seen_names = set()
        deduplicated = []
        
        for item in data:
            # Normalize name for comparison
            name = item.get('recipe_id', '')
            # Use first 50 chars of output as fingerprint
            fingerprint = item['output'][:50].lower().strip()
            
            if fingerprint not in seen_names:
                seen_names.add(fingerprint)
                deduplicated.append(item)
        
        print(f"  Original: {len(data)} recipes")
        print(f"  After deduplication: {len(deduplicated)} recipes")
        if len(data) > 0:
            print(f"  Removed: {len(data) - len(deduplicated)} duplicates ({(len(data) - len(deduplicated))/len(data)*100:.1f}%)")
        
        if not deduplicated:
            print("ERROR: No recipes after deduplication!")
            return [], [], []
        
        data = deduplicated
        
        print("\nStratified splitting (balanced by difficulty)...")
        
        # Group by difficulty for stratified split
        difficulty_groups = {'easy': [], 'medium': [], 'hard': []}
        for item in data:
            diff = item.get('difficulty', 'medium')
            if diff in difficulty_groups:
                difficulty_groups[diff].append(item)
        
        train, val, test = [], [], []
        
        # Split each difficulty group proportionally
        np.random.seed(42)
        for diff, items in difficulty_groups.items():
            if not items:
                continue
            
            np.random.shuffle(items)
            n = len(items)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train.extend(items[:train_end])
            val.extend(items[train_end:val_end])
            test.extend(items[val_end:])
        
        # Final shuffle to mix difficulties
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
        
        n_total = len(train) + len(val) + len(test)
        print(f"  Train: {len(train)} ({len(train)/n_total*100:.1f}%)")
        print(f"  Val: {len(val)} ({len(val)/n_total*100:.1f}%)")
        print(f"  Test: {len(test)} ({len(test)/n_total*100:.1f}%)")
        
        # Print difficulty distribution per split
        for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
            diff_counts = Counter([item['difficulty'] for item in split_data])
            print(f"  {split_name} difficulty: ", end='')
            for diff in ['easy', 'medium', 'hard']:
                pct = diff_counts[diff] / len(split_data) * 100 if split_data else 0
                print(f"{diff}={pct:.1f}% ", end='')
            print()
        
        return train, val, test
    
    def save_data(self, train, val, test):
        """Save datasets as JSONL"""
        print("\nSaving datasets...")
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            filepath = self.output_dir / f"{name}_instructions.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  Saved {len(data)} examples to {filepath}")
    
    def prepare(self):
        """Main preparation pipeline"""
        print("="*70)
        print("PREPARING TRAINING DATA")
        print("="*70)
        
        # Load
        df = self.load_recipes()
        
        # Process
        data, stats = self.process_recipes(df)
        
        # Explicitly release DataFrame memory
        del df
        import gc
        gc.collect()
        print(f"\nDataFrame memory released")
        
        # Split
        train, val, test = self.split_data(data)
        
        # Check if we have valid splits
        if not train or not val or not test:
            print("\n" + "!"*70)
            print("FATAL ERROR: No valid data after filtering and splitting!")
            print("!"*70)
            print("\nDebugging information:")
            print(f"  Total recipes loaded: {stats['total']}")
            print(f"  Filtered (ingredients): {stats['filtered_ingredients']}")
            print(f"  Filtered (instructions): {stats['filtered_instructions']}")
            print(f"  Filtered (time): {stats['filtered_time']}")
            print(f"  Successfully processed: {stats['processed']}")
            print("\nMost likely cause: Instructions quality filter is too strict")
            print("Check the clean_instructions() method and relax the filters.")
            return
        
        # Save
        self.save_data(train, val, test)
        
        # Save stats
        stats_file = self.output_dir / "preparation_stats.json"
        with open(stats_file, 'w') as f:
            # Convert Counter to dict for JSON
            stats['by_difficulty'] = dict(stats['by_difficulty'])
            stats['by_cuisine'] = dict(stats['by_cuisine'])
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {stats_file}")
        
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETE")
        print("="*70)

def main():
    preparator = DataPreparator()
    preparator.prepare()

if __name__ == "__main__":
    main()