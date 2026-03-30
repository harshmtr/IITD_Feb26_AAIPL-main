"""
Optimized Q-Agent for Logical Reasoning
Generates 1000 high-quality MCQs across 4 topics
Target: 100% validity, Fast generation
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
from typing import List, Dict
import random

class LogicalReasoningQAgent:
    def __init__(self, model_path="unsloth/Qwen2-7B-Instruct"):
        """Initialize Q-Agent for logical reasoning"""
        print("🔧 Loading Logical Reasoning Q-Agent...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        # Define topics
        self.topics = {
            "syllogisms": {
                "count": 250,
                "templates": self._get_syllogism_templates()
            },
            "seating": {
                "count": 250,
                "templates": self._get_seating_templates()
            },
            "blood_relations": {
                "count": 250,
                "templates": self._get_blood_relation_templates()
            },
            "series": {
                "count": 250,
                "templates": self._get_series_templates()
            }
        }
        
        print("✅ Q-Agent ready for logical reasoning!")
    
    def _get_syllogism_templates(self) -> List[str]:
        """Return syllogism question templates"""
        return [
            "All {A} are {B}. All {B} are {C}. Conclusion: All {A} are {C}.",
            "No {A} are {B}. Some {C} are {A}. Conclusion: Some {C} are not {B}.",
            "All {A} are {B}. Some {B} are {C}. Which conclusion follows?",
            "Some {A} are {B}. No {B} are {C}. Which conclusion is valid?",
            "All {A} are {B}. No {C} are {B}. What can we conclude about {A} and {C}?"
        ]
    
    def _get_seating_templates(self) -> List[str]:
        """Return seating arrangement templates"""
        return [
            "linear_6_people",
            "linear_8_people",
            "circular_6_people_clockwise",
            "circular_8_people_facing_center",
            "linear_with_conditions"
        ]
    
    def _get_blood_relation_templates(self) -> List[str]:
        """Return blood relation templates"""
        return [
            "father_sister_son",
            "mother_brother_daughter", 
            "brother_wife_relation",
            "grandfather_grandson",
            "complex_family_tree"
        ]
    
    def _get_series_templates(self) -> List[str]:
        """Return series pattern templates"""
        return [
            "arithmetic_progression",
            "geometric_progression",
            "fibonacci_type",
            "alphanumeric_mixed",
            "alternating_series"
        ]
    
    def generate_syllogism_mcq(self) -> Dict:
        """Generate a syllogism MCQ"""
        
        # Sample entities
        entities = [
            ["cats", "animals", "mammals"],
            ["roses", "flowers", "plants"],
            ["doctors", "professionals", "educated people"],
            ["students", "learners", "young people"],
            ["books", "objects", "readable items"]
        ]
        
        entity_set = random.choice(entities)
        
        prompt = f"""Generate a challenging syllogism multiple choice question.

Use these entities: {entity_set[0]}, {entity_set[1]}, {entity_set[2]}

Create a logical syllogism problem with:
- Two clear premises
- A question about the conclusion
- 4 options where only ONE is logically valid

Requirements:
- Question must test syllogistic reasoning
- Options must be plausible but only one correct
- Follow rules of categorical syllogisms

Format:
Question: [Premises and question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [Letter]

Generate:"""

        return self._generate_with_prompt(prompt, "syllogism")
    
    def generate_seating_mcq(self, seating_type: str = "linear") -> Dict:
        """Generate a seating arrangement MCQ"""
        
        if seating_type == "linear":
            people = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
            setup = "6 people in a straight line"
        else:
            people = ["A", "B", "C", "D", "E", "F"]
            setup = "6 people around a circular table facing center"
        
        prompt = f"""Generate a {seating_type} seating arrangement question.

People: {', '.join(people)}
Setup: {setup}

Create a logical puzzle with:
- 3-4 clear clues about positions
- A specific question about one person's position
- 4 options where only ONE is correct

Requirements:
- All clues must be consistent
- Question must be solvable from given clues
- No ambiguity in the answer

Format:
Question: [Setup and clues, then ask specific question]
A) [Position]
B) [Position]
C) [Position]
D) [Position]
Correct Answer: [Letter]

Generate:"""

        return self._generate_with_prompt(prompt, "seating")
    
    def generate_blood_relation_mcq(self) -> Dict:
        """Generate a blood relation MCQ"""
        
        relations = [
            "father's sister's son",
            "mother's brother's daughter",
            "sister's husband's brother",
            "brother's wife's sister",
            "grandfather's only son"
        ]
        
        relation = random.choice(relations)
        
        prompt = f"""Generate a blood relation reasoning question.

Create a question about: {relation}

Requirements:
- Clear relationship chain
- Test understanding of family relations
- 4 options with only ONE correct relationship

Format:
Question: [Describe relationship scenario and ask]
A) [Relationship]
B) [Relationship]
C) [Relationship]
D) [Relationship]
Correct Answer: [Letter]

Generate:"""

        return self._generate_with_prompt(prompt, "blood_relations")
    
    def generate_series_mcq(self, series_type: str = "numeric") -> Dict:
        """Generate a series/pattern MCQ"""
        
        if series_type == "numeric":
            pattern_type = random.choice([
                "arithmetic progression (+n)",
                "geometric progression (×n)",
                "square numbers",
                "prime numbers",
                "fibonacci-like"
            ])
        else:
            pattern_type = random.choice([
                "alphabetic skip pattern",
                "alphanumeric alternating",
                "position-based"
            ])
        
        prompt = f"""Generate a {pattern_type} series question.

Create a series pattern question with:
- A sequence with one missing term
- Clear pattern (but not too obvious)
- 4 options where only ONE completes the pattern

Requirements:
- Pattern must be logical and consistent
- Missing term should be findable
- Distractors should be plausible

Format:
Question: Find the missing term in: [series with ?]
A) [Number/Letter]
B) [Number/Letter]
C) [Number/Letter]
D) [Number/Letter]
Correct Answer: [Letter]

Generate:"""

        return self._generate_with_prompt(prompt, "series")
    
    def _generate_with_prompt(self, prompt: str, category: str) -> Dict:
        """Generate MCQ from prompt"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Fast generation settings
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1  # Faster than beam search
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response
        mcq = self._parse_mcq(response, category)
        return mcq
    
    def _parse_mcq(self, text: str, category: str) -> Dict:
        """Parse MCQ from text"""
        
        mcq = {
            "category": category,
            "question": "",
            "options": {},
            "correct_answer": "",
            "raw_text": text
        }
        
        # Extract question
        q_match = re.search(r'Question:(.*?)(?=A\))', text, re.DOTALL)
        if q_match:
            mcq["question"] = q_match.group(1).strip()
        
        # Extract options
        for letter in ['A', 'B', 'C', 'D']:
            pattern = rf'{letter}\)(.*?)(?=[BCD]\)|Correct Answer:|$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                mcq["options"][letter] = match.group(1).strip()
        
        # Extract answer
        ans_match = re.search(r'Correct Answer:\s*([ABCD])', text)
        if ans_match:
            mcq["correct_answer"] = ans_match.group(1)
        
        return mcq
    
    def validate_mcq(self, mcq: Dict) -> tuple:
        """Validate MCQ format and content"""
        
        checks = {
            'has_question': bool(mcq['question']),
            'has_4_options': len(mcq['options']) == 4,
            'all_options_filled': all(mcq['options'].get(l) for l in ['A','B','C','D']),
            'has_answer': bool(mcq['correct_answer']),
            'valid_answer': mcq['correct_answer'] in ['A','B','C','D']
        }
        
        if all(checks.values()):
            return True, "Valid"
        
        failed = [k for k, v in checks.items() if not v]
        return False, f"Failed: {failed}"
    
    def generate_batch(self, category: str, count: int) -> List[Dict]:
        """Generate multiple MCQs for a category"""
        
        print(f"\n🔹 Generating {count} {category} questions...")
        valid_questions = []
        attempts = 0
        max_attempts = count * 2  # Allow retries
        
        while len(valid_questions) < count and attempts < max_attempts:
            attempts += 1
            
            # Generate based on category
            if category == "syllogisms":
                mcq = self.generate_syllogism_mcq()
            elif category == "seating":
                mcq = self.generate_seating_mcq(
                    random.choice(["linear", "circular"])
                )
            elif category == "blood_relations":
                mcq = self.generate_blood_relation_mcq()
            elif category == "series":
                mcq = self.generate_series_mcq(
                    random.choice(["numeric", "alphanumeric"])
                )
            else:
                continue
            
            # Validate
            is_valid, msg = self.validate_mcq(mcq)
            
            if is_valid:
                valid_questions.append(mcq)
                if len(valid_questions) % 10 == 0:
                    print(f"  ✓ {len(valid_questions)}/{count} generated")
            else:
                print(f"  ⚠️  Invalid question (attempt {attempts}): {msg}")
        
        success_rate = len(valid_questions) / attempts * 100
        print(f"✅ {category}: {len(valid_questions)} valid questions ({success_rate:.1f}% success rate)")
        
        return valid_questions
    
    def generate_all_1000(self) -> Dict:
        """Generate all 1000 questions"""
        
        print("\n" + "="*60)
        print("GENERATING 1000 LOGICAL REASONING MCQs")
        print("="*60)
        
        all_questions = {}
        
        for category, config in self.topics.items():
            questions = self.generate_batch(category, config['count'])
            all_questions[category] = questions
        
        total = sum(len(q) for q in all_questions.values())
        
        print(f"\n" + "="*60)
        print(f"✅ GENERATION COMPLETE: {total} total questions")
        print("="*60)
        
        return all_questions
    
    def save_questions(self, questions: Dict, filename: str = "questions_1000.json"):
        """Save questions to file"""
        
        with open(filename, 'w') as f:
            json.dump(questions, f, indent=2)
        
        print(f"💾 Saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = LogicalReasoningQAgent()
    
    # Test single question from each category
    print("\n" + "="*60)
    print("TESTING QUESTION GENERATION")
    print("="*60)
    
    print("\n1. Syllogism:")
    q1 = agent.generate_syllogism_mcq()
    print(f"Valid: {agent.validate_mcq(q1)}")
    print(f"Question: {q1['question'][:100]}...")
    
    print("\n2. Seating:")
    q2 = agent.generate_seating_mcq("linear")
    print(f"Valid: {agent.validate_mcq(q2)}")
    print(f"Question: {q2['question'][:100]}...")
    
    print("\n3. Blood Relations:")
    q3 = agent.generate_blood_relation_mcq()
    print(f"Valid: {agent.validate_mcq(q3)}")
    print(f"Question: {q3['question'][:100]}...")
    
    print("\n4. Series:")
    q4 = agent.generate_series_mcq("numeric")
    print(f"Valid: {agent.validate_mcq(q4)}")
    print(f"Question: {q4['question'][:100]}...")
    
    # Uncomment to generate all 1000
    # all_questions = agent.generate_all_1000()
    # agent.save_questions(all_questions)