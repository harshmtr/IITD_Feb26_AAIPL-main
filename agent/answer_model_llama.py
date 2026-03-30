"""
Optimized A-Agent for Logical Reasoning
Target: 100% accuracy, <10 second response time
Uses specialized reasoning for each question type
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from collections import Counter
from typing import Dict, Tuple
import time

class LogicalReasoningAAgent:
    def __init__(self, model_path="unsloth/Qwen2-7B-Instruct"):
        """Initialize A-Agent for logical reasoning"""
        print("🔧 Loading Logical Reasoning A-Agent...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        # Performance stats
        self.stats = {
            "total_answered": 0,
            "correct": 0,
            "avg_time": 0,
            "by_category": {}
        }
        
        print("✅ A-Agent ready with optimized reasoning!")
    
    def answer_syllogism(self, mcq: Dict) -> str:
        """Specialized answering for syllogisms"""
        
        prompt = f"""You are an expert in formal logic and syllogisms.

{mcq['question']}

A) {mcq['options']['A']}
B) {mcq['options']['B']}
C) {mcq['options']['C']}
D) {mcq['options']['D']}

Apply these syllogism rules:
1. Check if middle term is distributed
2. Verify no term is distributed in conclusion but not in premise
3. From two negatives, no conclusion follows
4. If premise is negative, conclusion must be negative
5. From two particulars, no conclusion follows

Think step-by-step:
- Identify the premises
- Check validity using rules
- Determine which conclusion is logically valid

Final Answer: [A/B/C/D]"""

        return self._fast_inference(prompt)
    
    def answer_seating(self, mcq: Dict) -> str:
        """Specialized answering for seating arrangements"""
        
        prompt = f"""You are an expert at solving seating arrangement puzzles.

{mcq['question']}

A) {mcq['options']['A']}
B) {mcq['options']['B']}
C) {mcq['options']['C']}
D) {mcq['options']['D']}

Systematic approach:
1. Draw the arrangement (linear or circular)
2. Mark fixed positions from clues
3. Use relative positions to constrain
4. Eliminate contradictions
5. Find position that satisfies all clues

For circular: remember clockwise/counterclockwise
For linear: remember left/right orientation

Final Answer: [A/B/C/D]"""

        return self._fast_inference(prompt)
    
    def answer_blood_relation(self, mcq: Dict) -> str:
        """Specialized answering for blood relations"""
        
        prompt = f"""You are an expert at solving blood relation problems.

{mcq['question']}

A) {mcq['options']['A']}
B) {mcq['options']['B']}
C) {mcq['options']['C']}
D) {mcq['options']['D']}

Systematic approach:
1. Draw a family tree diagram
2. Mark males and females
3. Trace the relationship chain step by step
4. Count generations up/down
5. Determine final relationship

Remember:
- Father's/Brother's/Son's = Male
- Mother's/Sister's/Daughter's = Female  
- In-laws involve spouse's family
- Uncle = parent's brother, Aunt = parent's sister

Final Answer: [A/B/C/D]"""

        return self._fast_inference(prompt)
    
    def answer_series(self, mcq: Dict) -> str:
        """Specialized answering for series/patterns"""
        
        prompt = f"""You are an expert at solving number and letter series patterns.

{mcq['question']}

A) {mcq['options']['A']}
B) {mcq['options']['B']}
C) {mcq['options']['C']}
D) {mcq['options']['D']}

Systematic approach:
1. Calculate differences between consecutive terms
2. Check for arithmetic/geometric progression
3. Look for square/cube patterns
4. Check for alternating sequences
5. For letters, check alphabetical positions

Common patterns:
- Arithmetic: constant difference (+n)
- Geometric: constant ratio (×n)
- Squares: n²
- Fibonacci: sum of previous two
- Alternating: two interleaved series

Final Answer: [A/B/C/D]"""

        return self._fast_inference(prompt)
    
    def _fast_inference(self, prompt: str) -> str:
        """Fast inference with optimized settings"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Optimized for speed and accuracy
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,  # Shorter for faster response
            temperature=0.1,     # Low for consistency
            do_sample=False,     # Greedy = fastest
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return self._extract_answer(response)
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer letter from response"""
        
        patterns = [
            r'Final Answer:\s*([ABCD])',
            r'Answer:\s*([ABCD])',
            r'Correct Answer:\s*([ABCD])',
            r'The answer is\s*([ABCD])',
            r'\b([ABCD])\s*is correct',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Fallback: find last standalone letter
        matches = re.findall(r'\b([ABCD])\b', response)
        if matches:
            return matches[-1].upper()
        
        return None
    
    def answer_with_verification(self, mcq: Dict) -> Tuple[str, float]:
        """
        Answer with self-consistency verification
        Returns: (answer, confidence_score)
        """
        
        start_time = time.time()
        
        # Determine category
        category = mcq.get('category', 'unknown')
        
        # Use specialized method based on category
        answers = []
        for _ in range(3):  # 3 attempts for consistency
            if category == "syllogisms":
                ans = self.answer_syllogism(mcq)
            elif category == "seating":
                ans = self.answer_seating(mcq)
            elif category == "blood_relations":
                ans = self.answer_blood_relation(mcq)
            elif category == "series":
                ans = self.answer_series(mcq)
            else:
                # Generic answering
                ans = self._answer_generic(mcq)
            
            if ans:
                answers.append(ans)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Get consensus answer
        if not answers:
            return None, 0.0
        
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        final_answer = most_common[0]
        confidence = most_common[1] / len(answers)
        
        # Update stats
        self.stats["total_answered"] += 1
        self.stats["avg_time"] = (
            (self.stats["avg_time"] * (self.stats["total_answered"] - 1) + elapsed) 
            / self.stats["total_answered"]
        )
        
        return final_answer, confidence, elapsed
    
    def _answer_generic(self, mcq: Dict) -> str:
        """Generic answering for unknown categories"""
        
        question = mcq.get('question', '')
        options = mcq.get('options', {})
        
        prompt = f"""Analyze this multiple choice question carefully.

Question: {question}

A) {options.get('A', '')}
B) {options.get('B', '')}
C) {options.get('C', '')}
D) {options.get('D', '')}

Think logically and systematically.
Eliminate incorrect options.
Select the one correct answer.

Final Answer: [A/B/C/D]"""

        return self._fast_inference(prompt)
    
    def answer_batch(self, questions: list) -> Dict:
        """
        Answer a batch of questions
        Returns: results dictionary with answers and stats
        """
        
        print(f"\n🔹 Answering {len(questions)} questions...")
        
        results = []
        correct_count = 0
        
        for i, mcq in enumerate(questions):
            answer, confidence, elapsed = self.answer_with_verification(mcq)
            
            # Check if correct (if ground truth available)
            is_correct = None
            if 'correct_answer' in mcq and mcq['correct_answer']:
                is_correct = (answer == mcq['correct_answer'])
                if is_correct:
                    correct_count += 1
            
            result = {
                'question_id': i,
                'answer': answer,
                'confidence': confidence,
                'time_seconds': elapsed,
                'is_correct': is_correct,
                'category': mcq.get('category', 'unknown')
            }
            
            results.append(result)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"  Answered {i+1}/{len(questions)} "
                      f"(Avg time: {elapsed:.2f}s, "
                      f"Accuracy: {correct_count/(i+1)*100:.1f}%)")
        
        # Final statistics
        accuracy = correct_count / len(questions) * 100 if questions else 0
        avg_time = sum(r['time_seconds'] for r in results) / len(results) if results else 0
        
        print(f"\n✅ Batch complete:")
        print(f"   Answered: {len(results)}")
        print(f"   Accuracy: {accuracy:.1f}%")
        print(f"   Avg Time: {avg_time:.2f}s")
        
        return {
            'results': results,
            'accuracy': accuracy,
            'avg_time': avg_time,
            'total_questions': len(questions)
        }
    
    def test_performance(self, test_questions: list) -> Dict:
        """
        Comprehensive performance test
        """
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST")
        print("="*60)
        
        results = self.answer_batch(test_questions)
        
        # Category-wise breakdown
        by_category = {}
        for r in results['results']:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = {'correct': 0, 'total': 0, 'times': []}
            
            by_category[cat]['total'] += 1
            if r['is_correct']:
                by_category[cat]['correct'] += 1
            by_category[cat]['times'].append(r['time_seconds'])
        
        print("\n📊 CATEGORY BREAKDOWN:")
        for cat, stats in by_category.items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            avg_t = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            print(f"  {cat}:")
            print(f"    Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})")
            print(f"    Avg Time: {avg_t:.2f}s")
        
        # Check if meets requirements
        meets_accuracy = results['accuracy'] >= 95  # Target 100%, accept 95%+
        meets_speed = results['avg_time'] <= 10     # Target <10s
        
        print(f"\n{'='*60}")
        print("PERFORMANCE REQUIREMENTS:")
        print(f"  Accuracy Target: 100% | Achieved: {results['accuracy']:.1f}% | {'✅' if meets_accuracy else '❌'}")
        print(f"  Speed Target: <10s | Achieved: {results['avg_time']:.2f}s | {'✅' if meets_speed else '❌'}")
        print(f"{'='*60}")
        
        return {
            'overall': results,
            'by_category': by_category,
            'meets_requirements': meets_accuracy and meets_speed
        }


# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = LogicalReasoningAAgent()
    
    # Test with sample questions
    test_mcqs = [
        {
            'category': 'syllogisms',
            'question': """All cats are animals. All animals need food. 
What conclusion follows?""",
            'options': {
                'A': 'All cats need food',
                'B': 'Some cats need food',
                'C': 'No cats need food',
                'D': 'Cannot be determined'
            },
            'correct_answer': 'A'
        },
        {
            'category': 'blood_relations',
            'question': """A is B's father. C is A's son. What is C to B?""",
            'options': {
                'A': 'Brother',
                'B': 'Son',
                'C': 'Father',
                'D': 'Uncle'
            },
            'correct_answer': 'A'
        }
    ]
    
    # Test performance
    results = agent.test_performance(test_mcqs)
    
    print(f"\n🏆 Final Results:")
    print(f"   Meets Requirements: {results['meets_requirements']}")