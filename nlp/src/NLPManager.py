from typing import Dict
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import re
import json

class NLPManager:
    def __init__(self):
        output_dir = './nlp_training/nlpModel1'
        self.model = BertForQuestionAnswering.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir)

    def qa(self, context: str) -> Dict[str, str]:
        # Define the questions
        questions = [
            "What is the heading?",
            "What is the target?",
            "What is the tool?"
        ]

        # Initialize variables to store results
        heading = ""
        target = ""
        tool = ""

        # Process each question
        for question in questions:
            # Encode the question and context
            inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt', add_special_tokens=True)

            # Get model's prediction
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            outputs = self.model(input_ids, token_type_ids=token_type_ids)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            # Find the position tokens with the highest scores
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores) + 1

            # Convert tokens to the answer string
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))

            # Process the answer based on the question
            if question == "What is the heading?":
                word_to_digit = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                    'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                    'eight': '8', 'nine': '9', 'niner': '9'
                }
                words = answer.split()
                words = [word for word in words if word in word_to_digit]
                digits = [word_to_digit[word] for word in words]
                heading = ''.join(digits)
            elif question == "What is the target?":
                target = answer
            elif question == "What is the tool?":
                tool = re.sub(r'\s*-\s*', '-', answer)

        return {'target': target, 'heading': heading, 'tool': tool}
