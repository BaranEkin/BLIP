import json
import time

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from openai import OpenAI
CLIENT = OpenAI()


class LanguageEvaluation:

    @staticmethod
    def evaluate(prediction, gt, tokenizer):
        
        eval = {}

        gt = {"1": [{"caption": gt}]}
        prediction = {"1": [{"caption": prediction}]}

        # =================================================
        # Set up scorers
        # =================================================
        start_time = time.time()
        gt  = tokenizer.tokenize(gt)
        prediction = tokenizer.tokenize(prediction)
        print(f"____Tokenizer took {str(time.time() - start_time)} seconds.____")

        # =================================================
        # Set up scorers
        # =================================================
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            start_time = time.time()
            score, scores = scorer.compute_score(gt, prediction)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval[m] = sc
            else:
                eval[method] = score

            print(f"____{method} took {str(time.time() - start_time)} seconds.____")
        
        return eval

class GPTEvaluation:
    
    @staticmethod
    def evaluate(prediction, gt):
        start_time = time.time()
        gpt_response = CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator who rates my answer based on the correct answer. "
                    "Rate my answer based on the correct answer out of 100, with higher scores indicating "
                    "that the answer is closer to the correct answer, and you should be accurate to single digits "
                    "like 62, 78, 41, etc. Only output the number."
                },
                {
                    "role": "user",
                    "content": f"'This is the correct answer:{gt}, This is my answer:{prediction}'",
                },
            ],
        )
        print(f"____GPT took {str(time.time() - start_time)} seconds.____")
        return gpt_response.choices[0].message.content
    

if __name__ == "__main__":
    g = "The ego vehicle is driving in the rain. It is night time. There are parked cars around."
    preds = {
    "same": "The ego vehicle is driving in the rain. It is night time. There are parked cars around.",
    "same_but_different" :"The ego vehicle is driving in the night among parked cars. It is raining.",
    "looks_same_but_wrong" : "The ego vehicle is driving in the snow. It is day time. There are parked trucks around.",
    "different" : "The ego vehicle is navigating through an intersection. A padestrian is crossing the road.",
    "unrelated" : "A baby is crying, waking her mother in the middle of the night.",
    "nonesense_traffic" : "The the the the ego ego the the the the the it ego the ego the ego the ego ego ego the it.",
    "nonesense_unrelated" : "Random monkey catch worry sad football issue not prestige tomato help is building."
    }

    """gpt = GPTEvaluation()
    for key, val in preds.items():
        print(f"{key}: {gpt.evaluate(val, g)}")"""

    results = {}
    for key, val in preds.items():
        results[key] = LanguageEvaluation.evaluate(val, g).copy()
    
    print(json.dumps(results, indent=4))
    