from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from openai import OpenAI
CLIENT = OpenAI()


class LanguageEvaluation:
    def __init__(self):
        self.eval = {}

    def evaluate(self, prediction, gt):

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gt  = tokenizer.tokenize(gt)
        prediction = tokenizer.tokenize(prediction)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gt, prediction)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.eval[m] = sc
            else:
                self.eval[method] = score

        return self.eval

class GPTEvaluation:
    def evaluate(self, prediction, gt):
        
        """gpt_response = CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
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
        )"""
        gpt_response = CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an evaluator who rates a given answer based on the correct answer. Your task is to "
                    "rate the given answer out of 100, with higher scores indicating that the answer is semantically close to "
                    "the correct answer. Meaning of the answer and provided information within the answer are utterly important while "
                    "similarity of words or similarity of sentence structures are less important. Evaluate 100 if meaning is same and provided information totally matches."
                    "Output 0 if the answer is totally unrelated or meaningless. Output your evaluation score "
                    "accurate to single digits like 62, 78, 41 etc. and state your reasoning in a single sentence."
                },
                {
                    "role": "user",
                    "content": f"'This is the correct answer:{gt}, This is my answer:{prediction}'",
                },
            ],
        )
        return gpt_response.choices[0].message.content
    

if __name__ == "__main__":
    g = "The ego vehicle is driving in the rain. It is night time. There are parked cars around."
    preds = {
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

    lang = LanguageEvaluation()
    for key, val in preds.items():
        data_val = {
            "1": [
                {"caption": val}
            ]
        }
        data_g = {
            "1": [
                {"caption": g}
            ]
        }

        score = lang.evaluate(data_val, data_g)
        print(f"{key}: {score}")
    