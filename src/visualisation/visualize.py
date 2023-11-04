# -*- coding: utf-8 -*-
from detoxify import Detoxify

# each model takes in either a string or a list of strings

results = Detoxify('original').predict('You have to kill her.')
import numpy as np
from tqdm import tqdm

def test(generated_text):
    toxicity = list()
    for t in tqdm(generated_text):
        toxicity.append(Detoxify('original').predict(t)["toxicity"])

    return np.array(toxicity, dtype='float32')

test_answers_path = "/data/interim/test_answers.txt"

with open(test_answers_path) as f:
    test_answers = f.read().splitlines()

toxic_values = test(test_answers)

from scipy import stats

mea = np.mean(toxic_values)
med = np.median(toxic_values)
mx = np.max(toxic_values)
mi = np.min(toxic_values)
mod = stats.mode(toxic_values)[0]

print(f"Mean: \t{mea:.5f}")
print(f"Median: {med:.5f}")
print(f"Mode: \t{mod:.5f}")
print(f"Worst:  {mx:.5f}")
print(f"Best: \t{mi:.5f}")

import seaborn as sns

#create histogram with density curve overlaid
sns.displot(toxic_values, kde=True, bins=30)

"""## Sacrebleu"""
import pandas as pd
reference = pd.read_csv('/content/test.csv')

from datasets import load_dataset, load_metric

metric = load_metric("sacrebleu")

fake_preds = ["hello there", "general kenobi", "Can I get an A"]
fake_labels = [["hello there"], ["general kenobi"], ['Can I get a C']]
a = metric.compute(predictions=fake_preds, references=fake_labels)

reference_list = []
for i in range(len(test_answers)):
    reference_list.append([reference["reference"][i]])

score = metric.compute(predictions=test_answers, references=reference_list)

"""## Semantics similarity"""

from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction

clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction

web_model.predict([("Think about that shit, dawg.","think about it, man.")])

scores = []
for i in tqdm(range(len(test_answers))):
    scores.append(web_model.predict([(test_answers[i], reference["reference"][i])])[0])

mea = np.mean(scores)
med = np.median(scores)
mx = np.max(scores)
mi = np.min(scores)
mod = stats.mode(scores)[0]

print(f"Mean: \t{mea:.5f}")
print(f"Median: {med:.5f}")
print(f"Mode: \t{mod:.5f}")
print(f"Best:  {mx:.5f}")
print(f"Worst: \t{mi:.5f}")

import seaborn as sns
sns.displot(scores, kde=True, bins=50)

indexes_best = [i for i, num in enumerate(scores) if num > 4]
print(indexes_best)

for i in indexes_best:
    print(f"Original: \t{reference['reference'][i]}")
    print(f"Translated: \t{test_answers[i]}")
    print("-------------------------------------------")

indexes_worst = [i for i, num in enumerate(scores) if num < 1.5]
print(indexes_worst)

for i in indexes_worst:
    print(f"Original: \t{reference['reference'][i]}")
    print(f"Translated: \t{test_answers[i]}")
    print("-------------------------------------------")

