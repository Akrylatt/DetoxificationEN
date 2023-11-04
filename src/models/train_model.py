# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

# Doing all the necessary imports
import os
from transformers import (
    AutoModelWithLMHead,
    AutoConfig,
    Trainer,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from transformers.trainer import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer

train_file = "/data/interim/combined.txt"

def modelTrainer(train_file, epochs, model='gpt2', batch_size=8, cache_dir='cache'):
    model = AutoModelWithLMHead.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm = False)
    train_dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = train_file,
        block_size = 256
    )
    training_args = TrainingArguments(
        output_dir = "gpt2_fine_tune",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=100,
        save_steps = 500,
        logging_steps = 200
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
    )

    trainer.train()
    trainer.save_model()

epochs = 3

modelTrainer(train_file, epochs)

model = AutoModelWithLMHead.from_pretrained("/content/gpt2_fine_tune")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
pad_token_id = generator.tokenizer.eos_token_id

def detoxify(input_sentence):
    p = generator('<s>' + input_sentence + '</s>>>>><p>', pad_token_id=generator.tokenizer.eos_token_id)
    return p[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]

test = ['His father would have used a booming voice to wrench a reply from this stupid machine.',
       'You have to send those idiots back in.',
       'Salina could be with that stupid cop.',
       "And don't let those idiots in radiology hold you up.",
       'My idiot friend here brought marijuana... - on the plane.',
       'That stupid couple told her to break up with you.',
       'The idiot, Max. He never should have sold it to you guys.',
       "Fuck you, soft. You're panicking",
       "It's coz of those two idiots. They escaped through a tunnel.",
       'You have to let this stupid plane crash make the decision for you.',
       "Audrey Cruz is an idiot. He's the boss.",
       "Why don't you get a job. Instead of playing golf with your stupid buddies?",
       'How do you like your stupid game now, Tom?',
       'Think about that shit, dawg.',
       'You idiots! You have betrayed the revolution.',
       'Why is this idiot Silencer shouting so much?',
       'Just like the rest of the stupid Irish in this country.',
       'Your shit is so tired, Justice.',
       'More than the stupid senators and congressmen that passed it.',
       'How do you possibly know, you stupid woman?']

answers = list()
for t in test:
    answers.append(detoxify(t))

"""## Generating test output"""

import pandas as pd
df_test = pd.read_csv('/data/interim/test.csv')
from tqdm import tqdm

predictions = list()
answers = []
for t in tqdm(df_test["reference"][:100]):
    answers.append(detoxify(t))

from tqdm import tqdm

xx = 0
for i in tqdm(range(10000000)):
    xx += i

with open(r'test_answers.txt', 'w') as fp:
    for line in answers:
        # write each item on a new line
        fp.write("%s\n" % line)
    print('Done')

import shutil
shutil.make_archive('model', 'zip', '', 'model')