'''
Author: Bin
Date: 2023-11-23
FilePath: /macbert/demo.py
'''
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("./models/shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("./models/shibing624/macbert4csc-base-chinese")

def ai_text(text):

    with torch.no_grad():
        outputs = model(**tokenizer([text, text], padding=True, return_tensors='pt'))

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    argmax = torch.argmax(outputs.logits[0], dim=-1)

    _text = tokenizer.decode(argmax, skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    corrected_text, details = get_errors(corrected_text, text)
    return corrected_text + ' ' + str(details)

ai_text("我脚的，少先队员因该为老人让坐，而不柿自己坐。为什么你会因为自已新苦而枪老认的座位呢？")