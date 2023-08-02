import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path('/root/user')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def map_blank(txt):
    return ' xxblk ' if not txt else txt

def flatten_json_by_tag(json_data, is_test=False):
    flattened_by_tag = []
    for row in json_data:
        turns = row['turns']
        dlg_len = len(turns)
        if is_test:
            flattened_by_tag.extend({
                'id': row['id'],
                'dlg_len': dlg_len,
                'turn': i + 1,
                'label': None,
                'sender': turn['sender'],
                'raw_text':
                    '<s> ' +
                    ' </s> </s> '.join(map(map_blank, turn['utterances'])) +
                    ' </s>'
            } for i, turn in enumerate(turns))
        else:
            flattened_by_tag.extend({
                'id': row['id'],
                'dlg_len': dlg_len,
                'turn': i + 1,
                'label': tag,
                'sender': turns[i]['sender'],
                'raw_text':
                    '<s> ' +
                    ' </s> </s> '.join(map(map_blank, turns[i]['utterances'])) +
                    ' </s>'
            } for a in row['annotations'] for i, tag in enumerate(a['nugget']))
    return flattened_by_tag

DS_NAMES = ('train', 'dev', 'test')
DS_LANGS = ('en', 'cn')

for ds_name in DS_NAMES:
    for ds_lang in DS_LANGS:
        input_p = DATA_DIR / f'dch2_{ds_name}_{ds_lang}.json'
        output_p = DATA_DIR / f'dch2_{ds_name}_trn-{ds_lang}.csv'
        is_test = ds_name == 'test'

        json_f = open(input_p)
        json_d = json.load(json_f)
        json_f.close()
        df_by_tag = pd.DataFrame(flatten_json_by_tag(json_d, is_test))
        df_by_tag = df_by_tag.assign(text=
            'xxlen ' + df_by_tag.dlg_len.astype(str) +
            ' xxtrn ' + df_by_tag.turn.astype(str) +
            ' xxsdr ' + df_by_tag.sender + ' ' +
            df_by_tag.raw_text
        )
        df_by_tag.to_csv(output_p, columns=('label','text', 'id', 'sender'))

import torch
from transformers import *
from fastai.text.all import *

from blurr.data.all import *
from blurr.modeling.all import *

model_cls = AutoModelForSequenceClassification

pretrained_model_name = "xlm-roberta-base"

config = AutoConfig.from_pretrained(pretrained_model_name)
config.num_labels = 7

hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)

len(hf_tokenizer)

hf_tokenizer.add_special_tokens({
    'additional_special_tokens': ['xxlen', 'xxtrn', 'xxsdr', 'xxblk']
})
hf_tokenizer.sanitize_special_tokens()
print(hf_tokenizer.all_special_tokens)

len(hf_tokenizer)

hf_model.resize_token_embeddings(len(hf_tokenizer))

trn_df = pd.read_csv(DATA_DIR / 'dch2_train_trn-en.csv', usecols=('label', 'text'))
print(trn_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))
trn_df = trn_df.assign(is_vld=False)
# Development (validation; vld) set
# a.  DataFrame
vld_df = pd.read_csv(DATA_DIR / 'dch2_dev_trn-en.csv', usecols=('label', 'text', 'id', 'sender'))
print(vld_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))
vld_df = vld_df.assign(is_vld=True)
tst_df = pd.read_csv(DATA_DIR / 'dch2_test_trn-en.csv', usecols=('text', 'id', 'sender'))
print(tst_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))

print(trn_df.text.iloc[0])
print(' '.join(hf_tokenizer.tokenize(trn_df.text.iloc[0])))

blocks = (
    # HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer),
    HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model),
    CategoryBlock(
        vocab=['CNUG', 'CNUG*', 'CNUG0', 'CNaN', 'HNUG', 'HNUG*', 'HNaN']
    )
)
dblock = DataBlock(
    blocks=blocks,
    get_x=ColReader('text'),
    get_y=ColReader('label'),
    splitter=ColSplitter(col='is_vld')
)

trn_bs = 8

# %%timeit -n 1 -r 1 global trn_bs, dls
# b.  DataLoader
# dls 包括 trn_df 和 vld_df
dls = dblock.dataloaders(trn_df.append(vld_df), bs=trn_bs, val_bs=128)

tst_dl = dls.test_dl(tst_df.text.tolist(), bs=128)

mdl = HF_BaseModelWrapper(hf_model)
lrnr = Learner(
    dls,
    mdl,
    opt_func=partial(Lamb, decouple_wd=True),
    loss_func=LabelSmoothingCrossEntropyFlat(),
    metrics=[
        accuracy,
        partial(top_k_accuracy, k=2),
        F1Score(average='weighted'),
        MatthewsCorrCoef(),
        CohenKappa(weights='linear'),
        Jaccard(average='weighted'),
        # PearsonCorrCoef(),
        # SpearmanCorrCoef(),
    ],
    cbs=[HF_BaseModelCallback],
    splitter=hf_splitter,
    path=DATA_DIR,
)
lrnr = lrnr.to_fp16()
lrnr.create_opt()

lr=6e-4
lrnr.fit_one_cycle(1, lr_max=slice(lr/70, lr))

lang = 'en'
cl_name = f'xlm_roberta_base-b{trn_bs}-cl1_lr6En4'
lrnr.save(f'turn_cf-{lang}-{cl_name}')

def save_nugget_submission_json(
    cl_name,
    lang='en',
    lrnr=lrnr,
    tst_dl=tst_dl,
    labels=dls.vocab,
    tst_df=tst_df,
    folder=DATA_DIR,
):
    preds, _ = lrnr.get_preds(dl=tst_dl)
    scrs_lst = [dict([*zip(labels, scrs.tolist())]) for scrs in preds]

    c_k = {'CNUG', 'CNUG*', 'CNUG0', 'CNaN'}
    h_k = {'HNUG', 'HNUG*', 'HNaN'}
    filtered_scrs = [
        {k: v for k, v in scrs.items() if k in c_k} if sndr == 'customer' else
        {k: v for k, v in scrs.items() if k in h_k}
        for scrs, sndr in zip(scrs_lst, tst_df.sender)
    ]

    out_df = tst_df.assign(nugget=filtered_scrs)
    out_df.id = out_df.id.astype(str)
    out_df.groupby('id')['nugget'].apply(list).reset_index().to_json(
        DATA_DIR / f'nugget_{lang}_submission-{cl_name}.json',
        orient='records',
        # indent=2
    )

lr=6e-4/2
lrnr.fit_one_cycle(1, lr_max=slice(lr/70, lr))

lr=6e-4/4
lrnr.fit_one_cycle(1, lr_max=slice(lr/70, lr))

# save_nugget_submission_json(cl_name)
# cl_name (cycle name) 
save_nugget_submission_json(
    cl_name,
    tst_dl=dls.valid,  # https://docs.fast.ai/data.core.html#DataLoaders.valid
    tst_df=vld_df
)
