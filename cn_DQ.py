import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path('/root/user')
DATA_DIR.mkdir(parents=True, exist_ok=True) 

RNK2LBL = {2: 'A', 1: 'B', 0: 'C', -1: 'D', -2: 'E'}
LBL2RNK = {v: str(k) for k, v in RNK2LBL.items()}

def map_blank(txt):
    return ' xxblk ' if not txt else txt


def flatten_json_by_dialog(json_data, is_test=False):
    flattened_by_dlg = []
    for row in json_data:
        trns = row['turns']
        dlg_len = len(trns)
        dlg_txt = f'xxlen {dlg_len} <s> ' + ' </s> </s> '.join(
            (f'xxtrn {i+1} xxsdr {trn["sender"]} ' +
             ' '.join(map(map_blank, trn['utterances'])))
            for i, trn in enumerate(trns)
        ) + ' </s>'

        rcds = None
        if is_test:
            rcds = [{
                'id': row['id'],
                'label_a_e_s': None,
                'text': dlg_txt
            }]
        else:
            rcds = [
                {
                    'id': row['id'],
                    'label_a_e_s': ';'.join((
                        'a' + RNK2LBL[a['quality']['A']],
                        'e' + RNK2LBL[a['quality']['E']],
                        's' + RNK2LBL[a['quality']['S']],
                    )),
                    'text': dlg_txt
                } for a in row['annotations']
            ]
        flattened_by_dlg.extend(rcds)

    return flattened_by_dlg

DS_NAMES = ('train', 'dev', 'test')
DS_LANGS = ('en', 'cn')

for ds_name in DS_NAMES:
    for ds_lang in DS_LANGS:
        input_p = DATA_DIR / f'dch2_{ds_name}_{ds_lang}.json'
        output_p = DATA_DIR / f'dch2_{ds_name}_dq-{ds_lang}.csv'
        is_test = ds_name == 'test'

        json_f = open(input_p)
        json_d = json.load(json_f)
        json_f.close()
        df_by_dlg = pd.DataFrame(flatten_json_by_dialog(json_d, is_test))
        df_by_dlg.to_csv(
            output_p,
            columns=('label_a_e_s', 'text', 'id')
        )

import torch
from transformers import *
from fastai.text.all import *

from blurr.data.all import *
from blurr.modeling.all import *

model_cls = AutoModelForSequenceClassification

pretrained_model_name = "xlm-roberta-base"

config = AutoConfig.from_pretrained(pretrained_model_name)
config.num_labels = 15

hf_arch, hf_config, hf_tokenizer, hf_model = BLURR.get_hf_objects(pretrained_model_name, model_cls=model_cls, config=config)

len(hf_tokenizer)

hf_tokenizer.add_special_tokens({
    'additional_special_tokens': ['xxlen', 'xxtrn', 'xxsdr', 'xxblk']
})
hf_tokenizer.sanitize_special_tokens()
print(hf_tokenizer.all_special_tokens)

len(hf_tokenizer)

hf_model.resize_token_embeddings(len(hf_tokenizer))

lang = 'cn'

trn_df = pd.read_csv(
    DATA_DIR / f'dch2_train_dq-{lang}.csv',
    usecols=('label_a_e_s', 'text')
)
trn_df = trn_df.assign(is_vld=False)
print(trn_df.head())
print(trn_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))
# Development (validation; vld) set
# a.  DataFrame
vld_df = pd.read_csv(
    DATA_DIR / f'dch2_dev_dq-{lang}.csv',
    usecols=('label_a_e_s', 'text', 'id'),
)
vld_df = vld_df.assign(is_vld=True)
print(vld_df.head())
print(vld_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))

tst_df = pd.read_csv(
    DATA_DIR / f'dch2_test_dq-{lang}.csv',
    usecols=('text', 'id')
)
print(tst_df.head())
print(tst_df.text.str.split(' ').str.len().describe(percentiles=[.5, .95]))

print(trn_df.text.iloc[0])
print(' '.join(hf_tokenizer.tokenize(trn_df.text.iloc[0])))

lang = 'cn'
trn_bs = 8 
mdl_id = 'xlm_roberta_base'

blocks = (
    HF_TextBlock(hf_arch, hf_config, hf_tokenizer, hf_model),
    MultiCategoryBlock()
)
dblock = DataBlock(
    blocks=blocks,
    get_x=ColReader('text'),
    get_y=ColReader('label_a_e_s', label_delim=';'),
    splitter=ColSplitter(col='is_vld')
)

# %%timeit -n 1 -r 1 global trn_bs, dls
# b.  DataLoader
# dls : trn_df & vld_df
dls = dblock.dataloaders(trn_df.append(vld_df), bs=trn_bs, val_bs=128)

# torch.save(dls, DATA_DIR / f'dls-dq-{mdl_id}-{lang}-b{trn_bs}.pth')

# dls = torch.load(DATA_DIR / f'dls-dq-{mdl_id}-{lang}-b{trn_bs}.pth')

dls.vocab

# dls.show_batch()

tst_dl = dls.test_dl(tst_df.text.tolist(), bs=128)

# torch.save(tst_dl, DATA_DIR / f'tst_dl-dq-{mdl_id}-{lang}.pth')

# tst_dl = torch.load(DATA_DIR / f'tst_dl-dq-{mdl_id}-{lang}.pth')

mdl = HF_BaseModelWrapper(hf_model)
lrnr = Learner(
    dls,
    mdl,
    opt_func=partial(Adam, decouple_wd=True),
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[
        accuracy_multi,
        APScoreMulti(average='weighted'),
        F1ScoreMulti(average='weighted'),
        JaccardMulti(average='weighted'),
    ],
    cbs=[HF_BaseModelCallback],
    splitter=hf_splitter,
    path=DATA_DIR,
    # wd_bn_bias=True, 
    moms=(0.8, 0.7, 0.8) 
)
lrnr = lrnr.to_fp16()
lrnr.create_opt()
len(lrnr.opt.param_groups)

# lrnr.summary()

lrnr.opt.param_groups

# lrnr.freeze()

lrnr.lr_find(suggest_funcs=(minimum, steep, valley, slide))

lrnr.fit_one_cycle(1, lr_max=2e-6)

cl_name = f'{mdl_id}-b{trn_bs}-cl1_lr2En3'
lrnr.save(f'dialog_cf-{lang}-{cl_name}')

def save_quality_submission_json(
    cl_name,
    lang='cn',
    lrnr=lrnr,
    tst_dl=tst_dl,
    labels=dls.vocab,
    tst_df=tst_df,
    folder=DATA_DIR,
):
    preds, _ = lrnr.get_preds(dl=tst_dl)
    scrs_lst = [dict([*zip(labels, scrs.tolist())]) for scrs in preds]

    ranks_dict_lst = []
    for lbl_scrs_dict in scrs_lst:
        ranks_dict = {'A': dict(), 'E': dict(), 'S': dict()}
        for lbl, scr in lbl_scrs_dict.items():
            ranks_dict[lbl[0].upper()][str(LBL2RNK[lbl[1]])] = scr
        ranks_dict_lst.append(ranks_dict)

    out_df = tst_df.assign(quality=ranks_dict_lst)
    out_df.id = out_df.id.astype(str)
    out_df[['id', 'quality']].to_json(
        DATA_DIR / f'quality_{lang}_submission-{cl_name}.json',
        orient='records',
        # indent=2
    )

# save_quality_submission_json(cl_name, tst_dl=tst_dl, tst_df=tst_df)

# lrnr.lr_find(suggestions=True)

lrnr.fit_one_cycle(1, lr_max=2e-6)

# cl_name += '-cl1_lr2En3'
# lrnr.save(f'dialog_cf-{lang}-{cl_name}')

# save_quality_submission_json(cl_name)

lrnr.fit_one_cycle(1, lr_max=2e-6)

# cl_name += '-cl1_lr2En3'
# lrnr.save(f'dialog_cf-{lang}-{cl_name}')

# save_quality_submission_json(cl_name)

lrnr.fit_one_cycle(1, lr_max=1e-6)

# cl_name += '-cl1_lr1En3'
# lrnr.save(f'dialog_cf-{lang}-{cl_name}')

# save_quality_submission_json(cl_name)

lrnr.fit_one_cycle(1, lr_max=5e-7)

# cl_name += '-cl1_lr5En4'
# lrnr.save(f'dialog_cf-{lang}-{cl_name}')

# save_quality_submission_json(cl_name)

lrnr.fit_one_cycle(1, lr_max=1e-7)

# cl_name += '-cl1_lr1En4'
# lrnr.save(f'dialog_cf-{lang}-{cl_name}')

# save_quality_submission_json(cl_name) 
# cl_name (cycle name) 
save_quality_submission_json(
    cl_name,
    tst_dl=dls.valid,  # https://docs.fast.ai/data.core.html#DataLoaders.valid
    tst_df=vld_df
)

# lrnr.export(fname=f'{cl_name}.pkl')

# lang
