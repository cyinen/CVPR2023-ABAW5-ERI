import torch
from main_vitmutil_with_classes import parse_args, get_loss_fn, get_eval_fn
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from utils import seed_worker
from model import build_network
import os
from config import config
import numpy as np
import pandas as pd
from config.config import REACTION_LABELS


def write_reaction_predictions(full_metas, full_preds, csv_dir, filename):
    os.makedirs(csv_dir, exist_ok=True)
    meta_arr = np.row_stack(full_metas).squeeze()
    preds_arr = np.row_stack(full_preds)
    pred_df = pd.DataFrame(columns=['File_ID']+REACTION_LABELS)
    pred_df['File_ID'] = meta_arr
    pred_df[REACTION_LABELS] = preds_arr
    pred_df.to_csv(os.path.join(csv_dir, filename), index=False)
    return None

def evaluate(task, model, data_loader, loss_fn, eval_fn, use_gpu=False, predict=False, prediction_path=None, filename=None):
    losses, sizes = 0, 0
    full_preds = []
    full_labels = []
    if predict:
        full_metas = []
    else:
        full_metas = None

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            if predict is not True:
                if torch.any(torch.isnan(labels)):
                    print('No labels available, no evaluation')
                    return np.nan, np.nan, np.nan
            batch_size = features[0].size(0) if task!='stress' else 1
            if use_gpu:
                model.cuda()
                features = [feature.cuda() for feature in features]
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds,_,_,_ = model(features)
            if torch.any(torch.isnan(preds)) and predict is not True:
                print('preds have nan values')
                return np.nan, np.nan, np.nan
            # only relevant for stress
            feature_lens = feature_lens.detach().cpu().tolist()
            cutoff = feature_lens[0] if task=='stress' else batch_size
            if predict:
                full_metas.append(metas.tolist()[:cutoff])

            loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1), feature_lens)

            losses += loss.item() * batch_size
            sizes += batch_size

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist()[:cutoff])
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist()[:cutoff])

        if predict:
            write_reaction_predictions(
                full_metas, full_preds, prediction_path, filename)
            
            score, score_details = eval_fn(full_preds, full_labels)
            total_loss = losses / sizes
            return total_loss, score, score_details
        
def predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # state_dict = torch.load(args.eval_model)
    # state_dict = state_dict = {
    #     k.replace('model.', ''): v for k, v in state_dict['state_dict'].items()}
    # model = build_network(args.model_name, args).cuda()
    # model.load_state_dict(state_dict)
    model = torch.load(args.eval_model)
    state_dict = model.state_dict()
    model = build_network(args.model_name, args).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    data = []
    for name in args.feature:
        data.append(
            load_data(args.task,
                      args.paths,
                      name,
                      args.emo_dim,
                      args.normalize,
                      args.win_len,
                      args.hop_len,
                      save=args.cache,
                      combined=args.combined,
                      halve_val=args.halve_val))
     
    dataset = MuSeDataset(data, 'devel', args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate_fn)
    
    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)
    
    total_loss, score, score_details = evaluate(
                                    args.task, model, 
                                    data_loader, loss_fn, 
                                    eval_fn, use_gpu=True,
                                    predict=args.predict,
                                    prediction_path=args.prediction_path,
                                    filename=args.filename)
    print(f'Mean Person: {score:.4f}')

if __name__ == '__main__':
    args = parse_args()
    args.d_in = [1024,1536]
    args.len_feature = 30
    args.model_name = 'VitModel2'
    args.eval_model = "model_101.pth"
    args.d_model = 256
    args.emo_dim=''
    args.predict = True
    args.prediction_path = 'result'
    args.filename = 'predict.csv'
    
    

    # adjust your paths in config.py
    args.paths = {
        'log': os.path.join(config.LOG_FOLDER, args.task),
        'data': os.path.join(config.DATA_FOLDER, args.task),
    }

    args.paths.update({
        'features': config.PATH_TO_FEATURES[args.task],
        'labels': config.PATH_TO_LABELS[args.task],
        'partition': config.PARTITION_FILES[args.task]
    })
    predict(args)
    
    
# usage
# python predict.py --feature  DeepSpectrum PosterV2+Vit