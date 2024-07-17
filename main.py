import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import time

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import random
import warnings
warnings.filterwarnings("ignore")

random.seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default="configs/plantdoc_clt.yaml")
    parser.add_argument('--n_clt', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="RN101")
    parser.add_argument('--w1', type=float, default=1)
    parser.add_argument('--w2', type=float, default=0.1)
    parser.add_argument('--w3', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--bbeta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)


    args = parser.parse_args()

    return args




def run_MVPDR(cfg, v_prototypes, v_labels, test_features, test_labels, textual_prototypes,
                      clip_model, train_loader_F, weights):
    n_class =v_labels.shape[-1]
    adapter = nn.Linear(v_prototypes.shape[0], v_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(v_prototypes.t())

    prompt_adapter = nn.Linear(textual_prototypes.shape[0], textual_prototypes.shape[1], bias=False).to(clip_model.dtype).cuda()
    prompt_adapter.weight = nn.Parameter(textual_prototypes.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    prompt_optimizer = torch.optim.AdamW(prompt_adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prompt_optimizer, cfg['train_epoch'] * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0

    gamma = cfg['gamma']
    bbeta = cfg['bbeta']
    alpha = cfg['alpha']

    labels = np.unique(list(range(n_class)))
    # initial_gamma = torch.tensor(float(initial_gamma), requires_grad=True).to(device)
    # gamma = nn.Parameter(initial_gamma)
    # gamma_optimizer = torch.optim.AdamW([gamma], lr=cfg['lr']*10, eps=3e-4)
    # gamma_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gamma_optimizer, cfg['train_epoch'] * len(train_loader_F))

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        prompt_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels
            t_logits = 100. * prompt_adapter(image_features)
            t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
            t_mean_logits = t_logits.mean(dim=-1)
            t_max_logits = t_logits.max(dim=-1)[0]
            t_logits = gamma * t_mean_logits + bbeta * t_max_logits

            MVPDR_logits = t_logits + v_logits * alpha
            w1, w2, w3 = weights

            loss1 = F.cross_entropy(v_logits, target)
            # loss2 = F.cross_entropy(t_logits, target)
            loss3 = F.cross_entropy(t_max_logits, target)
            loss4 = F.cross_entropy(t_mean_logits, target)
            loss = w1 * loss1 + w2 * loss3 + w3 * loss4

            acc = cls_acc(MVPDR_logits, target, labels=labels)["acc"]
            correct_samples += acc / 100 * len(MVPDR_logits)
            all_samples += len(MVPDR_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            prompt_optimizer.zero_grad()
            # gamma_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prompt_optimizer.step()
            # gamma_optimizer.step()
            scheduler.step()
            prompt_scheduler.step()
            # gamma_scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()
        prompt_adapter.eval()


        # print(gamma)

        # learned_weights = prompt_adapter.weight.t()

        affinity = adapter(test_features)
        v_logits = ((-1) * (bbeta - bbeta * affinity)).exp() @ v_labels
        t_logits = 100. * prompt_adapter(test_features)
        t_logits = t_logits.reshape(t_logits.shape[0], n_class, -1)
        t_mean_logits = t_logits.mean(dim=-1)
        t_max_logits = t_logits.max(dim=-1)[0]
        t_logits = gamma * t_mean_logits + bbeta * t_max_logits

        MVPDR_logits = t_logits + v_logits * alpha

        result = cls_acc(MVPDR_logits, test_labels, labels=labels)
        acc = result["acc"]
        precision = result["precision"]
        recall = result["recall"]
        f1_score = result["f1"]

        print("**** MVPDR test accuracy: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}. ****\n".format(acc, precision, recall, f1_score))
        if acc > best_acc:
            best_acc = acc
            best_precision = precision
            best_recall = recall
            best_f1 = f1_score
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
            torch.save(prompt_adapter.weight, cfg['cache_dir'] + "/best_prompt.pt")

    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, MVPDR best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


    best_bbeta, best_alpha = bbeta, alpha

    print("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    v_logits = ((-1) * (best_bbeta - best_bbeta * affinity)).exp() @ v_labels

    MVPDR_logits = t_logits + v_logits * best_alpha
    result = cls_acc_test(MVPDR_logits, test_labels, labels=labels)
    acc = result["acc"]
    precision = result["precision"]
    recall = result["recall"]
    f1_score = result["f1"]
    conf_matrix = result["conf_matrix"]

    if acc > best_acc:
        best_acc = acc
        best_precision = precision
        best_recall = recall
    best_f1_score = f1_score


    # print("**** MVPDR's test accuracy: {:.2f}. ****\n".format(acc))
    print("**** MVPDR test accuracy: {:.2f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}. ****\n".format(best_acc, best_precision, best_recall, best_f1_score))

    conf_matrix = np.array(conf_matrix.cpu())

    return best_acc, best_precision, best_recall, best_f1_score, best_epoch

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    start_time = time.time()

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['backbone'] = args.backbone
    cfg['init_alpha'] = args.alpha

    w1 = args.w1
    w2 = args.w2
    w3 = args.w3
    cfg["weights"] = [w1, w2, w3]

    cfg["alpha"] = args.alpha
    cfg["bbeta"] = args.bbeta
    cfg["gamma"] = args.gamma
    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    preprocess_size = preprocess.__dict__['transforms'][0].size

    # Prepare dataset

    set_random_seed(args.seed)

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    # dataset_F = build_dataset(cfg['dataset'], cfg['root_path'], 16)

    val_loader = build_data_loader(data_source=dataset.val, batch_size=32, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=32, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=preprocess_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=32, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=32, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    path1 = "gpt_files_plt44/plantwild_prompts_50_18.json"
    path2 = "gpt_files_plt44/plantdoc_prompts_50_25.json"
    path3 = "gpt_files_plt44/plantvillage_prompts_50_25.json"
    if cfg['dataset'] == "plantwild":
        textual_prototypes = build_textual_prototypes(dataset.classnames, dataset.template, clip_model, path1)
    elif cfg['dataset'] == "plantdoc":
        textual_prototypes = build_textual_prototypes(dataset.classnames, dataset.template, clip_model, path2)
    elif cfg['dataset'] == "plantvillage":
        textual_prototypes = build_textual_prototypes(dataset.origin_classes, dataset.template, clip_model, path3)
    else:
        textual_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing visual prototypes.")
    v_prototypes, v_labels = build_visual_prototypes(cfg, clip_model, train_loader_cache, len(dataset.classnames), n_clt=args.n_clt)


    print("\nLoading visual features and labels from test set.")



    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)


    best_acc, best_precision, best_recall, best_f1_score, best_epoch = run_MVPDR(cfg, v_prototypes, v_labels, test_features, test_labels,
                                         textual_prototypes, clip_model, train_loader_F, cfg["weights"])

    end_time = time.time()
    elapsed_time = end_time - start_time

    # output_path = f"output_mc/{cfg['dataset']}/{(args.backbone).replace('/', '')}/n_clt_{args.n_clt}"
    # w1, w2, w3 = cfg["weights"]
    output_path = f"outputs/{cfg['dataset']}/{args.backbone}/seed_{args.seed}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, f"{w1}_{w2}_{w3}_{cfg['alpha']}_{cfg['bbeta']}_{cfg['gamma']}.txt")

    with open(output_file, "w") as f:
        f.write("**** MVPDR test accuracy: {:.2f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}. ****\n".format(best_acc, best_precision, best_recall, best_f1_score))
        f.write(f"Best epoch: {best_epoch}/{cfg['train_epoch']}\n")
        f.write(f"Time used: {elapsed_time}")


if __name__ == '__main__':
    main()

