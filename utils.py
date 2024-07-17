import gc
import json
from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import clip

def cls_acc(output, target, topk=1, labels=None):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    y_pred = pred.squeeze().tolist()
    y_true = target.tolist()

    precision = metrics.precision_score(y_true, y_pred, average='macro', labels=labels) * 100
    recall = metrics.recall_score(y_true, y_pred, average='macro', labels=labels) * 100
    f1_score = metrics.f1_score(y_true, y_pred, average='macro', labels=labels) * 100
    acc = metrics.accuracy_score(y_true, y_pred) * 100

    result = {"acc":acc, "precision":precision, "recall":recall, "f1":f1_score}


    return result

def cls_acc_test(output, target, topk=1, labels=None):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    y_pred = pred.squeeze().tolist()
    y_true = target.tolist()
    precision = metrics.precision_score(y_true, y_pred, average='macro', labels=labels) * 100
    recall = metrics.recall_score(y_true, y_pred, average='macro', labels=labels) * 100
    f1_score = metrics.f1_score(y_true, y_pred, average='macro', labels=labels) * 100
    acc = metrics.accuracy_score(y_true, y_pred) * 100

    kinds = torch.max(target).item() + 1
    kinds = int(kinds)
    conf_matrix = torch.zeros(kinds, kinds).to(pred.device)
    conf_matrix = confusion_matrix(pred, target, conf_matrix)
    conf_matrix = conf_matrix.cpu()
    result = {"acc": acc, "precision": precision, "recall": recall, "f1": f1_score, "conf_matrix":conf_matrix}
    return result

def cls_acc_test_vit14(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]


    return acc

def confusion_matrix(preds, labels, conf_matrix):
    preds = preds.squeeze()
    for p, t in zip(preds, labels):
        p = int(p.item())
        t = int(t.item())
        conf_matrix[p, t] += 1
    return conf_matrix

def visualization(conf_matrix):
    num_classes = conf_matrix.shape[0]
    labels = list(range(num_classes))
    print("confusion matrix")
    # 绘制混淆矩阵

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontsize='xx-small',
                     color="red" if info > thresh else "green")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(num_classes), labels, size=5)
    plt.xticks(range(num_classes), labels, size=5)  # X轴字体倾斜45°
    plt.show()
    # plt.close()

def build_prompt_cache_model(classnames, template, clip_model, path):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []
        label_list = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for i in range(len(classnames)):
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classnames[i]]
            n_per_kind = len(texts)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)



            labels = [i] * n_per_kind
            label_list += labels

        clip_weights = torch.stack(clip_weights, dim=0).t().cuda()
    return clip_weights

def build_multi_prompt_cache_model(classnames, template, clip_model, path, n_clt=5):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).cuda().t()
    return clip_weights


def build_textual_prototypes(classnames, template, clip_model, path, n_clt=5):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).cuda().t()
    return clip_weights

def build_multi_prompt_cache_model_nprt(classnames, template, clip_model, path, n_prt):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname][:n_prt]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).cuda().t()

    return clip_weights



def build_multi_prompt_cache_model_vit_h14(classnames, template, clip_model, path, tokenizer, device):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            texts = tokenizer(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).half().cuda().t()
    return clip_weights

def build_multi_prompt_cache_model_vit14(classnames, template, clip_model, path, processor, device):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            texts = processor(text=texts, return_tensors="pt", padding=True)['input_ids'].to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.get_text_features(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.cat(clip_weights, dim=0).half().cuda().t()
    return clip_weights

def build_clustering_prompt_cache_model(classnames, template, clip_model, path, n_clt=5):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            im_arr = class_embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clt)
            kmeans.fit(im_arr)
            item = transforms.ToTensor()(kmeans.cluster_centers_)
            item = item.squeeze(dim=0)
            # class_embeddings = class_embeddings.mean(dim=0)
            # class_embeddings /= class_embeddings.norm()
            clip_weights.append(item)

        clip_weights = torch.cat(clip_weights, dim=0).cuda().t()
    return clip_weights.half()

def clip_classifier_gpt(classnames, template, clip_model, path):
    print("using gpt-prompts")
    with torch.no_grad():
        clip_weights = []

        with open(path, "r") as json_file:
            json_data = json.load((json_file))

            cnames = list(json_data.keys())

        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = json_data[classname]
            if len(texts) == 0:
                if classname.endswith("leaf"):
                    texts = [f"A photo of a healthy {classname}."]
                else:
                    texts = [f"A photo of a {classname}, a type of plant disease."]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            # redundant = [" "]
            # tr = clip.tokenize(redundant).cuda()
            # r_embeddings = clip_model.encode_text(tr)
            # r_embeddings /= r_embeddings.norm(dim=-1, keepdim=True)
            # r_embedding = r_embeddings.mean(dim=0)
            # r_embedding = r_embedding.norm()

            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            # class_embedding = class_embedding - r_embedding

            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).half().cuda()
    return clip_weights

def clip_classifier_vit_h14(classnames, template, clip_model, tokenizer, device):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = tokenizer(texts).to(device)

            # redundant = [" "]
            # tr = clip.tokenize(redundant).cuda()
            # r_embeddings = clip_model.encode_text(tr)
            # r_embeddings /= r_embeddings.norm(dim=-1, keepdim=True)
            # r_embedding = r_embeddings.mean(dim=0)
            # r_embedding = r_embedding.norm()

            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            # class_embedding = class_embedding - r_embedding

            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).half().cuda()
    return clip_weights

def clip_classifier_vit14(classnames, template, clip_model, processor, device):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = processor(text=texts, return_tensors="pt", padding=True)['input_ids'].to(device)

            # redundant = [" "]
            # tr = clip.tokenize(redundant).cuda()
            # r_embeddings = clip_model.encode_text(tr)
            # r_embeddings /= r_embeddings.norm(dim=-1, keepdim=True)
            # r_embedding = r_embeddings.mean(dim=0)
            # r_embedding = r_embedding.norm()

            # prompt ensemble for ImageNet
            class_embeddings = clip_model.get_text_features(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            # class_embedding = class_embedding - r_embedding

            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).half().cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()


        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values



def build_visual_prototypes(cfg, clip_model, train_loader_cache, n_cls, n_clt):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []
        feature_dict = {}
        label = list(range(n_cls))

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))


        cache_keys = torch.cat(cache_keys, dim=0)
        cache_keys = cache_keys.mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = torch.cat(cache_values, dim=0)
        # cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        for l in label:
            feature_dict[l] = []
        for i in range(len(cache_values)):
            l = cache_values[i].item()
            feature_dict[l].append(cache_keys[i].unsqueeze(0))

        features_list = []
        label_list = []
        for k in feature_dict.keys():
            tensors = torch.cat(feature_dict[k], dim=0)
            im_arr = tensors.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clt)
            kmeans.fit(im_arr)
            item = transforms.ToTensor()(kmeans.cluster_centers_)
            item = item.squeeze(dim=0)
            features_list.append(item)

            labels = [k] * n_clt
            label_list += labels



        cache_keys = torch.cat(features_list)
        cache_keys = cache_keys.permute(1, 0).half()

        cache_values = torch.Tensor(label_list).to(torch.int64)
        cache_values = F.one_hot(cache_values).half()

        

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(n_clt) + "clts.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(n_clt) + "clts.pt")

    return cache_keys.cuda(), cache_values.cuda()


def build_visual_prototypes_vit14(cfg, clip_model, train_loader_cache, n_cls, n_clt):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []
        feature_dict = {}
        label = list(range(n_cls))

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.get_image_features(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0)
        cache_keys = cache_keys.mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = torch.cat(cache_values, dim=0)
        # cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        for l in label:
            feature_dict[l] = []
        for i in range(len(cache_values)):
            l = cache_values[i].item()
            feature_dict[l].append(cache_keys[i].unsqueeze(0))

        features_list = []
        label_list = []
        for k in feature_dict.keys():
            tensors = torch.cat(feature_dict[k], dim=0)
            im_arr = tensors.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clt)
            kmeans.fit(im_arr)
            item = transforms.ToTensor()(kmeans.cluster_centers_)
            item = item.squeeze(dim=0)
            features_list.append(item)

            labels = [k] * n_clt
            label_list += labels

        cache_keys = torch.cat(features_list)
        cache_keys = cache_keys.permute(1, 0).half()

        cache_values = torch.Tensor(label_list).to(torch.int64)
        cache_values = F.one_hot(cache_values).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(n_clt) + "clts.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(n_clt) + "clts.pt")

    return cache_keys.cuda(), cache_values.cuda()


def build_visual_prototypes_p1(cfg, train_loader_cache, n_cls, n_clt=16):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []
        feature_dict = {}
        label = list(range(n_cls))

        with torch.no_grad():
            # Data augmentation for the cache model
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                B, C, H, W = images.shape
                images_1d = images.reshape(B, -1)
                # image_features = clip_model.encode_image(images)
                target = target.cuda()
                cache_values.append(target)
                cache_keys.append(images_1d)

        cache_keys = torch.cat(cache_keys, dim=0)
        # cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_values = torch.cat(cache_values, dim=0)
        # cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        for l in label:
            feature_dict[l] = []
        for i in range(len(cache_values)):
            l = cache_values[i].item()
            feature_dict[l].append(cache_keys[i].unsqueeze(0))

        del cache_keys
        gc.collect()

        features_list = []
        label_list = []
        for k in tqdm(feature_dict.keys()):
            tensors = torch.cat(feature_dict[k], dim=0)
            im_arr = tensors.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clt)
            kmeans.fit(im_arr)
            element = transforms.ToTensor()(kmeans.cluster_centers_).squeeze()
            #todo: reshape element for features
            element = element.reshape(element.shape[0], C, H, W)
            features_list.append(element.squeeze())

            labels = [k] * n_clt
            label_list += labels

        del kmeans
        del feature_dict
        del tensors
        del im_arr
        del images_1d
        gc.collect()
        return features_list, label_list, C, H, W

def build_visual_prototypes_p2(clip_model, features_list, label_list):
    clip_list = []
    for each_clt in features_list:
        with torch.no_grad():  #!!!!!!!!!!!
            clip_features = clip_model.encode_image(each_clt.cuda())
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
        clip_list.append(clip_features)


    cache_keys = torch.cat(clip_list)
    cache_keys = cache_keys.permute(1, 0).half()

    cache_values = torch.Tensor(label_list).to(torch.int64)
    cache_values = F.one_hot(cache_values).half()

    #     torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
    #     torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(n_clt) + "clts.pt")
    #
    # else:
    #     cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")
    #     cache_values = torch.load(cfg['cache_dir'] + '/keys_' + str(n_clt) + "clts.pt")

    return cache_keys.cuda(), cache_values.cuda()

def load_kmeans_cache_model(cfg, clip_model, train_loader_cache):
    print("Loading cluster centroid features to build cache model:")
    cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(8) + "clts.pt").half().cuda()
    cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(8) + "clts.pt").cuda()

    return cache_keys, cache_values

def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features.half(), labels.half()


def pre_load_features_vit14(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features.half(), labels.half()

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None, beta=None):
    n_class = cache_values.shape[-1]

    if cfg['search_hp'] == True and beta == None:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                clip_logits = clip_logits.reshape(clip_logits.shape[0], n_class, -1)
                clip_logits = clip_logits.mean(dim=-1)

                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

    else:
        best_acc = 0
        best_beta = beta
        best_alpha = 0
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(features)
            else:
                affinity = features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights
            clip_logits = clip_logits.reshape(clip_logits.shape[0], n_class, -1)
            clip_logits = clip_logits.mean(dim=-1)

            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, labels)

            if acc > best_acc:
                # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha


        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_hp_v2(cfg, cache_keys, cache_values, features, labels, prompt_adapter=None, adapter=None, gamma=1):
    n_class = cache_values.shape[-1]

    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * prompt_adapter(features)
                clip_logits = clip_logits.reshape(clip_logits.shape[0], n_class, -1)
                clip_mean_logits = clip_logits.mean(dim=-1)
                clip_max_logits = clip_logits.max(dim=-1)[0]
                clip_logits = (1 - gamma) * clip_mean_logits + gamma * clip_max_logits

                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    # print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def similarity_based_weights(clip_logits, cache_logits, target):
    softmax_logits = torch.nn.Softmax(dim=1)
    oht = torch.zeros(clip_logits.shape[0], clip_logits.shape[1])
    for i in range(len(target)):
        c = target[i].item()
        oht[i, c] = 1
    oht = oht.cuda()

    l1 = softmax_logits(clip_logits)
    s1 = l1 * oht

    l2 = softmax_logits(cache_logits)
    s2 = l2 * oht

    cs1 = torch.sum(s1).item()
    cs2 = torch.sum(s2).item()
    w1 = cs1 / (cs1 + cs2)
    w2 = cs2 / (cs1 + cs2)
    return w1, w2

