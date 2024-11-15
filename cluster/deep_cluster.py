import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, idx
    
class CNN(nn.Module):
    def __init__(self, feature_dim, 
                 num_clusters, pretrained=True, freeze_conv=False):
        super(CNN, self).__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        # use ResNet18 as the backbone model
        resnet18 = models.resnet18(weights=None)
        if pretrained:
            print('Use ResNet18 encoder pretrained on satellite imagery')
            state_dict = torch.load("pretrained_resnet18.pth", weights_only=True)
            state_dict = {k.replace('model.',''): v for k, v in state_dict.items() if 'model.' in k and not k.startswith('model.fc')}
            resnet18.load_state_dict(state_dict, strict=False)
        else:
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # remove the last fc layer
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(resnet18.fc.in_features, self.feature_dim),
            nn.ReLU(inplace=True)
        )
        self.fc_cluster = nn.Linear(self.feature_dim, self.num_clusters)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_features(dataloader, model, num_samples, device):
    """
        Use CNN model to extract high-level features in images
        Every image in `dataloader` will be fed into `model` to extract features
    """
    model.eval()
    print("Start computing features...")
    features = torch.zeros(num_samples, model.feature_dim, device=device)
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            batch_features = model(images.to(device))
            features[i * dataloader.batch_size: min((i + 1) * dataloader.batch_size, num_samples)] = batch_features.cpu()
            # 索引计算：i * dataloader.batch_size：计算当前批次在整个数据集中的起始索引。i 是当前的批次索引，dataloader.batch_size 是每个批次的大小。min((i + 1) * dataloader.batch_size, num_samples)：计算当前批次在整个数据集中的结束索引。(i + 1) * dataloader.batch_size 是下一个批次的起始索引，但不能超过数据集的总样本数 num_samples。
            # 切片赋值：features[...]：这是一个切片操作，表示将 features 数组中从起始索引到结束索引的部分进行赋值。batch_features.cpu()：将当前批次的特征数据从GPU内存转移到CPU内存。.cpu() 方法将张量从GPU设备转移到CPU设备。
            # 赋值操作：将 batch_features.cpu() 的结果赋值给 features 数组的相应部分。
    print("Finish computing features!")
    return features.cpu().numpy()

def optimize_cluster_number(features, k_range):
    """
    Find optimal cluster number by silhouette score
    Args:
        features: feature matrix
        k_range: range of cluster number
    Returns:
        optimal_k: the optimal cluster number
        silhouette_scores: silhouette scores for each cluster number
    """
    silhouette_scores = []
    
    print("Start optimizing cluster number...")
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init='auto')
        cluster_labels = kmeans.fit_predict(features)
        
        score = silhouette_score(features, cluster_labels)
        silhouette_scores.append(score)
        print(f"silhouette score for k={k}: {score}")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Cluster_nums (k)')
    plt.xticks(k_range)
    plt.ylabel('silhouette score')
    plt.title('Inspecting the optimal cluster number')
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    return optimal_k, silhouette_scores

def train_deep_cluster(dataset, clusterloader, trainloader, criterion,
                       num_clusters=20, num_epochs=10, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(feature_dim=512, num_clusters=num_clusters, pretrained=True, freeze_conv=False).to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)                                            
    losses = []
    print("Start training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        features = compute_features(clusterloader, model, len(dataset), device)     # features for all images
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(features)           # cluster into 50 classes
        cluster_labels = kmeans.labels_                                             # get clustered pseudo labels for all images in dataset (i.e. clusterloader)

        fc_cluster = nn.Linear(model.feature_dim, num_clusters).to(device)          # model.feature_dim = 512
        fc_cluster.weight.data.normal_(0, 0.01)
        fc_cluster.bias.data.zero_()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        model.train()
        running_loss = 0.0
        
        train_iter = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, (images, indices) in train_iter:
            images, batch_labels = images.to(device), torch.tensor(cluster_labels[indices], dtype=torch.long).to(device)
            features = model(images)                          # features for current batch, [batchsize, feature_dim]
            # outputs = model.fc_cluster(features)            # cluster labels for current batch, [batchsize, num_clusters]
            outputs = fc_cluster(features)
            loss = criterion(outputs, batch_labels)       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        print(f"Loss: {epoch_loss:.4f}")
        losses.append(epoch_loss)
        
    print("Finish training!")
    torch.save(model.state_dict(), "deep_cluster.pth")
    
if __name__ == "__main__":
    set_seed(SEED)

    root_dir = "../data/classified_images/Mountain"
    cluster_dir = "cluster_results"
    if os.path.exists(cluster_dir):
        shutil.rmtree(cluster_dir)
    os.makedirs(cluster_dir)
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),                
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            #transforms.RandomGrayscale(p=0.1),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    batch_size = 24
    num_epochs = 30
    num_clusters = 20
    criterion = nn.CrossEntropyLoss()

    print("Loading datasets...")
    dataset = SatelliteDataset(root_dir, transform=transform)
    clusterloader = DataLoader(dataset, batch_size=24, shuffle=False)
    trainloader = DataLoader(dataset, batch_size=24, shuffle=True)
    print("Finished loading datasets")

    # if the model doesn't exists, starting training
    if not os.path.exists("deep_cluster.pth"):
        train_deep_cluster(dataset, clusterloader, trainloader, criterion, 
                           num_clusters=num_clusters, num_epochs=num_epochs)
    
    # load the model
    print("Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(feature_dim=512, num_clusters=20, pretrained=False, freeze_conv=False).to(device)
    model.load_state_dict(torch.load("deep_cluster.pth", weights_only=True))

    # extract features
    features = compute_features(clusterloader, model, len(dataset), device)
    
    # check optimal cluster number
    k_range = range(3, 21)
    optimal_k, silhouette_scores = optimize_cluster_number(features, k_range)
    print(f"optimal cluster number: {optimal_k}")

    # cluster all images into optimal cluster number
    kmeans = KMeans(n_clusters=optimal_k, n_init='auto').fit(features)
    final_cluster_labels = kmeans.labels_

    # save images to corresponding cluster folders
    for i in range(optimal_k):
        cluster_subdir = os.path.join(cluster_dir, f"cluster_{i}")
        os.makedirs(cluster_subdir)

    for idx, label in enumerate(final_cluster_labels):
        img_name = dataset.images[idx]
        src_path = os.path.join(root_dir, img_name)
        dst_path = os.path.join(cluster_dir, f"cluster_{label}", img_name)
        shutil.copy2(src_path, dst_path)

    print(f"cluster results saved in {cluster_dir}")
