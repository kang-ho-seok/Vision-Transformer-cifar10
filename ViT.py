#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import copy
import math
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
import torch.nn.utils as utils
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from transformers import ViTImageProcessor, ViTModel

device = "cuda:1" if torch.cuda.is_available() else "cpu"


# In[2]:


#(b c h w)-> (b n (p^2 c)) 원래는 reshape->linear인디 cnn->reshape이 더 좋음.. ep(1)적용
#p=patch, n=hw/p^2->  patch 갯수
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, img_size = 224):
        super(PatchEmbedding, self).__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')# h = h/p, w = w/p, (h w) = hw/p^2 = n
          )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))#EP(1)#cls정보
        self.position = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))#(1,N+1,E), position embed

    def forward(self, x):
        B = x.size(0)#(b c h w)
        x = self.projection(x)#(b n e)... flatten
        x = x * math.sqrt(self.embed_dim)#scaling
        n = x.size(1)#(n)
        cls_token = repeat(self.cls_token, '1 n e -> b n e', b=B)
        #learnable 1D position embedding
        x = torch.cat([cls_token, x], dim=1)#[xE ... xE]

        # 2D Interpolation
        if x.shape[1] != self.position.shape[1]:
            cls_pos_embed = self.position[:, 0].unsqueeze(1) # [1, 1, 768]
            patch_pos_embed = self.position[:, 1:]
            P_pre = int(patch_pos_embed.shape[1] ** 0.5)
            patch_pos_embed = patch_pos_embed.reshape(1, P_pre, P_pre, -1).permute(0, 3, 1, 2)
            N_new = int((x.shape[1] - 1) ** 0.5)
            resized_pos_embed = F.interpolate(
            patch_pos_embed, 
            size=(N_new, N_new), # (24, 24)
            mode='bicubic', 
            align_corners=False # ViT 논문에서 권장
            )
            resized_pos_embed = resized_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            final_pos_embed = torch.cat((cls_pos_embed, resized_pos_embed), dim=1)
        else:
            x += self.position[:, :(n + 1)]# + Epos
        return x


# In[13]:


#이거 shape 작성...
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dr_rate=0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = nn.Linear(d_model, d_model)
        self.k_fc = copy.deepcopy(self.q_fc)
        self.v_fc = copy.deepcopy(self.q_fc)
        self.out_fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dr_rate)

    def CalculateAtteantion(self, Q, K, V, mask):
        d_k = Q.size(-1)#q, k, v = (b_n, h, n, d/h)
        attention_score = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k)#(b_n, h, n, n)
        if mask is not None:#self.supervised
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, V)#(b_n, h, n, d/h)
        return out

    def forward(self, q, k, v, mask=None):
        
        n_batch = q.size(0)#(n, seq_len, d_embed)
        
        def transform(x, fc):
            out = fc(x)#(d_embed, d_model)
            #임베딩 차원을 head수로 나눔 : (n_batch, seq_len, d_model) -> (n, seq_len, h, dk)
            #head수 만큼 병렬 계산
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1, 2)
            return out
        q, k, v = transform(q, self.q_fc), transform(k, self.k_fc), transform(v, self.v_fc)#(b_n, h, n, d/h)
        out = self.CalculateAtteantion(q, k, v, mask)#(b_n, h, n, d/h)
        
        out.transpose(1, 2)#(b_n, n, h, d/h)
        out = rearrange(out, 'b h n d -> b n (h d)')#(b_n, n, d)
        out = self.out_fc(out)#(b_n, n, d)
        out = self.dropout(out)
        return out


# In[14]:


class MLP(nn.Module):
    def __init__(self, dim, h_dim, dr_rate=0):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(dim, h_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dr_rate)
        self.mlp2 = nn.Linear(h_dim, dim)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.mlp2(out)
        return out


# In[15]:


import pdb
class ResidualConnection(nn.Module):
    def __init__(self, dim):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        out = x
        out = self.norm(out)#Ep. 2, 3
        out = sublayer(out)
        return x + out

class EncoderBlock(nn.Module):
    def __init__(self, dim, h_dim, heads, dr_rate, n_layer):
        super(EncoderBlock, self).__init__()
        self.MHA = MultiHeadAttention(dim, h= heads, dr_rate = dr_rate)
        self.MLP = MLP(dim, h_dim, dr_rate = dr_rate)
        self.residual_connection1 = ResidualConnection(dim)
        self.residual_connection2 = ResidualConnection(dim)

    def forward(self, x):
#         pdb.set_trace()
        out =self.residual_connection1(x, lambda x: self.MHA(x, x, x))
        out = self.residual_connection2(out, self.MLP)
        return out
    
class Encoder(nn.Module):
    def __init__(self, dim, h_dim, heads, dr_rate, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        self.layers = nn.ModuleList([
        EncoderBlock(dim, h_dim, heads, dr_rate, n_layer) for _ in range(n_layer)
    ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
#             pdb.set_trace()
            out = layer(out)
        out = self.norm(out)
        return out


# In[16]:


class MLPHead(nn.Sequential):
    def __init__(self, dim, n_class):
        super(MLPHead, self).__init__()
        self.cls = nn.Linear(dim, n_class)
        # o으로 초기화
        nn.init.zeros_(self.cls.weight) 
        if self.cls.bias is not None:
            nn.init.zeros_(self.cls.bias)

    def forward(self, x):
        x = self.cls(x)
        return x


# In[17]:


class VIT(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dr_rate, h_dim, n_layer, n_class, pool='cls'):
        super(VIT, self).__init__()
        self.encoder = Encoder(dim, h_dim, heads= heads, dr_rate = dr_rate, n_layer=n_layer)
        self.classification = MLPHead(dim, n_class)
        self.pool = pool

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        nn.Identity()
        out = self.classification(x)
        return out


# In[18]:


#build VIT transformer...
class build_ViT(nn.Module):
    def __init__(self, img_size, in_channels=3, patch_size=16, d_embed=768,
                n_layer=12, d_model=768, h=12, d_ff=4, dr_rate = 0.1, norm_eps = 1e-5):
        super().__init__()
        self.patchembedding = PatchEmbedding(
                                         in_channels = in_channels,
                                         patch_size = patch_size,
                                         embed_dim = d_embed,
                                         img_size = img_size)

        self.dropout = nn.Dropout(dr_rate)

        self.model = VIT(
                            dim=d_embed, heads=h, mlp_dim=d_embed, dr_rate=dr_rate,
                            h_dim=d_embed*d_ff, n_layer=n_layer, n_class=1000).to(device)

    def forward(self, x):
        x = self.patchembedding(x)
        x = self.dropout(x)
        x = self.model(x)
        return x


# In[19]:


def train_model(model, train_dataloader, val_dataloader, criterion, scheduler, optimizer, num_epochs) :
    model.to(device)
    torch.backends.cudnn.benchmark = True
    MAX_NORM = 1.0
    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        # ====== 학습(Training) 단계 ======
        model.train() # 모델을 학습 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward()
            #Gradient Clipping Global Norm1
            utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_dataloader)
        epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)

        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc.double():.4f}')

        # ====== 검증(Validation) 단계 ======
        model.eval() # 모델을 평가 모드로 설정

        best_val_acc = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item()*inputs.size(0)
                val_running_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_dataloader.dataset)

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc.double():.4f}')
        print('*' * 30)

        train_accuracy_list.append(epoch_train_acc.item())
        train_loss_list.append(epoch_train_loss)
        val_accuracy_list.append(epoch_val_acc.item())
        val_loss_list.append(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f'New best model found at epoch {epoch + 1} with Validation Accuracy: {best_val_acc:.4f}. Saving checkpoint...')
            torch.save(model.state_dict(), 'best_model_checkpoint1.pth')

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list
  
def test_model(net, dataloader, criterion, num_epochs) :
    net.eval()
    net.load_state_dict(torch.load('best_model_checkpoint1.pth'))
    accuracy_list = []
    loss_list = []

    for epoch in range(num_epochs) :
        print(f'Epoch {epoch + 1}/ {num_epochs}')
        print('*' * 30)

        epoch_test_loss = 0.0
        epoch_test_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                epoch_test_loss += loss.item() * inputs.size(0)
                epoch_test_corrects += torch.sum(preds == labels.data)

            epoch_test_loss = epoch_test_loss / len(dataloader.dataset)
            epoch_acc = epoch_test_corrects.double() / len(dataloader.dataset)

            print(f'Loss: {epoch_test_loss:.4f} Acc: {epoch_acc:.4f}')

            accuracy_list.append(epoch_acc.item())
            loss_list.append(epoch_test_loss)
        return accuracy_list, loss_list


# In[29]:


#data cifar-10
# fine-tuning시 배치 크기 512, resolution 384
# lr은 {0.001, 0.003, 0.01, 0.03}에서 grid search해야 됨..
batch_size = 16
num_epochs = 100

data_dir = './data'

transform_train = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomCrop(384, padding=int(384/8)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)

train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True
    )
test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=1, pin_memory=True
    )


# In[21]:


model = build_ViT(img_size=224)
# imagenet 21k로 pre_train된 vit_b
pre_trained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

pre_trained_weights = pre_trained_model.state_dict()
model_state_dict = model.state_dict()
new_state_dict = {}

model_keys = list(model_state_dict.keys())
pre_trained_values = list(pre_trained_weights.values())

# MLP_HEAD제외
length = len(model_keys)-2

if len(model_keys) == len(pre_trained_values):
    for i in range(length):
        k_custom = model_keys[i]
        v_pre = pre_trained_values[i]
        new_state_dict[k_custom] = v_pre

model.load_state_dict(new_state_dict, strict=False)


# In[30]:


#CIFAR10 10,000step cos lr scheular {0.001, 0.003, 0.01, 0.03}
# epoch = 100
#fine-tuning SGD momentum 0.9, 배치 크기 512 사용, 해상도 384, weight decay X

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.00)
scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=10000,
        pct_start=5/100,
        div_factor=25,
        final_div_factor=0.001/1e-6,
        anneal_strategy='cos'
    )

train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list = train_model(model, train_loader, test_loader, criterion, scheduler, optimizer, num_epochs=num_epochs)

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="w")

# 정확도(Accuracy) 그래프
ax1.plot(train_accuracy_list, label="Train Accuracy")
ax1.plot(val_accuracy_list, label="Validation Accuracy")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# 손실(Loss) 그래프
ax2.plot(train_loss_list, label="Train Loss")
ax2.plot(val_loss_list, label="Validation Loss")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('inception_resnet_v2_train_val_loss_plot1.png')
plt.show()
#torchinfo.summary(model, input_size=(1, 3, 224, 224))
test_acc, test_loss = test_model(model, test_loader, criterion, num_epochs=1)

# model = Inception_Resnet_V2(num_classes=1000, in_channels=3)
# print(len(model.state_dict().keys()))


# In[ ]:




