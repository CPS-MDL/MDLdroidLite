from optmizer.ours_adam import AdamW
from model.CNN import LeNet5_standard
from torch.optim import Adam
from torch.optim import AdamW
from optmizer.ours_adam import AdamW

model = LeNet5_standard()
# opt = AdamW(model.parameters(), lr=0.05)
#
# opt1 = Adam(model.parameters(), lr=0.05)

# opt = AdamW(model.parameters(), lr=0.05)
# para = list(model.parameters())
# for group in para:
#     for p in group['params']:
#         print(p)
# print('test')


reg_lambda=0.01
l2_reg = 0
loss = 0

for W in model.named_parameters():
    if "weight" in W[0]:
        layer_name = W[0].replace(".weight", "")
        l2_reg = l2_reg + W[1].data.norm(2)

loss = loss + l2_reg * reg_lambda
loss.backward()
