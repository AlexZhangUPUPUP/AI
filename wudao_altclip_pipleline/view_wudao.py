import json,random,os,jieba
dir = "/mnt/datasets/multimodal/wudao/json/"
filenames=os.listdir(dir)
#filepth="/mnt/datasets/multimodal/wudao/Cars.json"
for filename in filenames:
    with open(dir+filename,'r',encoding='utf-8') as f:
        J=json.load(f)    
        print(len(J),J[0])
    # n=10 #number of sample
    # idx=random.sample(range(len(J)),n)
    # print(idx)
    # J=[J[i] for i in idx]
    # img_pths = [j["url"] for j in J]#name
    #img_pths = ["/mnt/datasets/multimodal/wudao/Social_contact/"+j["name"] for j in J]
    texts=[j["captions"] for j in J]
    for i,t in zip(img_pths,texts):
        print(i,t)