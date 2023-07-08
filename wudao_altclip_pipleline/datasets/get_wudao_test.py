import os,random,json
# cd /mnt/datasets/multimodal/wudao
imgdir="/mnt/datasets/multimodal/wudao/wudao_test/img3k"
pair_filepth="/mnt/datasets/multimodal/wudao/wudao_test/pairs3k.txt"#每行 img_pth,text \t分隔
test_num=3200#总的测试集数量 会有下载失败的图片 实际会更少

#抽取数据，得到img_urls, texts_origin 两个对齐的list:
filenames=os.listdir("json") 
n=test_num//len(filenames)+2#每个文件采样个数
img_urls=[]
texts_origin=[]
for filename in filenames:
    with open("json/"+filename,"r") as f:
        J=json.load(f)
        idx=random.sample(range(len(J)),n)
        #print(idx)
        J=[J[i] for i in idx]#随机抽取
        img_urls.extend([j["url"] for j in J])#name
        texts_origin.extend([j["captions"] for j in J])
#print(len(texts_origin))

#下载图片
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import multiprocessing as mp
def download_img(data):
    #code from open_clip/src/data/gather_cc.py
    url,dir,filename = data
    try:
        dat = requests.get(url, timeout=20)
        if dat.status_code != 200:
            print("404 file", url)
            return

        # Try to parse this as an Image file, we'll fail out if not
        im = Image.open(BytesIO(dat.content))
        # im.thumbnail((512, 512), PIL.Image.BICUBIC)
        # if min(*im.size) < max(*im.size)/3:
        #     print("Too small", url)
        #     return
        im_pth = os.path.join(dir,"%s.jpg"%(filename))

        im.save(im_pth)

        # Another try/catch just because sometimes saving and re-loading
        # the image is different than loading it once.
        try:
            o = Image.open(im_pth)
            o = np.array(o)

            print("Success", o.shape, filename)
            return im_pth
        except:
            print("Failed", filename)            
    except Exception as e:
        print("Unknown error", e)
        pass

def save_file(img_urls,texts_origin,imgdir,pair_filepth):
    img_pths=[]
    texts=[]
    p = mp.Pool(200)#多线程下载
    results = p.map(download_img,\
        [(url,imgdir,str(i)) for i,url in enumerate(img_urls)])
    #result_pths是按输入url顺序排列的
    img_id=0
    result_pths = []
    for i,img_pth in enumerate(results):
        if not img_pth:#下载失败情况
            continue
        else:
            result_pths.append(img_pth)
            img_pths.append(os.path.join(imgdir,"%s.jpg"%(str(img_id))))#重命名图片路径，因为可能有下载失败导致id不连续
            img_id+=1
            #因为result_pths是与texts_origin对齐的，所以相同的索引i对应了原始图文对
            txt = texts_origin[i].split()[0]#去掉文本中换行及以后内容
            texts.append(txt)
    for src,tgt in zip(result_pths,img_pths):#重命名图片路径
        os.rename(src,tgt)
    print(len(img_pths),len(texts))#2819
    with open(pair_filepth,"w") as f:
        for i,t in zip(img_pths,texts):
            f.writelines(i+'\t'+t+'\n')#保存图片绝对路径

save_file(img_urls,texts_origin,imgdir,pair_filepth)