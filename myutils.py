import torchvision.transforms as T

def save_trainimg(generated_image, epoch):
    transform = T.ToPILImage()
    norm = T.Normalize(mean=[0,0,0],std=[1,1,1])
    generated_image = norm(generated_image)
    gimg = transform(generated_image.squeeze(0))
    #lab = transform(label.squeeze(0))
    #inp = transform(img.squeeze(0))
    gimg.save("trainimg/gen"+ str(epoch) +".png")
    #lab.save("trainimg/label"+str(epoch) + ".png")
    #inp.save("trainimg/input"+ str(epoch) +".png")