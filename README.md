# Group 2 - Ethnicity Cycle GAN

**Used articles and papers**
1. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
Implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
Paper: https://arxiv.org/pdf/1703.10593.pdf  
![alt text](https://cdn-images-1.medium.com/max/800/1*nKe_kwZoefrELGHh06sbuw.jpeg)
2. Article having (1.) as basis


**Goal**  
Having Cycle GANs for ethnicity transformation, e.g.  
1. *Black and White*  
![alt text](https://cdn-images-1.medium.com/max/800/1*yFZY_gIOXP5Squmq0TBItA.png)  
 
1. *White and Asian*  
![alt text](https://cdn-images-1.medium.com/max/800/1*3ihWND1xfqTNP_uEgZviYw.png)


**Used Datasets**  
1. CelebA ~ 200.000 images  
![alt text](http://mmlab.ie.cuhk.edu.hk/projects/celeba/intro.png)  
2. UTKFace ~ 20.000 images  
![alt text](http://aicip.eecs.utk.edu/mediawiki/images/thumb/e/ef/LogoFaceWall2.jpg/700px-LogoFaceWall2.jpg) 
3. LFW (Labeled Faces in the Wild) Database  ~ 13.000 images

**Faced Problems**  
1. CelebA is a huge dataset but does not have ethnicity labels unfortunately.  
*Approach: train Classifier on UTKFace and LFW to label CelebA.*  