![image](https://github.com/user-attachments/assets/eccd1f61-36a3-460e-9f1a-062e1d88594b)
![image](https://github.com/user-attachments/assets/ea8183ac-db62-43cd-9936-fe18a79b758a)


Making a dataset from SORA, Midjourney, Leonardo AI. Often bugged with asking for subscriptions and limited image generations? 
Want a synthetic dataset to train your model for research? But manually typing in prompts again and again pisses you off?
Synthetic-Dataset-Gen got you. 
It has 2 parts integrated into a single web interface (Open to ideas to scale).
Want GAN/Stable-Diffusion generated images at a time, auto augmented, ready for classification and detection tasks?

**Prompt generation** - This big but computationally optimal guy got you covered. 
                        What does it do uniquely: 
                        Problem : Running the stable diffusion model on colab to use it's T4 GPU. But generation of multiple images eventually tears the resources leading to the GPU dying and kernel 
                                  reconnecting. 
                        Solution: Generates image (takes in positive and negative prompt), Generates a single image and auto augments it, clears the gpu memory, then again generates an image. 
                                  Hence, generating large datasets which can be used for training various classification, detection models.  
                                  Jeez, WHO IS GONNA SIT ON ROBOFLOW AND DRAW BOUNDING BOXES THEN??
                        What do you get : A zip file with the base image and the number of augmented images that you choose.
![image](https://github.com/user-attachments/assets/54ac97b7-e294-47ab-b669-1e3da454c4d9)

**Overlay generation** - Want to place an multiple objects in multiple backgrounds for an eventual object detection task? But concerned about the bounding box generation?
                        Then OVERLAY GENERATION is your go to tool.
                        Features: 1. Select the foreground image category/categories. (Eg: A pine tree, A pokemon)
                                  2. Upload any number of background images you need. 
                                  3. On the click of a button, you get a zip file with the foreground image placed on the background and bounding boxes in JSON format. 
