from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

#Now we Should fix the size of our batchs also the size of the generated images(64,64)
BSize = 64, ISize = 64

#create the transformations
TransF = transforms.Compose([transforms.Scale(ISize), transforms.ToTensor(),])###########

#loading the data set
dataset = dset.CIFAR10(root= '/.data', download = True, transform= TransF)
dataLoader =  torch.utils.data.DataLoader(dataset, batch_size= BSize, shuffle= True, num_workers= 2)


# now we gonna define a function that takes as input a NN and initialize all its weights, bcuz we have 2 NN , we did this F
def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# The Generator    
class G(nn.modules): #we should introduce a class that describe our architecture(modules, number of layers ...) of our a generator, Then I will create an instance, Thus for simplecity I will use "Héritage" from torch that contains all "outils nécessaire pour moi déja existants" 
    
    def __init__(self):
        super(G, self).__init__()# pour activer l'héritage OOP python
        # create the meta module that will contain all modules in a sequence of layers
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100,512, 4, 1, 0, bias= False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias= False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias= False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias= False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias= False),
                nn.Tanh()
                )


    #now we add our forward function that will forward the signal into NN
    # the input will be some vector size of 100, ranom input for the generator to generate some noise to generate fake images, this fake image will be the output for the Generator and the input for the descriminator  
    def forward(self,input):
        output = self.main(input)
        return output   
    

    
# next step w'll create the object, and w'll just apply the weights to initialize our NN
NNGenerator = G()
NNGenerator.apply(weights_init)

# now w'll create the Dicriminator
# by experimentations , i should Use the "Leaky RelU" not "ReLu"
class D(nn.modules):

    def __init(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias= False),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Conv2d(64, 128, 4, 2, 1, bias= False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Conv2d(128, 256, 4, 2, 1, bias= False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Conv2d(256, 512, 4, 2, 1, bias= False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace= True),
                nn.Conv2d(512, 1, 4, 1, 0, bias= False),
                nn.Sigmoid()                                             
                )
    def forward(self, input):
         output = self.main(input)
         return output.view(-1)  # we must flatten the result of the output into 1 Dim
     
        
        
#creating the discriminator
NNDiscriminator = D()
NNDiscriminator.apply(weights_init)

#training Part 
#here w'll have 2 big steps 
#                                                     1 first 
#one w'll be to update the weights of the D  
### w'll train the the D to see understand, what's real, what's fake , so w'll train it first by giving it a real
#image, and w'll set the target to 1, and then by fake image and set it to 0, andthis fake image will be generated 
#by tye generator.
#                                                     2 first 
#one w'll be to update the weights of the G 
### w'll take the fake image again , and feed it into the D,to get the output which will be a value between 0 and 1,
# the di scriminating value, but then w'll set the new target always to 1, and will compute the computerless between the 
#output of the discriminator( 0 - 1) and 1(the new target).
# key thing => w'll be back propagate the error  back inside the generator

criterion = nn.BCEloss()
optimizer4D = optim.Adam(NNDiscriminator.params(), lr = 0.0002, betas = (0.5, 0.999))# lr = learning rate
optimizer4G = optim.Adam(NNGenerator.params(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(25):
    
    for i, data in enumerate(dataLoader, 0):# data = w'll breaking the whole data into manu batches
    
        ##############################   1st  Step : Updating the weights of the NN of the Disriminator   ####################
        
        NNDiscriminator.zero_grad()
        
        # training the Discriminator with a real image of the dataset
        
        real, _ = data
        input = Variable(real) # input => the input  of the NN , it is an object of the Variable class , in a torch variable
        
        #specify to the Discriminator that the ground truth is 1, the image is real.
        # since we training the discriminator with a real image of the data set, we gonna create a torch tensor which will have the size of the inpt images and initialize them with 1
        target = Variable(torch.ones(input.size()[0]))
        output = NNDiscriminator(input) # numer between 0 and 1
        errorD_real = criterion(output, target) # compute the los error btw the output and the target
        
        
        
        
        #training the Discriminator with a fake image generated by the generator
        
        noise = Variable(torch.randn(input.size()[0] , 100, 1, 1))   # create a mini batch of random vectors of size 100, inside of having 100 vlues of the vectors, it's like we w'll have 100 features maps of 1*1
        fake = NNGenerator(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = NNDiscriminator(fake.detach())
        errorD_fake = criterion(output, target)
        
        # we can save up memory, remember that the fake is a total variable because the output of a PyTorch NN is also a torch Variable , and therefore it contains not only the tensor of the prediction , but also the gradients 
        # bur w're not going to use this gradient after back propagate the error, back inside the NN and apply the stochastic gradient decsent.
        # so we gonna detach the gradient of this fake torch variable, and w'll save up, and that will speed up the computations
        
        
        
        
        # Backpropagating the total error 
        errD = errorD_real + errorD_fake
        
        # now let's backpropagate it into the NN of the D, + apply stochastic gardien descent
        errD.backward()
        optimizer4D.step()



        ##############################   2st  Step : Updating the weights of the NN of the Generator   ####################
    
        ### initialize the weights of the gradient
        
        NNGenerator.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = NNDiscriminator(fake)  # here w'll not detach , bcz we gonna update the weights of the NNG , and to update them we will actualluy need the gradient of the Fake, so not detach it.
        errorG_fake = criterion(output, target)
        errorG_fake.backward()
        optimizer4G.step()

                         
                             
                            
                            