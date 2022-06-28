import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------------------------------------------------------- #

def test(model, criterion, device, test_loader, verbose=True):
    '''
    This function tests the models resnet18 and pooling_resnet18

    '''
    model.eval()

    with torch.no_grad():

        test_loss = 0
        number_samples = 0
        
        auc_sum = []
        target_list = []
        output_list = []

        # Loop over batches 
        for  i, (data, target) in enumerate(test_loader):
          
            data, target = data.to(device), target.to(device)

            # Apply model 
            output = model(data)
            
            # Compute loss function
            loss = criterion(output, target)
            test_loss += loss.item()
            number_samples += len(target)

            # Compute output probabilities 
            m = nn.Softmax(dim=1)
            predicted_batch = m(output)

            # Compute AUC 
            auc_sum.append(roc_auc_score(target.cpu().detach().numpy(),predicted_batch.cpu().detach().numpy(), multi_class='ovo', labels=[0, 1, 2, 3, 4, 5], average='macro'))  

            target_list += list(target.cpu().detach().numpy())
            output_list+= list(np.argmax(predicted_batch.cpu().detach().numpy(),axis=1))
            
        final_loss = test_loss/len(test_loader)
        final_auc = np.mean(auc_sum)

        # Compute balanced accuracy 
        final_acc = balanced_accuracy_score(target_list,output_list)
        
        # Compute confusion matrix
        cmatrix = confusion_matrix(np.array(target_list), np.array(output_list))
        
    # Show results
    if verbose :
        # print('Test Results :  Loss : {} - AUC  : {} - Accuracy : {} '.format(final_loss,final_auc, final_acc))
        disp = ConfusionMatrixDisplay(cmatrix,display_labels=[0,1,2,3,4,5])
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title('Confusion Matrix', fontsize=18)
        disp.plot(cmap='Blues', ax = ax)
        plt.grid(False)
        
    return final_auc, final_acc

# -------------------------------------------------------------------------------------------------- #

def test_MIL(model, criterion, device, test_loader, verbose=True):
    '''
    This function tests the MIL model

    '''

    model.eval()
    
    with torch.no_grad():

      target_list = []
      output_list = []
      test_loss = 0

      # Loop over batches 
      for _ , (data, target) in enumerate(test_loader):

          data, target = data.to(device), target.to(device)

          # Apply model 
          output = model(data) 

          # Compute loss function
          loss = criterion(output, target) #.ordinal_regression(output, target)
          test_loss += loss.item()
          target_list.append(target.item())

          # Compute output probabilities 
          m = nn.Softmax(dim=1)
          output = m(output)
          output_list.append(output.data.cpu().numpy()[0])

      # Compute AUC 
      test_auc = roc_auc_score(target_list, output_list, multi_class='ovo', labels=[0, 1, 2, 3, 4, 5], average='macro')

      # Compute balanced accuracy 
      test_acc = balanced_accuracy_score(target_list, np.argmax(np.array(output_list), axis=1))
      average_test_loss = test_loss/len(test_loader)
      
      # Compute confusion matrix 
      cmatrix = confusion_matrix(np.array(target_list), np.argmax(np.array(output_list), axis=1))
      
      # Show results
      if verbose :
        # print('Test Results :  Loss : {} - AUC  : {} - Accuracy : {} '.format(average_test_loss,test_auc, test_acc))
        disp = ConfusionMatrixDisplay(cmatrix,display_labels=[0,1,2,3,4,5])
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_title('Confusion Matrix', fontsize=18)
        disp.plot(cmap='Blues', ax = ax)
        plt.grid(False)

    return  test_auc, test_acc