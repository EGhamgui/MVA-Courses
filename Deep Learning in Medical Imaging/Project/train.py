import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from test import test, test_MIL
from utils import read_data_augmented, read_data_MIL

# -------------------------------------------------------------------------------------------------- #

def train(model, epochs,optimizer, criterion, train_loader, val_loader=None, verbose =True, device=device, start_epoch=0, val=True,show=True):
    '''
    This function trains the models resnet18 and pooling_resnet18

    '''

    model = model.to(device)

    training_loss = []
    train_auc = []
    train_acc = []
    val_auc = []
    val_acc = []

    # Loop over epochs
    for epoch in range(start_epoch, epochs+start_epoch): 
        
        model.train()

        auc = []
        target_list = []
        output_list = []
        epoch_train_loss = 0
        epoch_total_samples = 0

        # Loop over batches 
        for _ , (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Apply model
            output = model(data) 

            # Compute loss function
            loss = criterion(output, target)

            # Backpropagate the loss
            loss.backward()

            # Update optimizer
            optimizer.step()

            # Compute epoch loss
            epoch_train_loss += loss.item()
            
            # Compute output probabilities 
            m = nn.Softmax(dim=1)
            predicted_batch = m(output)

            epoch_total_samples += len(target)
            
            # Compute AUC 
            target_list += list(target.cpu().detach().numpy())
            output_list+= list(np.argmax(predicted_batch.cpu().detach().numpy(),axis=1))
            auc.append(roc_auc_score(target.cpu().detach().numpy(),predicted_batch.cpu().detach().numpy(), multi_class='ovo', labels=[0, 1, 2, 3, 4, 5],average='macro'))
        epoch_auc = np.mean(auc)

        # Compute balanced accuracy  
        epoch_acc = balanced_accuracy_score(target_list,output_list)

        train_auc.append(epoch_auc)
        train_acc.append(epoch_acc)
        average_epochloss = epoch_train_loss/len(train_loader)
        training_loss.append(average_epochloss)
        
        # Compute model performance on validation dataset
        if val : 
            validation_auc, validation_acc = test(model, criterion, device, val_loader, verbose=((epoch==epochs+start_epoch-1)and(show)))
            val_auc.append(validation_auc)
            val_acc.append(validation_acc)

        # Show training results
        if verbose :
            if val : 
                if epoch == 0 : 
                    print('Train Epoch    Training loss    Training AUC    Training Acc    Validation AUC    Validation Acc')
                    print('-----------    -------------    ------------    ------------    --------------    ---------------')
                if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
                    if epoch < 9 : 
                        print('     {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))
                    else: 
                        print('    {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))

            else : 
                if epoch == 0 : 
                    print('Train Epoch    Training loss    Training AUC    Training Acc')
                    print('-----------    -------------    ------------    ------------')
                if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
                    if epoch < 9 :
                        print('     {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))
                    else: 
                        print('    {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))

    return train_auc , training_loss , train_acc , val_auc , val_acc

# -------------------------------------------------------------------------------------------------- #

def train_augmented(working_dir,train_df,model, epochs, optimizer, criterion, verbose =True, device=device, start_epoch=0 , val = True, batch_size=20):
    '''
    This function trains the models resnet18 and pooling_resnet18 using data augmentation

    '''

    model = model.to(device)

    training_loss = []
    train_auc = []
    train_acc = []
    val_auc = []
    val_acc = []

    # Loop over epochs
    for epoch in range(start_epoch, epochs+start_epoch): 

        # Read augmented data
        X,y = read_data_augmented(working_dir,train_df)
        if val : 
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.3, stratify=y, shuffle = True )
            val_tiles = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val))
            val_loader = DataLoader(dataset=val_tiles, batch_size=batch_size, shuffle=True)
        
        train_tiles = TensorDataset(torch.tensor(X).float(), torch.tensor(y))
        train_loader = DataLoader(dataset=train_tiles, batch_size=batch_size, shuffle=True)

        epoch_train_loss = 0
        epoch_total_samples = 0
        model.train()
        
        auc = []
        target_list = []
        output_list = []

        # Loop over batches 
        for _ , (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Apply model
            output = model(data) 

            # Compute loss function
            loss = criterion(output, target)

            # Backpropagate the loss
            loss.backward()

            # Update optimizer
            optimizer.step()

            # Compute epoch loss
            epoch_train_loss += loss.item()

            # Compute output probabilities 
            m = nn.Softmax(dim=1)
            predicted_batch = m(output)

            epoch_total_samples += len(target)
            target_list += list(target.cpu().detach().numpy())
            output_list+= list(np.argmax(predicted_batch.cpu().detach().numpy(),axis=1))

            # Compute AUC 
            auc.append(roc_auc_score(target.cpu().detach().numpy(),predicted_batch.cpu().detach().numpy(), multi_class='ovo', labels=[0, 1, 2, 3, 4, 5],average='macro'))
        epoch_auc = np.mean(auc)

        # Compute balanced accuracy  
        epoch_acc = balanced_accuracy_score(target_list,output_list)
        
        train_auc.append(epoch_auc)
        train_acc.append(epoch_acc)
        average_epochloss = epoch_train_loss/len(train_loader)
        training_loss.append(average_epochloss)
        
        # Compute model performance on validation dataset
        if val : 
            validation_auc, validation_acc = test(model, criterion, device, val_loader, verbose=(epoch==epochs+start_epoch-1))
            val_auc.append(validation_auc)
            val_acc.append(validation_acc)
        
        # Show training results
        if verbose :
            if val : 
                if epoch == 0 : 
                    print('Train Epoch    Training loss    Training AUC    Training Acc    Validation AUC    Validation Acc')
                    print('-----------    -------------    ------------    ------------    --------------    ---------------')
                if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
                    if epoch < 9 : 
                        print('     {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))
                    else: 
                        print('    {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))

            else: 
                if epoch == 0 : 
                    print('Train Epoch    Training loss    Training AUC    Training Acc')
                    print('-----------    -------------    ------------    ------------')
                if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
                    if epoch < 9 :
                        print('     {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))
                    else: 
                        print('    {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))

    return train_auc , training_loss , train_acc , val_auc , val_acc

# -------------------------------------------------------------------------------------------------- #

def train_MIL(model, epochs,optimizer,scheduler, criterion, train_loader,val_loader, verbose =True, device=device, start_epoch=0 , val = True, batch_size=1,show=True):
  '''
    This function trains the MIL model

  '''

  model = model.to(device)

  training_loss = []
  train_auc = []
  train_acc = []
  val_auc = []
  val_acc = []

  # Loop over epochs
  for epoch in range(start_epoch, epochs+start_epoch): 

    epoch_train_loss = 0
    epoch_total_samples = 0   
    target_list = []
    output_list = []
    
    model.train()

    # Loop over batches 
    for _ , (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Apply model 
        output = model(data)
        
        # Compute loss function
        loss = criterion(output, target)   #.ordinal_regression(output, target)

        # Backpropagate the loss
        loss.backward()

        # Update optimizer
        optimizer.step()

        # Compute epoch loss
        epoch_train_loss += loss.item()
        epoch_total_samples += len(target)
        target_list.append(target.item())

        # Compute output probabilities 
        m = nn.Softmax(dim=1)
        output = m(output)
        output_list.append(output.data.cpu().numpy()[0])
    
    scheduler.step()

    # Compute AUC
    epoch_auc = roc_auc_score(target_list, output_list, multi_class='ovo', labels=[0, 1, 2, 3, 4, 5],average='macro')

    # Compute balanced accuracy  
    epoch_acc = balanced_accuracy_score(target_list, np.argmax(np.array(output_list), axis=1))
        
    train_auc.append(epoch_auc)
    train_acc.append(epoch_acc)
    average_epochloss = epoch_train_loss/len(train_loader)
    training_loss.append(average_epochloss)

    # Compute model performance on validation dataset
    if val : 
      validation_auc, validation_acc = test_MIL(model, criterion, device, val_loader, verbose=((epoch==epochs+start_epoch-1)and(show)))
      val_auc.append(validation_auc)
      val_acc.append(validation_acc)
    
    # Show training results
    if verbose :
      if val : 
        if epoch == 0 : 
          print('Train Epoch    Training loss    Training AUC    Training Acc    Validation AUC    Validation Acc')
          print('-----------    -------------    ------------    ------------    --------------    ---------------')
        if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
          if epoch < 9 : 
            print('     {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))
          else: 
            print('    {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))

      else: 
        if epoch == 0 : 
          print('Train Epoch    Training loss    Training AUC    Training Acc')
          print('-----------    -------------    ------------    ------------')
        if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
          if epoch < 9 :
            print('     {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))
          else: 
            print('    {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))

  return train_auc , training_loss , train_acc , val_auc , val_acc

# -------------------------------------------------------------------------------------------------- #

def train_MIL_augmented(working_dir,train_df,model, epochs,optimizer,scheduler, criterion, verbose =True, device=device, start_epoch=0 , val = True, batch_size=1,show=True):
  '''
    This function trains the MIL model with data augmentation

  '''

  model = model.to(device)
  
  training_loss = []
  train_auc = []
  train_acc = []
  val_auc = []
  val_acc = []

  # Read augmented  data 
  X,y = read_data_MIL(working_dir + 'train_clean_normalized', train_df)
  if val : 
      X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
      val_tiles = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val))
      val_loader = DataLoader(dataset=val_tiles, batch_size=batch_size, shuffle=True)
     
  train_tiles = TensorDataset(torch.tensor(X).float(), torch.tensor(y))
  train_loader = DataLoader(dataset=train_tiles, batch_size=batch_size, shuffle=True)

  # Loop over epochs
  for epoch in range(start_epoch, epochs+start_epoch): 

    epoch_train_loss = 0
    epoch_total_samples = 0   
    target_list = []
    output_list = []

    model.train()

    # Loop over batches 
    for _ , (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Apply model 
        output = model(data)
        
        # Compute loss function
        loss = criterion(output, target) #criterion.ordinal_regression(output, target)
        
        # Backpropagate the loss
        loss.backward()

        # Update optimizer
        optimizer.step()

        # Compute epoch loss
        epoch_train_loss += loss.item()
        epoch_total_samples += len(target)
        target_list.append(target.item())

        # Compute output probabilities 
        m = nn.Softmax(dim=1)
        output = m(output)
        output_list.append(output.data.cpu().numpy()[0])
    
    scheduler.step()

    # Compute AUC 
    epoch_auc = roc_auc_score(target_list, output_list, multi_class='ovo', labels=[0, 1, 2, 3, 4, 5],average='macro')

    # Compute balanced accuracy  
    epoch_acc = balanced_accuracy_score(target_list, np.argmax(np.array(output_list), axis=1))
        
    train_auc.append(epoch_auc)
    train_acc.append(epoch_acc)
    average_epochloss = epoch_train_loss/len(train_loader)
    training_loss.append(average_epochloss)

    # Compute model performance on validation dataset
    if val : 
      validation_auc, validation_acc = test_MIL(model, criterion, device, val_loader, verbose=((epoch==epochs+start_epoch-1)and(show)))
      val_auc.append(validation_auc)
      val_acc.append(validation_acc)
    
    # Show training results
    if verbose :
      if val : 
        if epoch == 0 : 
          print('Train Epoch    Training loss    Training AUC    Training Acc    Validation AUC    Validation Acc')
          print('-----------    -------------    ------------    ------------    --------------    ---------------')
        if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
          if epoch < 9 : 
            print('     {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))
          else: 
            print('    {}             {:.4f}          {:.4f}          {:.4f}           {:.4f}            {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc,validation_auc, validation_acc))

      else: 
        if epoch == 0 : 
          print('Train Epoch    Training loss    Training AUC    Training Acc')
          print('-----------    -------------    ------------    ------------')
        if epoch % 1 == 0 or epoch == epochs+start_epoch - 1 : 
          if epoch < 9 :
            print('     {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))
          else: 
            print('    {}             {:.4f}          {:.4f}          {:.4f}'.format(epoch+1, average_epochloss,epoch_auc, epoch_acc))

  return train_auc , training_loss , train_acc , val_auc , val_acc