import torch

def Splitter(data,splits,subject_no=0):
    if subject_no!=0:
        x_train=[data[i]['eeg'][:,20:460] for i in splits['splits'][0]['train'] if data[i]['subject']==subject_no]
        x_val=[data[i]['eeg'][:,20:460]  for i in splits['splits'][0]['val'] if data[i]['subject']==subject_no]
        x_test=[data[i]['eeg'][:,20:460] for i in splits['splits'][0]['test'] if data[i]['subject']==subject_no]
        
        y_train=[data[i]['label'] for i in splits['splits'][0]['train'] if data[i]['subject']==subject_no]
        y_val=[data[i]['label']  for i in splits['splits'][0]['val'] if data[i]['subject']==subject_no]
        y_test=[data[i]['label'] for i in splits['splits'][0]['test'] if data[i]['subject']==subject_no]
    
    else:
        x_train=[data[i]['eeg'][:,20:460] for i in splits['splits'][0]['train']]
        x_val=[data[i]['eeg'][:,20:460] for i in splits['splits'][0]['val']]
        x_test=[data[i]['eeg'][:,20:460] for i in splits['splits'][0]['test']]
        
        y_train=[data[i]['label'] for i in splits['splits'][0]['train']]
        y_val=[data[i]['label']  for i in splits['splits'][0]['val']]
        y_test=[data[i]['label'] for i in splits['splits'][0]['test']]
        
    return x_train,x_val,x_test,y_train,y_val,y_test

def train_model(model, train_loader, val_loader, criterion, optimizer,device,num_epochs=50):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate loss

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total_train += labels.size(0)  # Total number of labels
            correct_train += (predicted == labels).sum().item()  # Correct predictions

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                val_loss += loss.item()  # Accumulate validation loss

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class
                total_val += labels.size(0)  # Total number of labels
                correct_val += (predicted == labels).sum().item()  # Correct predictions

        # Calculate average losses and accuracies
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = correct_train / total_train * 100  # Convert to percentage
        val_accuracy = correct_val / total_val * 100  # Convert to percentage

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Training Accuracy: {train_accuracy:.2f}%, "
              f"Validation Accuracy: {val_accuracy:.2f}%")
        
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
